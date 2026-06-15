---
layout: ../../layouts/BlogPost.astro
title: "Writing FlashAttention in Triton (Part 1): From the Algorithm to a Real Kernel, and Fusing RoPE "
date: 2026-06-15
---

In [Part 1](https://navneetkanna.com/blog/fa_1/) we worked out *why* FlashAttention is fast, the memory wall, and the streaming-softmax trick that lets us read Q, K and V from HBM exactly once.

In this post we will see how to turn that algorithm into an actual Triton kernel running on a GPU. We'll write the forward pass, add causal masking, convince ourselves it's correct, then **fuse Rotary Positional Embeddings (RoPE) directly into the kernel**, including the bugs and the Triton-version wall I hit along the way. Finally we benchmark it against PyTorch.

---

## The Triton programming model

Triton lets you write a GPU kernel in Python. You don't think about individual threads; you think about **programs**, where each program processes one tile of the output. You launch a *grid* of programs, and each one figures out which tile it owns from its program id.

For attention, the natural unit of work is **one block of query rows, for one (batch, head)**. So the grid is two-dimensional:

```python
grid = (N // BLOCK_Q, B * H)
```

Inside the kernel, a program reads its coordinates and computes a base offset into the right (batch, head) slice:

```python
block_row = tl.program_id(0) # which block of query rows
batch_head_idx = tl.program_id(1) # which (batch, head)
offset = batch_head_idx * stride_q_h
```

The tiles from Part 1 become **block pointers**, a view that says "starting from this address, with this shape and these strides, hand me a `(BLOCK_Q, BLOCK_D)` chunk":

```python
q_block_ptr = tl.make_block_ptr(
    base=Q + offset,
    shape=(N, BLOCK_D),
    strides=(stride_q_n, stride_q_d),
    offsets=(block_row * BLOCK_Q, 0),
    block_shape=(BLOCK_Q, BLOCK_D),
    order=(1, 0),
)
q = tl.load(q_block_ptr, boundary_check=(0, 1))
```

This is the concrete form of "load a tile from HBM into SRAM". With $N=8$, $\text{BLOCK\_Q}=4$, the first program (`block_row=0`) loads query rows 0–3; the second (`block_row=1`) loads rows 4–7.

---

## The forward kernel

The query block is loaded once. Then we **stream over the K and V blocks**, and inside that loop we run exactly the recurrence from Part 1. Here's the heart of it:

```python
for start_kv in range(0, N, BLOCK_KV):
    k = tl.load(k_block_ptr, boundary_check=(0, 1))   # (BLOCK_D, BLOCK_KV), transposed
    v = tl.load(v_block_ptr, boundary_check=(0, 1))   # (BLOCK_KV, BLOCK_D)

    qk = tl.dot(q, k) * qk_scale                      # S = Q @ K.T / sqrt(d)  (one tile)

    # causal mask
    qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk, float("-inf"))

    # --- streaming softmax (Part 1) ---
    new_mi = tl.maximum(mi, tl.max(qk, axis=1))       # m_new = max(m_old, m_local)
    alpha = tl.math.exp2(mi - new_mi)                 # correction factor
    p = tl.math.exp2(qk - new_mi[:, None])            # unnormalised probs

    o_acc = o_acc * alpha[:, None] + tl.dot(p, v)     # O_new = O_old * alpha + P @ V
    mi = new_mi
    li = li * alpha + tl.sum(p, axis=1)               # d_new = d_old * alpha + d_local

    k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_KV))
    v_block_ptr = tl.advance(v_block_ptr, (BLOCK_KV, 0))

o_acc = o_acc / li[:, None]                            # the single final division
```

Line for line, this is Part 1: `mi` is $m$, `li` is $d$, `o_acc` is $O$, and `alpha` is the correction factor $e^{m_\text{old}-m_\text{new}}$ that rescales the running state whenever a new maximum appears. The only thing we never do is materialise the full $N\times N$ score matrix, each `qk` tile lives in SRAM and is discarded after it updates the three running variables.

One subtlety: `K` is loaded **transposed** (block shape `(BLOCK_D, BLOCK_KV)`), so that `tl.dot(q, k)` directly computes $Q K^\top$ without a separate transpose.

---


## Three details that make it real

The algorithm was explained in Part 1. Turning it into a kernel surfaces three details worth calling out, because each one is a small window into how the hardware actually works.

### 1. Why `exp2` instead of `exp`

You may have noticed `tl.math.exp2` instead of `exp`, and a scale that looks odd:

```python
qk_scale = scale * 1.44269504089 # scale * log2(e)
...
p = tl.math.exp2(qk - new_mi[:, None])
```

GPUs have a fast hardware instruction for $2^x$, but not for $e^x$. So instead of computing $e^{x}$, we compute $2^{x\log_2 e}$, which is identically equal, we just pre-fold the $\log_2 e = 1.4427\ldots$ factor into the scale. Its the same math, but cheaper instruction.

### 2. Causal masking, and a free optimisation

A causal model can't attend to the future, so a query at position $i$ may only see keys at positions $j \le i$. We enforce it by setting the disallowed scores to $-\infty$ before the softmax:

```python
qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk, float("-inf"))
```

If a whole K block sits entirely in the future of the current query block, every one of its scores becomes $-\infty$ and contributes nothing. So we need not even load it. Stopping the loop at the query block's own diagonal gives, for the later query blocks, roughly half the work:

```python
kv_end = (block_row + 1) * BLOCK_Q
for start_kv in range(0, kv_end, BLOCK_KV):
    ...
```

### 3. The 16-wide rule for `tl.dot`

`tl.dot` runs on the GPU's **tensor cores**, which only multiply fixed-size tiles, the shared inner dimension must be at least 16.

---

## Correctness before speed

It is important to test your logic against pytorch

```python
out = flash_attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)
```

A tip from experience: a logic bug often hides under the high tolerance.

---

## Fusing RoPE into the kernel

### RoPE in two paragraphs

Attention is blind to word order, $Q K^\top$ doesn't know where tokens sit. RoPE fixes this not by *adding* a position vector, but by **rotating** each query and key by an angle proportional to its position. Split a vector into pairs of coordinates; pair $i$ at position $p$ is rotated by angle $p\cdot\theta_i$, where the per-pair speeds $\theta_i$ run from fast to slow.

The payoff: when a rotated query at position $m$ meets a rotated key at position $n$, their dot product depends only on $m-n$. Absolute positions cancel. And RoPE only ever touches $Q$ and $K$ — never $V$ — and only affects the *scores*. So in the kernel it sits in right before `tl.dot(q, k)`, and nothing downstream changes.

### Where it goes

There are two honest ways to add RoPE:

1. **Rotate outside the kernel.** Apply RoPE to $Q$ and $K$ in plain PyTorch, then feed the rotated tensors into the unchanged attention kernel.
2. **Fuse it.** Rotate inside the kernel, in SRAM, so you never write rotated $Q$/$K$ back to HBM.
Fusing trades a little redundant compute for saved memory traffic. Because each query block streams over all K blocks, a fused kernel re-rotates each K block once per query block, its redundant work, but it avoids a full round-trip of rotated $Q$/$K$ through HBM. Whether that's a win is an empirical question, which is exactly what the benchmark below address.
