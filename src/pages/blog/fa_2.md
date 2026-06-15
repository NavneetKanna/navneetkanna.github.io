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

    qk = tl.dot(q, k) * qk_scale                      # S = Q K.T / sqrt(d)  (one tile)

    # causal mask
    qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk, float("-inf"))

    # --- streaming softmax (Part 1) ---
    new_mi = tl.maximum(mi, tl.max(qk, axis=1))       # m_new = max(m_old, m_local)
    alpha = tl.math.exp2(mi - new_mi)                 # correction factor
    p = tl.math.exp2(qk - new_mi[:, None])            # unnormalised probs

    o_acc = o_acc * alpha[:, None] + tl.dot(p, v)     # O_new = O_old . alpha + P @ V
    mi = new_mi
    li = li * alpha + tl.sum(p, axis=1)               # d_new = d_old . alpha + d_local

    k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_KV))
    v_block_ptr = tl.advance(v_block_ptr, (BLOCK_KV, 0))

o_acc = o_acc / li[:, None]                            # the single final division
```

Line for line, this is Part 1: `mi` is $m$, `li` is $d$, `o_acc` is $O$, and `alpha` is the correction factor $e^{m_\text{old}-m_\text{new}}$ that rescales the running state whenever a new maximum appears. The only thing we never do is materialise the full $N\times N$ score matrix — each `qk` tile lives in SRAM and is discarded after it updates the three running variables.

One subtlety: `K` is loaded **transposed** (block shape `(BLOCK_D, BLOCK_KV)`), so that `tl.dot(q, k)` directly computes $Q K^\top$ without a separate transpose.

---
