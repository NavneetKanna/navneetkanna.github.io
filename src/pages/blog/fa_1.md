---
layout: ../../layouts/BlogPost.astro
title: "Writing FlashAttention in Triton (Part 1): The Memory Wall and the Online Softmax Trick"
date: 2026-05-26
---

Before we dive into this, it is important that you have understood the attention mechanism first, you can read my [blog about it](https://navneetkanna.com/blog/transformers1/).


### Softmax

Suppose we have this matrix

$$
\begin{bmatrix}
2.0 & 1.0 & 0.1 \\
1.0 & 3.0 & 0.5
\end{bmatrix}
$$

We take the exponent of each value

$$
\begin{bmatrix}
7.38 & 2.71 & 1.1 \\
2.71 & 20.08 & 1.64
\end{bmatrix}
$$

Now we sum each row `Row 1 sum: 11.21` and `Row 2 sum: 24.43`.

Next we have to divide each exponent by its row sum to get probabilities

$$
\begin{bmatrix}
0.65 & 0.24 & 0.04 \\
0.11 & 0.82 & 0.06
\end{bmatrix}
$$

Notice how the highest score in each row (2.0 in the first row, 3.0 in the second) becomes the highest probability (0.65 and 0.82). The sum of each row is exactly 1.

## Flash Attention

First lets clear up on some jargon:

1. SRAM (Static RAM) - is the fastest memory in the hierarchy, it is built directly onto the SM die.
2. VRAM (Video RAM) or HBM (High Bandwidth Memory) - this is the value that `nvidia-smi` shows. It's stacked DRAM dies right next to the GPU die, giving very short, wide interconnects.

Now, the main speedup that comes when using flash attention is when we avoid all the intermediary memory transfers between the HBM and the SM when processing attention. So lets see the navie attention
mechanism and the memory transfers it requires:

$$

X = { Q K^T \over \sqrt{d} } \\[1em]

Y = softmax(X) \\[1em]

O = Y V
$$

1. First, $Q$ and $K$ are loaded from the HBM and $X$ is computed. After the computation, $X$ is written back to HBM.
2. Now, $X$ is loaded back again from the HBM and softmax is computed after which $Y$ is witten back to the HBM.
3. Again, $Y$ is loaded back from HBM as well as $V$ and $O$ is computed and written back to HBM.


As it can be seen, there are too many unnecessary reads and writes which slows down the process. And you can imagine it for huge matrices, multiple heads, multiple blocks these reads and writes affects the
overall speed.

The way flash attention solves this is by tiling and fusing these two steps into one

$$
Y = softmax(X) \\[1em]
O = Y V
$$

First we need to understand a variant of softmax called online/streaming softmax.

Lets take 1 row of a matrix $[1, 2, 3, 4]$ with tile size 2. The way this works is:

There are 2 variables that are initialized: $m_{old} = -inf$ and $d_{old} = 0$, where $m$ is the running maximum and $d$ the running denominator. For each element, we update the state using 

$$
m_i = max(m_{i-1}, x_i) \\[1em]

d_i = d_{i-1} e^{m_{i-1}-m_i} + e^{x_i - m_i}
$$

More specifically

1. Find the Local Max: Find the maximum value within just this tile ($m_{local}$).
2. Update the Global Max: Figure out the new overall maximum:$$m_{new} = \max(m_{old}, m_{local})$$
3. Compute the Local Denominator: Calculate the sum of exponentials for just this tile, using the new global max to keep numbers stable:$$d_{local} = \sum_{x \in \text{tile}} e^{x - m_{new}}$$
4. Update the Global Denominator: Scale the old global denominator using the correction factor, then add the local denominator:$$d_{new} = d_{old} \cdot e^{m_{old} - m_{new}} + d_{local}$$

The trick here is the correction factor $e^{m_{i-1} - m_i}$. Whenever we hit a new maximum value, this factor scales down the previously accumulated denominator. It mathematically adjusts the old sum
so it acts as if we had known the new global maximum from the very beginning.

Pass 1: Streaming the Tiles
Processing Tile 1: [1, 2]. Load [1, 2] from VRAM into registers/SRAM.
  - Local Max: $m_{local} = \max(1, 2) = 2$
  - New Global Max: $m_{new} = \max(-\infty, 2) = \mathbf{2}$
  - Local Denom: $d_{local} = e^{1 - 2} + e^{2 - 2} = e^{-1} + 1 \approx 0.367 + 1 = 1.367$
  - New Global Denom: $d_{new} = 0 \cdot e^{-\infty - 2} + 1.367 = \mathbf{1.367}$
  - Current State: $m = 2, d = 1.367$
Processing Tile 2: [3, 4]. Load [3, 4] into registers.
  - Local Max: $m_{local} = \max(3, 4) = 4$
  - New Global Max: $m_{new} = \max(2, 4) = \mathbf{4}$
  - Local Denom: $d_{local} = e^{3 - 4} + e^{4 - 4} = e^{-1} + 1 \approx \mathbf{1.367}$
  - New Global Denom: Here is where the magic happens. We scale the old denom ($1.367$) by the difference between the old max ($2$) and the new max ($4$).

$$
d_{new} = 1.367 \cdot e^{2 - 4} + 1.367 \\[1em]
d_{new} = 1.367 \cdot (0.135) + 1.367 \\[1em]
d_{new} = 0.185 + 1.367 = \mathbf{1.552}
$$

Final Output State: $m = 4, d = 1.552$. $1.552$ is the exact same global denominator we got in the element-by-element example. The math perfectly guarantees that chunking the data doesn't change the
final answer.

Pass 2: Computing the Probabilities
Now that we have our true global max ($4$) and global denom ($1.552$), we do our second pass over the tiles to compute and write the final probabilities.
  - Load Tile 1 [1, 2]: Compute $(e^{1-4}/1.552)$ and $(e^{2-4}/1.552)$. Write [0.03, 0.09] to VRAM.
  - Load Tile 2 [3, 4]: Compute $(e^{3-4}/1.552)$ and $(e^{4-4}/1.552)$. Write [0.24, 0.64] to VRAM.

