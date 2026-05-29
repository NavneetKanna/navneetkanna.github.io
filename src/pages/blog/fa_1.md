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

$$Y = softmax(X)$ and $O = Y V$$
