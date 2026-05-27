---
layout: ../../layouts/BlogPost.astro
title: "Writing FlashAttention in Triton (Part 1): The Memory Wall and the Online Softmax Trick"
date: 2026-05-26
---

Before we dive into this, it is important that you have understood the attention mechanism first, you can read my [blog anout it](https://navneetkanna.com/blog/transformers1/).


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

Now we sum each row

$$
Row 1 sum: 11.21
Row 2 sum: 24.43
$$

Next we have to divide each exponent by its row sum to get probabilities

$
\begin{bmatrix}
0.65 & 0.24 & 0.04 \\
0.11 & 0.82 & 0.06
\end{bmatrix}
$$

Notice how the highest score in each row (2.0 in the first row, 3.0 in the second) becomes the highest probability (0.659 and 0.821). The sum of each row is exactly 1.
