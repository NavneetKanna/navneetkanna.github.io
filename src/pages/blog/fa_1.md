---
layout: ../../layouts/BlogPost.astro
title: "Writing FlashAttention in Triton (Part 1): The Memory Wall and the Online Softmax Trick"
date: 2026-05-26
---

Before we dive into this, it is important that you have understood the attention mechanism first, you can read my [blog anout it](https://navneetkanna.com/blog/transformers1/).


### Softmax


$$
\begin{bmatrix}
2.0 & 1.0 & 0.1 \\
1.0 & 3.0 & 0.5
\end{bmatrix}
$$



