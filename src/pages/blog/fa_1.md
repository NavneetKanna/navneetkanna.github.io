---
layout: ../../layouts/BlogPost.astro
title: "Writing FlashAttention in Triton (Part 1): The Memory Wall and the Online Softmax Trick"
date: 2026-05-26
---

Before we dive into this, it is important that you have understood the attention mechanism first, you can read my [blog about it](https://navneetkanna.com/blog/transformers1/).

## Flash Attention

First lets clear up on some jargon:

1. SRAM (Static RAM) - is the fastest memory in the hierarchy, it is built directly onto the SM die.
2. VRAM (Video RAM) or HBM (High Bandwidth Memory) - this is the value that `nvidia-smi` shows. It's stacked DRAM dies right next to the GPU die, giving very short, wide interconnects.

Now, the main speedup that comes when using flash attention is when we avoid all the intermediary memory transfers between the HBM and the SM when processing attention. So lets see the naive attention
mechanism and the memory transfers it requires:

$$

X = { Q K^T \over \sqrt{d} } \\[1em]

Y = softmax(X) \\[1em]

O = Y V
$$

1. First, $Q$ and $K$ are loaded from the HBM and $X$ is computed. After the computation, $X$ is written back to HBM.
2. Now, $X$ is loaded back again from the HBM and softmax is computed after which $Y$ is written back to the HBM.
3. Again, $Y$ is loaded back from HBM as well as $V$ and $O$ is computed and written back to HBM.

As it can be seen, there are too many unnecessary reads and writes which slows down the process. And you can imagine it for huge matrices, multiple heads, multiple blocks these reads and writes affects the
overall speed. The GPU's compute units can do arithmetic far faster than HBM can supply data. Naive attention is bottlenecked by bandwidth, not by the matmuls.

For a sequence of length n, the S matrix alone requires $O(n^2)$ memory. At n = 8192 with fp16, that's ~134MB just for attention scores, per head, per layer. Flash Attention reduces memory to $O(n)$
and more importantly, Q, K, and V are each read from HBM exactly once.

The way Flash Attention achieves this is by fusing the softmax and the output multiplication into a single pass. To understand how, we first need to look at streaming softmax.

### Streaming Softmax

The main idea is to do softmax in tiles. The way this works is:

There are 2 variables that are initialized: $m_{old} = -inf$ and $d_{old} = 0$, where $m$ is the running maximum and $d$ the running denominator.

More specifically

1. Find the Local Max: Find the maximum value within just this tile ($m_{local}$).
2. Update the Global Max: Figure out the new overall maximum:
$$
m_{new} = \max(m_{old}, m_{local})
$$
3. Compute the Local Denominator: Calculate the sum of exponentials for just this tile, using the new global max to keep numbers stable:
$$
d_{local} = \sum_{x \in \text{tile}} e^{x - m_{new}}
$$
4. Update the Global Denominator: Scale the old global denominator using the correction factor, then add the local denominator:
$$
d_{new} = d_{old} \cdot e^{m_{old} - m_{new}} + d_{local}
$$

The trick here is the correction factor $e^{m_{old} - m_{new}}$. Whenever we hit a new maximum value, this factor scales down the previously accumulated denominator. It mathematically adjusts the old sum
so it acts as if we had known the new global maximum from the very beginning.

Lets take 1 row of a matrix $[1, 2, 3, 4]$ with tile size 2. 

Lets see how its done normally

Pass 1: Find the max of the full row
  - Load all the values from VRAM and calculate the max.

Pass 2: Compute Exponentials & Sum
  - Load the values again from VRAM and calculate the exponentials and the sum of it.
  - After calculating the exponentials we need to store them back to VRAM.

Pass 3: Final Division
  - Now we need to load those exponentials from VRAM and divide by the row sum.

Now lets see how it is done in streaming version

Pass 1: Streaming the Tiles

Processing Tile 1: [1, 2].
  - Load [1, 2] from VRAM into registers/SRAM.
  - Local Max: $m_{local} = \max(1, 2) = 2$
  - New Global Max: $m_{new} = \max(-\infty, 2) = \mathbf{2}$
  - Local Denominator: $d_{local} = e^{1 - 2} + e^{2 - 2} = e^{-1} + 1 \approx 0.367 + 1 = 1.367$
  - New Global Denominator: $d_{new} = 0 \cdot e^{-\infty - 2} + 1.367 = \mathbf{1.367}$
  - Current State: $m = 2, d = 1.367$

Processing Tile 2: [3, 4].
  - Load [3, 4] into registers.
  - Local Max: $m_{local} = \max(3, 4) = 4$
  - New Global Max: $m_{new} = \max(2, 4) = \mathbf{4}$
  - Local Denominator: $d_{local} = e^{3 - 4} + e^{4 - 4} = e^{-1} + 1 \approx \mathbf{1.367}$
  - New Global Denominator: Here is where the magic happens. We scale the old denominator ($1.367$) by the difference between the old max ($2$) and the new max ($4$).

$$
d_{new} = 1.367 \cdot e^{2 - 4} + 1.367 \\[1em]
d_{new} = 1.367 \cdot (0.135) + 1.367 \\[1em]
d_{new} = 0.185 + 1.367 = \mathbf{1.552}
$$

Final Output State: $m = 4, d = 1.552$. $1.552$ is the exact same global denominator we got in the element-by-element example. The math perfectly guarantees that chunking the data doesn't change the
final answer.

Pass 2: Computing the Probabilities

Now that we have our true global max ($4$) and global denominator ($1.552$), we do our second pass over the tiles to compute and write the final probabilities.
  - Load Tile 1 [1, 2]: Compute $(e^{1-4}/1.552)$ and $(e^{2-4}/1.552)$. Write [0.03, 0.09] to VRAM.
  - Load Tile 2 [3, 4]: Compute $(e^{3-4}/1.552)$ and $(e^{4-4}/1.552)$. Write [0.24, 0.64] to VRAM.

So far we have discussed streaming softmax, now lets see how flash attention incorporates it. Along with the two running variables, flash attention maintains one more, which is the output $O$ (this is the
last step of the attention process, see above).

Steps 1, 2 and 3 are the same. Now, there is an additional step

The values in $O_{old}$ accumulator were multiplied by exponentials using the old maximum. We can fix the entire matrix block of $O$ using the exact same scalar correction factor!
We scale the old $O$ matrix down, and then add the new block's contribution:

$$
O_{new} = O_{old} \cdot e^{m_{old} - m_{new}} + \sum_{j \in \text{tile}} e^{S_j - m_{new}} \cdot V_j
$$

where $S_{local} = Q \times K_{local}^T$.

Lets see how it works with the same example, but this time we need the value matrix and $S$.

$S$: [2, 3, 5, 4] \\[1em]
$V$: [10, 20, 30, 40]


Tile 1

Load the first block from VRAM into fast memory: $S = [2, 3]$ and $V = [10, 20]$.

1. Find New Max:

$$
m_{local} = 3 \\[1em]
m_{new} = \max(-\infty, 3) = \mathbf{3}
$$

2. Update Denominator ($d$): 

$$
d_{new} = d_{old} \cdot e^{-\infty - 3} + (e^{2-3} + e^{3-3}) \\[1em]
d_{new} = 0 + (e^{-1} + 1) \\[1em]
d_{new} \approx 0.368 + 1 = \mathbf{1.368}
$$

3. Update Output Accumulator ($O$): Instead of dividing by $d$, we just multiply the exponentials directly against the Values.

$$
O_{new} = O_{old} \cdot e^{-\infty - 3} + (e^{2-3} \cdot 10 + e^{3-3} \cdot 20) \\[1em]
O_{new} = 0 + (0.368 \cdot 10 + 1 \cdot 20) \\[1em]
O_{new} = 3.68 + 20 = \mathbf{23.68}
$$

Tile 1 is done. It gets dumped from SRAM. Current hardware state: $m = 3, d = 1.368, O = 23.68$.

Tile 2

Load the next block: $S = [5, 4]$ and $V = [30, 40]$.

1. Find New Max:

$$
m_{local} = 5 \\[1em]
m_{new} = \max(3, 5) = \mathbf{5}
$$

Because the global max just changed from $3$ to $5$, we need to calculate our correction factor: $e^{3 - 5} = e^{-2} \approx \mathbf{0.135}$.

2. Update Denominator ($d$):

We apply the correction factor to fix the old denominator, then add the new tile's sum.

$$
d_{new} = (d_{old} \cdot 0.135) + (e^{5-5} + e^{4-5}) \\[1em]
d_{new} = (1.368 \cdot 0.135) + (1 + 0.368) \\[1em]
d_{new} = 0.185 + 1.368 = \mathbf{1.553}
$$

3. Update Output Accumulator ($O$):

We apply the exact same correction factor to fix our running $O$ matrix, then add the new tile's unnormalized matrix multiplication.

$$
O_{new} = (O_{old} \cdot 0.135) + (e^{5-5} \cdot 30 + e^{4-5} \cdot 40) \\[1em]
O_{new} = (23.68 \cdot 0.135) + (1 \cdot 30 + 0.368 \cdot 40) \\[1em]
O_{new} = 3.20 + (30 + 14.72) \\[1em]
O_{new} = 3.20 + 44.72 = \mathbf{47.92}
$$

Tile 2 is done. We have reached the end of the sequence. Final hardware state: $m = 5, d = 1.553, O = 47.92$.

The Final Division

We streamed through the entire sequence without ever writing a single intermediate probability or exponential to main memory. We just maintained three variables in our fast registers.
To get the final, true attention output, we do one division right before we write to VRAM:

$$
\text{Final Output} = \frac{O}{d} = \frac{47.92}{1.553} = \mathbf{30.86}
$$

For a sequence of length n, the $S$ matrix alone requires $O(n^2)$ memory At n = 8192 and fp16, that's ~134MB just for attention scores, per head, per layer. Flash Attention reduces memory to O(n).
We streamed through all tiles in a single pass: Q, K, and V are each read from HBM exactly once.
