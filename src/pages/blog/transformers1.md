---
layout: ../../layouts/BlogPost.astro
title: "Transformers (Decoder-Only) (Part 1)"
date: 2025-06-18
---

In the post, I will try to explain the transformer architecture (decoder-only) from scratch, so lets begin. 

### High level overview

1. Given a prompt, it is first tokenized and embedded with position. This flows into a block.
2. There are multiple blocks. Each block consists of Multi-Head Attention layers, followed by layer norms, followed by feed-forward network.
3. After the blocks, there is a linear and softmax linear, which outputs probabilites of the token should come next.


### Input data

Lets come back to encoding the input later, but lets assume after embedding with position, the input is as follows,

```python
inp = (2, 6, 4)

"""
where,
    2 is the batch size
    6 is the context length
    4 is the embedding dimension
"""

"""
Let this be the dataset

The sun dipped below the horizon, painting the sky with hues of orange and pink.
A gentle breeze rustled the leaves, creating a soothing melody.
In that peaceful moment, the world seemed to pause and breathe.
"""

# Lets assume the input is as follows

inp = [
    [
        [0.4553, 0.3277, 0.4210, 0.6628],   # The
        [0.0874, 0.3216, 0.2850, 0.1438],   # sun
        [0.8880, 0.2221, 0.6271, 0.8234],   # dipped
        [0.9180, 0.8070, 0.4281, 0.1977],   # below
        [0.4874, 0.9018, 0.4258, 0.1630],   # the
        [0.0441, 0.1988, 0.6751, 0.8757],   # horizon
    ],
    [
        [0.9347, 0.9255, 0.6341, 0.0567],   # ,
        [0.7656, 0.2911, 0.6161, 0.0123],   # painting
        [0.4200, 0.3295, 0.1863, 0.7694],   # the
        [0.3197, 0.9724, 0.6066, 0.2184],   # sky
        [0.4521, 0.9276, 0.3951, 0.1281],   # with
        [0.7899, 0.7894, 0.2245, 0.2889]    # hues
    ]
]

"""
We have 2 batches, each with 6 rows (these are words from the dataset, 
since that is our context length) and 4 columns (the length of our 
embedding dimension).
"""
```

This input now gets fed into individual blocks which are independent of each other.

### Block

The block consists of multiple attention heads, layer normalization layers, and a feed-forward layer.

1. The input is first passed through a layer normalization layer. (In the original paper, layer normalization is applied after the attention heads, but recently, it has become more common to apply it before the attention heads.)
2. The normalized input is then fed into the individual attention heads.
3. The outputs from the multi-head attention are added to the original input.
4. The result of this addition is passed through another layer normalization layer.
5. The normalized output is then fed into the feed-forward layer.
6. Finally, the output of the feed-forward layer is added to the output from step 3.

<div style="text-align:center;">
  <img src="/assets/images/block.svg" alt="block" style="display:inline-block;">
</div>

#### Multi-Attention heads

1. The input is fed into multiple attention heads in parallel.
2. The outputs from each attention head are concatenated.
3. The concatenated output is then passed through a linear layer.

<div style="text-align:center;">
  <img src="/assets/images/mha.svg" alt="mha" style="display:inline-block;">
</div>

#### Single Attention head (Attention mechanism)

First lets understand what is the goal of attention mechanism, what does it try to achieve,

1. Decide what parts of the input sequence (the context length) to focus on when processing or generating next token.

For our example, lets take the first input sequence, ```The sun dipped below the horizon``` (6 is our context length). Let's say we are predicting the last word ```The sun dipped below the ____```, to predict this word, it needs to learn that the word ```sun``` or ```dipped``` is more important or relevant compared to others, and give more attention towards those words.

To accomplish this, there are 3 vectors that are used, query, key and value. The intuition is as follows

| Name          | Intuition                 |
| ------------- | -------------------------- |
| **Query (Q)** | The current token in question |
| **Key (K)**   | Each token's relevance wrt the query token |
| **Value (V)** | The actual representation of the tokens |

Each token in the sequence has got all 3 vectors associated with them. The way they are derived is by shifting or projecting them from embedding space into a query, key and value space using a linear transformation ```nn.linear(bias=False)```. The weights associated with this linear layer are learnt during training. In other words, the model tries to learn a good weight matrix that can transform the input embedding into reasonable representations of the query, key and value space for the given dataset. The reason this is done is because, say the word apple is used in a sentence, based on the context, we can tell if the word apple is referring to the fruit or the company, however, in the embedding space, the word apple has got 1 fixed representation.

We get the query, key and value matrix like so

```python
query = nn.Linear(n_embd, head_size, bias=False)
key = nn.Linear(n_embd, head_size, bias=False)
value = nn.Linear(n_embd, head_size, bias=False)
```

where ```n_embd``` is 4. ```head_size``` is calculated as follows

```python
head_size = n_embd // n_head
```

```n_head``` is a parameter we can choose based on the embedding dimension, for example, if the embedding dimension is 4, we can set ```n_head``` to 2, so that ```head_size``` is 2. Therefore,

```python
query = nn.Linear(4, 2, bias=False)     # query.weight.shape = (2, 4)
key = nn.Linear(4, 2, bias=False)       # key.weight.shape = (2, 4)
value = nn.Linear(4, 2, bias=False)     # value.weight.shape = (2, 4)
```

The input that is fed into each head remains the same, ie, (2, 6, 4). So lets feed this into the attention head and see what happens

```python
inp = (2, 6, 4)

# first, we get the respective query, key and value matrix by projecting 
# them into the new space
q = query(inp)          # inp @ query.weight.T = (2, 6, 4) @ (4, 2) = (2, 6, 2)
k = key(inp)            # inp @ key.weight.T = (2, 6, 4) @ (4, 2) = (2, 6, 2)
v = value(inp)          # inp @ value.weigth.T = (2, 6, 4) @ (4, 2) = (2, 6, 2)

# second, we take the dot product (matmul) between the queries and keys
r = q @ k.transpose(-2, -1)   # (2, 6, 2) @ (2, 2, 6) = (2, 6, 6)

# third, we scale it by the square root of the head_size
# (hence the name scaled dot-product attention)
r = r * k.shape[-1]**-0.5     # (2, 6, 6)

# fourth, we apply causal mask
tril = torch.tril(torch.ones(6, 6))
# (2, 6, 6)
r = r.masked_fill(tril[:inp.shape[1], :inp.shape[1]] == 0, float('-inf'))

# fifth, we normalize it using softmax
r = F.softmax(r, dim=-1)    # (2, 6, 6)

# sixth, perform weighted sum wrt the values
out = r @ v     # (2, 6, 6) @ (2, 6, 2) = (2, 6, 2)
```
<div style="text-align:center;">
  <img src="/assets/images/sha.svg" alt="mha" style="display:inline-block;">
</div>

Lets see what is happening with 1 token say ```horizon```, 

```python
# (2, 6, 4)
inp = [
    [
        [],                                   # the
        [],                                   # sun
        [],                                   # dipped
        [],                                   # below
        [],                                   # the 
        [0.0441, 0.1988, 0.6751, 0.8757],     # horizon
    ], 
    [
        ...
    ]
]

# q = query(inp)
# q = (2, 6, 2) -> each head
q = [
    [
        [],                   # the
        [],                   # sun
        [],                   # dipped
        [],                   # below
        [],                   # the 
        [0.9100, 0.3448],     # horizon
    ], 
    [
        ...
    ]
]

# k = key(inp)
# k = (2, 6, 2) -> each head
k = [
    [
        [0.0921, 0.9907],   # the
        [0.5637, 0.7303],   # sun
        [0.1860, 0.4071],   # dipped
        [0.8067, 0.1776],   # below
        [0.7002, 0.6632],   # the
        [0.9094, 0.3594]    # horizon
    ], 
    [
        ...
    ]
]

# v = value(inp)
# v = (2, 6, 2) -> each head
v = [
    [
        [0.5637, 0.4056],     # the
        [0.9803, 0.0100],     # sun
        [0.4111, 0.3980],     # dipped
        [0.6882, 0.9797],     # below
        [0.5551, 0.7583],     # the 
        [0.3060, 0.2141],     # horizon
    ], 
    [
        ...
    ]
]
```

The token ```horizon``` has now shifted/projected to a new query, key and value space. The query matrix as the name suggests, is trying to query other tokens in the sequence and ask each of them which one of you are relevant to me ? The key matrix contains the answer to this question. Remember that these are all vectors in n-dim space, when we take a dot product between 2 vectors, it signifies how close those 2 vectors are or in other words if they point in the same direction. So when we take the dot product between the query and key vectors, the scalar output tells us how much one token in the sequence (key) is related to the token in question (query).

Lets remove the BS dim to make things simpler, so now we have

```python
# (6, 2)
q = [
        [],                   # the
        [],                   # sun
        [],                   # dipped
        [],                   # below
        [],                   # the 
        [0.9100, 0.3448],     # horizon
    ]

# (2, 6)
k.T = [
    # the   # sun   # dipped # below # the   # horizon
    [0.0921, 0.5637, 0.1860, 0.8067, 0.7002, 0.9094],
    [0.9907, 0.7303, 0.4071, 0.1776, 0.6632, 0.3594]
] 

# now when we do q @ k.T, we are taking the dot product between 
# the horizon token query vector and all other tokens key vectors

# r (6, 6) = q @ k.T (6, 2) @ (2, 6)
"""         
          the     sun    dipped   below    the   horizon
the          
sun
dipped
below
the
horizon  0.4254, 0.7648,  0.3096, 0.7953, 0.8659, 0.9515
"""
```

The (6, 6) matrix we get tells us how much each token in the sequence is relevant to the query token (horizon in this case). In other words how much the query and key vector point in the same direction.

``` python
r = r * k.shape[-1]**-0.5
```

We now divide the matrix by square root of `head_size`, this is done to because if `head_size` is large then the dot product values become large, therefore, we scale them by the square root of `head_size`. It also helps in the next step, when we perform softmax.

Now we apply masking to the attention matrix. This means the current token can only look and learn from itself and the tokens that comes before it, not after it. This makes sense, since we are trying to predict the next token in the sequence. This is done by masking out the upper triangle of the matrix.

```python
"""
          the     sun    dipped   below    the   horizon
the              -inf     -inf    -inf    -inf    -inf
sun                       -inf    -inf    -inf    -inf  
dipped                            -inf    -inf    -inf
below                                     -inf    -inf
the                                               -inf
horizon  0.4254  0.7648  0.3096  0.7953  0.8659  0.9515
"""

# (2, 6, 6)
tril = torch.tril(torch.ones(6, 6))
r = r.masked_fill(tril[:inp.shape[1], :inp.shape[1]] == 0, float('-inf'))
```

Next we apply softmax along the last dim to convert the raw attention scores into a probability distribution so that all rows sum to 1. This will be useful in the next step when we want to weight each value vector.

```python
r = F.softmax(r, dim=-1)

# it can be seen that, higher attention scores get higher values 
# and all of them sum to 1
"""
          the     sun    dipped   below    the   horizon
the                0       0        0       0       0
sun                        0        0       0       0
dipped                              0       0       0
below                                       0       0
the                                                 0
horizon  0.1252  0.1758  0.1115  0.1812  0.1945  0.2119
"""
```

This matrix tells us how much how much weightage we need to give to other tokens wrt the query token `horizon`, or in the other words, it tells us how much attention `horizon` token should pay to every other token. To do this we can just matmul `r` with the `value` matrix, remember that the `value` matrix contains the actual content of the sequence,

```python
# (6, 2) = (6, 6) @ (6, 2)
out = r @ v

"""

          the     sun    dipped   below    the   horizon
the                0       0        0       0       0
sun                        0        0       0       0
dipped                              0       0       0
below                                       0       0
the                                                 0
horizon  0.1252  0.1758  0.1115  0.1812  0.1945  0.2119

@

[
    [0.5637, 0.4056],     # the
    [0.9803, 0.0100],     # sun
    [0.4111, 0.3980],     # dipped
    [0.6882, 0.9797],     # below
    [0.5551, 0.7583],     # the 
    [0.3060, 0.2141],     # horizon
]

out

[
    []                # the
    []                # sun
    []                # dipped
    []                # below
    []                # the
    [0.5863, 0.4673]  # horizon
]
"""
```

Therefore, the final output blends together context from all tokens, weighted by their relevance.

Great, so remember that all this is done for a single head, but there are multiple heads that run in parallel, once the attention mechanisim is complete, we concatenate them along the last dim

```python
out = torch.cat([h(x) for h in heads], dim=-1)

"""
In our example, we have 2 heads, and as we have seen in the previous step, 
the output of a single head is of shape (2, 6, 2), therefore when we concatenate 
2 heads along the last dim, we get the final output shape as (2, 6, 4).
"""
```

The output now gets passed to a linear layer

```python
# if we have divided the embedding dimension equally, then the linear layer 
# can just be (n_emdb, n_embd)
# nn.Linear(2*2, 4)
proj = nn.Linear(head_size * num_heads, n_embd) 
# (2, 6, 4) = (2, 6, 4) @ (4, 4)
out = self.proj(out)
```

Each head captures or learn different aspects of the data, for example, one head might learn grammer context, and another head might learn time context, etc. When we concatenate them, we are just stacking them, but to improve the learning a linear layer is used.

Next, the output is added with the input. This is called as residual connections or skip connections. Basically, these help in optimization during backpropogation, because, in very deep networks such as transformers, the gradient can become very small as they flow backward which hinders the learning process. One way to solve this is by adding the input to the output after some layers. We know that the addition node in an autograd graph distributes the gradient equally to both of its input, hence keeping the gradient alive till it reaches the input.


```python
# therefore we finally have
# (2, 6, 4) = (2, 6, 4) + (2, 6, 4)
x = x + self_attention(layer_norm1(x))
```

Now the output goes through another layernorm and a feed-forward network. The feed-forward network is a simple network with the hidden layer size being 4 times the embedding dimension. The expression 4 times the embeddding dimension comes from the original paper.

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

ffwd = FeedFoward(4)
# (2, 6, 4) + (((2, 6, 4) @ (4, 16)) @ (16, 4))
# (2, 6, 4) + ((2, 6, 16) @ (16, 4))
# (2, 6, 4) + ((2, 6, 4))
# (2, 6, 4)
x = x + ffwd(layer_norm1(x))
```

So we have finally finished processing 1 block. Now, there are n blocks that run sequentially one after the other, the ouput of 1 block is the input to the second block.

```python
blocks = nn.Sequential(*[Block() for _ in range(n_blocks)])

# (2, 6, 4)
x = blocks(x)
```

Although the blocks are processed sequentially, the multi-attention heads are processed parallelly. Once this is complete, we pass the output to a last layernorm and then through a last feed-forward network that procjects the output from the embedding space to the vocabulary space.

```python
ln_f = nn.LayerNorm(n_embd)
lm_head = nn.Linear(n_embd, vocab_size)

x = blocks(x) # (2, 6, 4)
x = ln_f(x) # (2, 6, 4)
# (2, 6, 4) @ (4, vocab_size)
logits = lm_head(x) # (2, 6, vocab_size)
```

Here `vocab_size` is the unique characters that occur in the dataset. In our example,

```python

text = """
The sun dipped below the horizon, painting the sky with hues of orange and pink.
A gentle breeze rustled the leaves, creating a soothing melody.
In that peaceful moment, the world seemed to pause and breathe.
"""

chars = sorted(list(set(text)))
vocab_size = len(chars) # 30
```

Therefore the final output gives the logits, which when softmax is appleid to, tells us the probability of the next word occuring.
