---
layout: ../../layouts/BlogPost.astro
title: "Transformers (Decoder-Only) (Part 2)"
date: 2026-01-26
---

In part 1, I explained transformers using block diagrams and small snippets of code. But in this part, I will explain it in detail.

Let's use the TinysStories dataset and the framework will dlgrad. The GPT implementation can be found in the dlgrad repo [here](https://github.com/NavneetKanna/dlgrad/blob/main/examples/gpt.py).

Note that the code here might be slightly different than shown in part 1, since in this part we will be using 4d tensors.

Let's use this config,

```python
class GPTConfig:
    vocab_size = 0
    block_size = 256 # Context length
    n_layer = 6
    n_head = 4
    n_embd = 128
    dropout = 0.2
    learning_rate = 1e-4
    max_iters = 10000
    batch_size = 16
    eval_interval = 500
    device = "cpu"

'''
vocab_size = This is a number that represents the unique words or charcters in the dataset.
block_size = Also called context_length or time dimension, this represents how much of the dataset the model sees.
             Its the window size, higher the context length, the better the response can be since it can see more
             from the past.
n_layer    = The number of blocks.
n_head     = The number of heads.
n_embd     = The embedding dimension length. This number represents the size of each token.
'''

```

### Step 1: Embedding

```python
class GPT:
    def __init__(self, config):
        self.config = config
        # (82, 128)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # (256, 128)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx):
        # (16, 256)
        B, T = idx.shape

        tok_emb = self.wte(idx) # (16, 256, 128) # (BS, context_len, n_embd)

        pos_idxs = np.arange(T)
        pos_idxs_t = Tensor(pos_idxs)
        pos_emb = self.wpe(pos_idxs_t) # (256, 128) (context_len, n_embd)

        # (16, 256, 128) + (256, 128)
        x = tok_emb + pos_emb
```

```python
with open('TinyStories-valid.txt') as f:
    lines = f.read()

text = "".join(lines)

tokenizer = CharTokenizer(text)
# 82
config.vocab_size = tokenizer.vocab_size

model = GPT(config)

data = np.array(tokenizer.encode(text), dtype=np.int32)

# (16,)
ix = np.random.randint(0, len(data) - config.block_size, (config.batch_size,))

x_batch = np.stack([data[j : j+config.block_size] for j in ix])
y_batch = np.stack([data[j+1 : j+config.block_size+1] for j in ix])

# (16, 256) (BS, context_len)
x_t = Tensor(x_batch)

# (16, 256) (BS, context_len)
y_t = Tensor(y_batch)

logits, loss = model(x_t, y_t)
```

- Each word (or character) in the dataset is first converted into an integer using the CharTokenizer. For example, 'a' might become 12.
- However, feeding raw numbers like 12 into a neural network isn't very useful. We need a richer representation.
- This is solved using Embeddings. An embedding layer is essentially a lookup table that swaps every integer for a dense vector (a list of 128 numbers in this case) that learns to represent
  the meaning of that token.
- WTE: tok_emb = self.wte(idx); looks up the identity of the token. It converts our input of shape (16, 256) into (16, 256, 128). Now, every character/word is a vector.
- WPE: Transformers have no inherent sense of order. If you scrambled the sentence, the math would look the same. To fix this, we create a second list of numbers [0, 1, ... T] representing positions
  and look up vectors for them too. This gives us pos_emb with shape (256, 128).
- When we add WTE and WPE: x = tok_emb + pos_emb; we are mixing the information from both the vectors for each character/word. And this is broadcasted for the full batch.
- The resulting tensor tells the model two things at once: "I am the letter 'A' AND I am at the 5th position." This tensor x is what finally enters the Transformer blocks.

### Step 2: Inputs

- The inputs to a transformer model are the current word and the words that come after it. For example, if the text is: "Once upon a time" and block_size is 3, x_t is [Row 1]: ['Once', 'upon', 'a'] and 
target y_t is [Row 1]: ['upon', 'a', 'time'].

### Step 3: Forward pass

```python
class GPT:
    def __init__(self, config):
        self.config = config
        # (82, 128)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # (256, 128)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx):
        # (16, 256, 128) + (256, 128)
        x = tok_emb + pos_emb

        # (16, 256, 128)
        for block in self.blocks:
            x, _, _ = block(x)
        # (16, 256, 128)
        x = self.ln_f(x)
        # (16, 256, 82)
        logits = self.lm_head(x)
        # (4096, 82)
        logits_flat = logits.reshape((B * T, self.config.vocab_size))
        # (4096, 1)
        targets_flat = targets.reshape((B * T, 1))
        loss = logits_flat.cross_entropy_loss(targets_flat)
        return logits, loss
```

- The input x flows through all the blocks sequentially, the shape is retained.
- After the blocks, we need to turn the vectors back into vocabulary probabilities. The lm_head is a Linear layer that projects the embedding size (128) up to the vocabulary size (82).
- The model outputs logits of shape (Batch, Time, Vocab).
- However, the Cross Entropy Loss function expects a simple list of predictions and a list of targets.
- So, we flatten (reshape) the output into (Batch * Time, Vocab). Effectively, we stack all the words from the whole batch into one giant column to calculate the error.

### Step 4: Blocks

```python
class Block:
    def __init__(self, config):
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x, past_k=None, past_v=None):
        # x = (16, 256, 128)
        # norm1 = (16, 256, 128)
        norm1 = self.ln1(x)
        # (16, 256, 128), (16, 4, 256, 32), (16, 4, 256, 32)
        attn_out, new_k, new_v = self.attn(norm1, past_k, past_v)
        # (16, 256, 128)
        x = x + attn_out
        # (16, 256, 128)
        x = x + self.mlp(self.ln2(x))
        return x, new_k, new_v
```

- We apply ln1 (RMSNorm) before the attention layer. This is known as "Pre-Norm" architecture (used in GPT-3 and Llama).
- Attention is the communication step: tokens exchange information with each other.
- The addition step is called residual connections. This is used because GPT is a very deep architecture. During backpropagation, the gradients can vanish as they flow backwards. To prevent this,
  we use addition, this is because, for the addition op, the gradient flows equally to both the parents of the op.

### Step 5: Attention

```python
class CausalSelfAttention:
    def __init__(self, config):
        # n_embd = 128, n_head = 4, head_dim = 32
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5

        # Key, Query, Value projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def __call__(self, x, past_k=None, past_v=None):
        B, T, C = x.shape # (16, 256, 128)

        # Linear Projections
        # q, k, v become (16, 256, 128)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split Heads (Reshape & Transpose)
        # Reshape: (16, 256, 4, 32) -> Transpose: (16, 4, 256, 32)
        q = q.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)
        k = k.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)
        v = v.reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)

        # Attention Score (Queries x Keys)
        # (16, 4, 256, 32) @ (16, 4, 32, 256) -> (16, 4, 256, 256)
        att = (q @ k.transpose(2, 3)) * self.scale

        # Causal Masking (Hide the future)
        mask = Tensor.tril(Tensor.ones((T, T)), k=0.0)
        att = att.masked_fill(mask == Tensor(0.0), float('-inf'))
        att = att.softmax(dim=3)

        # Aggregate Information (Scores x Values)
        # (16, 4, 256, 256) @ (16, 4, 256, 32) -> (16, 4, 256, 32)
        y = att @ v

        # Reassemble Heads
        # (16, 256, 128)
        y = y.transpose(1, 2).reshape((B, T, C))
        return self.c_proj(y), k, v
```

- We take our input x and project it into three different versions: Q, K, V (Query, Key, Value)
- Multi-Head Attention: Instead of one giant vector of size 128, we split it into 4 heads of size 32. It allows the model to focus on different things simultaneously.
  One head might look for grammar relationships, another for names, another for tone.
- Batch Dimension: We reshape the tensor from (Batch, Time, n_embd) to (Batch, Heads, Time, Head_Size). The 4 heads can be processed in parallel.
- Scaled Dot Product (q @ k): We multiply Query and Key to get an "affinity score". If the dot product is high, the two tokens are related.
  Result Shape (256, 256): This is a grid showing how much every word relates to every other word.
- The Causal Mask: This is the most important line for GPT. We can't let the 5th word see the 6th word. We use tril (Triangle Lower) to force the upper-right corner of the matrix to -infinity.
  When we run softmax, these become 0.
- Weighted Sum (att @ v): Finally, we multiply the scores by the Values. This effectively says: "I will take 90% of the information from the word 'King', 5% from 'The', and 5% from 'is'."
- When we reassemble the heads, we just stack them side-by-side. The top 32 numbers come purely from Head 1, the next 32 from Head 2, etc. They haven't interacted yet.
  c_proj is a Linear layer that "mixes" all these independent findings together. This ensures that the information flowing out of the attention block is a rich blend of all the different perspectives (heads),
  ready to be added back to the main stream via the residual connection.

### Step 6: The MLP (Feed Forward)

```python
class MLP:
    def __init__(self, config):
        # We expand the dimension by 4x (128 -> 512)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        # x is (16, 256, 128)
        
        # Expand (Project Up)
        # (16, 256, 128) -> (16, 256, 512)
        x = self.c_fc(x)
        
        x = x.relu()
        
        # Contract (Project Down)
        # (16, 256, 512) -> (16, 256, 128)
        x = self.c_proj(x)
        return x
```

- It is standard practice in Transformers to project the embedding dimension up by a factor of 4 inside the MLP.

<div style="text-align:center;">
  <img src="/assets/images/block.svg" alt="block" style="display:inline-block;">
</div>
