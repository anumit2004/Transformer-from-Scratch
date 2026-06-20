# Transformer from Scratch - Complete Guide

## Overview

This implementation builds a **Transformer** neural network for sequence-to-sequence tasks (e.g., English to French translation). The transformer relies entirely on **attention mechanisms** instead of RNNs or CNNs, enabling parallel processing and capturing long-range dependencies efficiently.

**Architecture Flow:**
```
Input → Encoder (with self-attention) → Decoder (with cross-attention) → Output
```

---

## 1. Positional Encoding

**Purpose:** Transformers don't inherently understand token position. Positional encodings inject position information into embeddings.

**Formula:**
- Even dimensions: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- Odd dimensions: `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

**Implementation:**
- Pre-computed for all positions up to `max_len`
- Mixed sine/cosine at different frequencies creates a unique pattern for each position
- Added to token embeddings before processing

**Why it works:** Different frequencies allow the model to learn position-dependent patterns at various scales (nearby tokens, distant tokens, etc.).

---

## 2. Scaled Dot-Product Attention

**Core Attention Mechanism**

**Formula:** `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

**Components:**
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What can you offer?"
- **Value (V):** "Here's the actual information"

**Process:**
1. Compute similarity scores: `Q × K^T` (which keys match this query?)
2. Scale by `√d_k` to prevent exploding gradients
3. Apply softmax to get attention weights (0-1, sum to 1)
4. Multiply weights by Values to get weighted sum

**Why scaling:** In high dimensions, dot products can become very large, causing softmax to produce near-zero gradients. Scaling stabilizes training.

---

## 3. Multi-Head Attention

**Purpose:** Allow the model to attend to different representation subspaces simultaneously.

**How it works:**
1. Project Q, K, V into `h` different subspaces (heads): each head has dimension `d_k = d_model / h`
2. Run attention independently on each head
3. Concatenate all head outputs
4. Project back to `d_model` dimensions

**Benefit:** Each head can focus on different patterns:
- Head 1: Short-range dependencies
- Head 2: Long-range semantic relationships
- Head 3: Syntactic structure
- ...and so on

---

## 4. Feed-Forward Network

**Purpose:** Add non-linearity and increase model capacity within each token's representation.

**Architecture:** Position-wise (applied to each token independently)
```
Input → Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model) → Output
```

**Typical dimensions:** `d_ff = 4 × d_model` (e.g., 512 → 2048 → 512)

**Why it works:** 
- First layer expands the representation space (allows complex interactions)
- ReLU adds non-linearity
- Second layer projects back to original dimensions

---

## 5. Encoder Layer (Building Block)

**Components:**
1. **Multi-Head Self-Attention:** Each token attends to all tokens in the sequence
2. **Feed-Forward Network:** Pointwise enrichment

**Key Detail - Pre-Layer Normalization:**
```
x = LayerNorm(x)
x = x + MultiHeadAttention(x, x, x)  # residual connection

x = LayerNorm(x)
x = x + FeedForward(x)  # residual connection
```

**Why Pre-LN:** More stable training, especially for deep models. Normalization happens before processing, preventing gradient issues.

---

## 6. Encoder

**Purpose:** Transform input sequence into contextual representations.

**Steps:**
1. **Token Embedding:** Convert token IDs to dense vectors
2. **Scale:** Multiply by `√d_model` (stabilize magnitude before adding PE)
3. **Positional Encoding:** Add position information
4. **Stack of Layers:** Pass through N identical encoder layers (typically 6)

**Output:** Contextualized representation of the entire input sequence

---

## 7. Decoder Layer (With Cross-Attention)

**Components:**
1. **Masked Self-Attention:** Target attends to itself (causally - can only see past tokens)
2. **Cross-Attention:** Target queries encoder's output (learns what to extract from source)
3. **Feed-Forward:** Pointwise enrichment

**Causal Masking:** Position `i` can only attend to positions `≤ i` (prevents "cheating" during training by looking at future tokens)

---

## 8. Decoder

**Purpose:** Generate target sequence token-by-token, informed by encoder output.

**Steps:**
1. **Token Embedding & Positional Encoding:** Same as encoder
2. **Stack of Layers:** N identical decoder layers
   - Each layer attends to its own previous tokens (masked self-attention)
   - Each layer attends to encoder output (cross-attention)
   - Each layer applies feed-forward

**Output:** Hidden states that are projected to vocabulary logits for next token prediction

---

## 9. Complete Transformer

**Architecture:**
```python
Transformer
├── Encoder (6 layers)
│   ├── Token Embedding
│   ├── Positional Encoding
│   └── [6× EncoderLayer]
│       ├── MultiHeadAttention (self)
│       └── FeedForward
│
├── Decoder (6 layers)
│   ├── Token Embedding
│   ├── Positional Encoding
│   └── [6× DecoderLayer]
│       ├── MultiHeadAttention (masked self)
│       ├── MultiHeadAttention (cross - attends to encoder)
│       └── FeedForward
│
└── Output Projection: d_model → vocab_size
```

**Data Flow (Training):**
1. Encoder processes source sequence: `src_ids → encoder → enc_out`
2. Decoder processes target sequence with encoder output:
   - Masked self-attention: each target token sees only previous targets
   - Cross-attention: each target token queries encoder output
   - Feed-forward enrichment
   - Result: `dec_out`
3. Linear projection: `dec_out → logits → softmax → probabilities`
4. Loss: Compare predicted tokens to actual target tokens

**Data Flow (Inference):**
1. Encode source once (reuse for all decoding steps)
2. Start with `<BOS>` (beginning-of-sequence) token
3. Loop:
   - Decode current target to get logits
   - Pick next token (greedy: argmax)
   - Append to sequence
   - Stop if `<EOS>` or max length reached

---

## 10. Training Pipeline

**Components:**
1. **Dataset:** Helsinki-NLP/opus_books (English-French translation)
2. **Tokenizer:** T5-small tokenizer (handles multilingual encoding)
3. **Loss:** Cross-Entropy Loss (only on non-padding tokens)
4. **Optimizer:** Adam with learning rate scheduling
5. **Warmup & Scheduling:** Linear warmup for 10% of training, then linear decay

**Batch Processing:**
- Tokenize English sentences (source) and French sentences (target)
- Pad to sequence length 64
- Decoder input: `[<BOS>, word1, word2, ..., word_n]` (shift targets left by 1)
- Decoder target: `[word1, word2, ..., word_n, <EOS>]` (shift targets right by 1)

**Training Details:**
- Gradient clipping (`max_norm=1.0`) prevents exploding gradients
- Dropout during training reduces overfitting
- Layer normalization stabilizes deep stacking

---

## 11. Key Insights

| Component | Purpose | Key Parameter |
|-----------|---------|---|
| **Positional Encoding** | Inject position info | Sine/cosine frequencies |
| **Attention** | Find relevant tokens | Query-Key similarity |
| **Multi-Head** | Diverse representations | Number of heads (8) |
| **Feed-Forward** | Non-linearity & capacity | Expansion factor (4×) |
| **Residual Connection** | Stable gradient flow | Skip connection |
| **Layer Norm** | Stabilize training | Applied before sublayers (Pre-LN) |
| **Masking** | Enforce causality | Lower-triangular matrix |

---

## 12. Mathematical Summary

| Operation | Formula | Purpose |
|-----------|---------|---------|
| **Attention** | `softmax(QK^T/√d_k)V` | Weighted aggregation |
| **Multi-Head** | `Concat(head_1,...,head_h)W^O` | Multiple perspectives |
| **FFN** | `Linear2(ReLU(Linear1(x)))` | Non-linear enrichment |
| **Residual** | `x + Sublayer(x)` | Enable deep stacking |
| **LayerNorm** | `(x - μ) / (σ + ε)` | Normalize activations |

---

## 13. Performance Metrics

During training, the model tracks:
- **Loss:** Cross-entropy loss (lower = better)
- **Accuracy:** Token-level accuracy (% of correct tokens predicted)

Example progression:
```
Epoch 1: Loss=5.2, Accuracy=15%  (random-like guessing)
Epoch 5: Loss=3.1, Accuracy=35%  (learning patterns)
Epoch 10: Loss=2.4, Accuracy=42% (reasonable predictions)
```

---

## 14. When to Use Transformers

✅ **Good for:**
- Sequence-to-sequence (translation, summarization)
- Language modeling
- Question answering
- Any task with long-range dependencies
- Parallel processing (no sequential bottleneck like RNNs)

❌ **Bad for:**
- Very long sequences (quadratic attention complexity O(n²))
- Tasks requiring explicit state transitions
- Real-time inference with strict latency requirements

---

## 15. Hyperparameter Guide

| Hyperparameter | Value | Impact |
|---|---|---|
| `d_model` | 512 | Model capacity |
| `num_heads` | 8 | Diversity of attention |
| `num_layers` | 6 | Model depth/capacity |
| `d_ff` | 2048 | FFN capacity (4× d_model) |
| `dropout` | 0.1 | Regularization strength |
| `max_len` | 5000 | Max sequence length |
| `learning_rate` | 0.0003 | Optimization speed |
| `warmup_steps` | 10% of total | Learning rate schedule |

---

## Summary

The Transformer is built from simple, elegant components:
1. **Attention:** Learns what to focus on
2. **Multi-Head:** Attends to multiple patterns
3. **Feed-Forward:** Adds expressiveness
4. **Residuals & Normalization:** Enables deep stacking
5. **Positional Encoding:** Preserves order information

These components combine into a powerful architecture that achieves SOTA results on many NLP tasks while remaining highly parallelizable.

