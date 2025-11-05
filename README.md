# LLM Architectures in PyTorch - Complete Educational Guide

This repository contains **production-quality implementations** of three major LLM architectures with **2,400+ lines of extensively commented code** (41% comment ratio). Perfect for learning, interviews, and understanding how modern language models really work.

## 📖 Table of Contents
- [Quick Start](#-quick-start-5-minutes)
- [Three Core Architectures](#-three-core-architectures)
- [Repository Structure](#-repository-structure)
- [Learning Paths](#-learning-paths)
- [Core Concepts](#-core-transformer-concepts)
- [Training Techniques](#-training-techniques)
- [Inference Strategies](#-inference-strategies)
- [Advanced Topics](#-advanced-topics)
- [Hyperparameter Reference](#-hyperparameter-reference)
- [Customization Ideas](#-customization-ideas)
- [Troubleshooting](#-troubleshooting)
- [Further Reading](#-further-reading)

---

## 🚀 Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick demo (trains all 3 architectures on toy data)
python quick_start.py

# 3. Generate visualizations
python visualizations.py

# 4. Compare architectures
python compare_architectures.py
```

**That's it!** You now have trained models and visualizations. Read on to understand what just happened.

---

## 🎯 Three Core Architectures

| Architecture | Purpose | Attention | Best For | Pros | Cons |
|--------------|---------|-----------|----------|------|------|
| **Encoder-Only** (BERT) | Understand | Bidirectional | Classification, NER | ✅ Full context | ❌ Can't generate naturally |
| **Decoder-Only** (GPT) | Generate | Unidirectional | Text generation | ✅ Efficient generation | ❌ Limited context integration |
| **Encoder-Decoder** (T5) | Transform | Both | Translation, summarization | ✅ Best for seq2seq | ❌ More parameters |

### Quick Decision Guide
- **Understanding tasks?** → Encoder-Only
- **Generating text?** → Decoder-Only
- **Transforming sequences?** → Encoder-Decoder

---

## 📁 Repository Structure

| File | Lines | Description | Key Concepts |
|------|-------|-------------|--------------|
| **README.md** | - | **You are here!** Complete guide | Everything |
| **requirements.txt** | - | Python dependencies | torch, numpy, tqdm, matplotlib |
| | | | |
| **toy_dataset.py** | 350 | Number→word translation | Vocabulary, tokenization, masking, DataLoader |
| **encoder_only.py** | 650 | BERT-style (bidirectional) | Multi-head attention, LayerNorm, residual connections |
| **decoder_only.py** | 700 | GPT-style (causal) | Causal masking, autoregressive generation, sampling |
| **encoder_decoder.py** | 850 | T5-style (seq2seq) | **Cross-attention**, beam search, teacher forcing |
| | | | |
| **quick_start.py** | 250 | 5-minute demo of all 3 | Quick overview, minimal training |
| **train_all.py** | 500 | Full training pipeline | LR scheduling, gradient accumulation |
| **compare_architectures.py** | 400 | Side-by-side comparison | Decision making, use cases |
| **visualizations.py** | 400 | Generate 7 diagram types | Positional encoding, masks, architectures |

**Total: 4,100+ lines** with **1,670+ lines of comments** (41% ratio!)

---

## 🎓 Learning Paths

### Path 1: Practical (Recommended - 3 hours)
Perfect for getting hands-on quickly:
```bash
1. python quick_start.py              # 5 min - See everything in action
2. python visualizations.py           # 2 min - Generate diagrams
3. python compare_architectures.py    # 5 min - Understand differences
4. Read encoder_only.py               # 30 min - Study bidirectional attention
5. Read decoder_only.py               # 30 min - Study causal attention
6. Read encoder_decoder.py            # 45 min - Study cross-attention
7. Experiment with hyperparameters    # Variable - Make it your own
```

### Path 2: Theoretical (For ML Students - 4 hours)
Start with theory, then code:
```bash
1. Read "Core Concepts" section below # 60 min - Understand theory
2. python visualizations.py           # 2 min - See concepts visually
3. Study encoder_only.py              # 45 min - Code + theory
4. Study decoder_only.py              # 45 min - Code + theory
5. Study encoder_decoder.py           # 60 min - Code + theory
6. python train_all.py                # Variable - Advanced techniques
```

### Path 3: Comprehensive (For Experts - 6-10 hours)
Master everything:
```bash
1. Read entire README                 # 90 min - Full context
2. Study toy_dataset.py               # 15 min - Data preparation
3. Study + modify encoder_only.py     # 90 min - Deep dive
4. Study + modify decoder_only.py     # 90 min - Deep dive
5. Study + modify encoder_decoder.py  # 120 min - Deep dive
6. Implement your own variations      # Variable - Innovation
```

---

## 🧠 Core Transformer Concepts

### 1. Embeddings
**What**: Convert discrete tokens (words, subwords) into continuous vectors.

**Why**: Neural networks need numerical input, embeddings learn semantic relationships.

```python
embedding(token_id) -> vector of size d_model
```

**Key Points**:
- Each token gets a unique learnable vector
- Similar tokens have similar embeddings
- Typical dimensions: 128 (small), 768 (BERT), 12288 (GPT-3)

### 2. Positional Encoding
**What**: Add position information to embeddings.

**Why**: Transformers have no inherent notion of order (unlike RNNs).

**Sinusoidal Formula**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Benefits**:
- No learnable parameters
- Works for any sequence length
- Smooth interpolation between positions

### 3. Scaled Dot-Product Attention

**The Core Innovation**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**Components**:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What do I output?"
- **Scaling**: Prevents vanishing gradients (sqrt(d_k))

**Intuition**: For "The cat sat on the mat":
- "sat" attends to "cat" (who sat?) and "mat" (where?)
- Creates contextual representations automatically!

### 4. Multi-Head Attention

**Why Multiple Heads?**
- Each head learns different relationships:
  - Head 1: Grammatical relationships
  - Head 2: Semantic relationships  
  - Head 3: Long-range dependencies

**Formula**:
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_O
where head_i = Attention(Q*W_Q^i, K*W_K^i, V*W_V^i)
```

**Typical Values**: 8-16 heads, d_model / num_heads = d_k

### 5. Self-Attention vs Cross-Attention

| Type | Query | Key/Value | Used In | Purpose |
|------|-------|-----------|---------|---------|
| **Self-Attention** | Same seq | Same seq | All models | Token attends to same sequence |
| **Cross-Attention** | Decoder | Encoder | Encoder-Decoder | Decoder attends to encoder output |

**Cross-attention** is the secret sauce of translation models!

### 6. Attention Masks

**Three Types**:

1. **Padding Mask**: Ignore padding tokens
   ```python
   mask = (tokens != PAD_TOKEN)  # True = real, False = padding
   ```

2. **Causal Mask**: Prevent looking ahead (decoder-only)
   ```python
   mask = torch.tril(torch.ones(n, n))  # Lower triangular
   ```

3. **Combined Mask**: Both padding and causal
   ```python
   mask = padding_mask & causal_mask
   ```

**Implementation**:
```python
scores = scores.masked_fill(mask == 0, float('-inf'))
# -inf becomes ~0 after softmax
```

### 7. Layer Normalization & Residual Connections

**Layer Normalization**:
```python
LN(x) = γ * (x - μ) / sqrt(σ² + ε) + β
```
- Stabilizes training
- Enables deeper networks

**Residual Connections**:
```python
output = layer(input) + input  # Skip connection
```
- Addresses vanishing gradients
- Model learns "what to change" not "what to output"

**Pre-Norm vs Post-Norm**:
- **Post-Norm**: `x = LN(x + sublayer(x))` (original)
- **Pre-Norm**: `x = x + sublayer(LN(x))` ✓ Better for deep models

### 8. Feed-Forward Networks

**Architecture**:
```
Input -> Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)
```

**Why?**
- Adds non-linearity after attention
- Typically d_ff = 4 × d_model (expansion then compression)
- Applied to each position independently

---

## 🎯 Training Techniques

### 1. Teacher Forcing
**What**: Use ground truth as input during training, not model predictions.

**Example** (translation):
```
Step 1: Decoder gets "<SOS>"          → Predict "Bonjour"
Step 2: Decoder gets "<SOS> Bonjour"  → Predict "le"
         (Use ground truth "Bonjour", not prediction!)
```

**Benefits**: Faster training, stable gradients  
**Downside**: Exposure bias (training ≠ inference)

### 2. Learning Rate Scheduling

**Warmup + Cosine Decay** (State-of-the-art):
```python
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)  # Linear warmup
else:
    progress = (step - warmup) / (total - warmup)
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

**Why Warmup?** Initial parameters are random, large gradients destabilize training  
**Why Decay?** As loss decreases, smaller updates for fine-tuning

### 3. Gradient Accumulation

**Problem**: Limited GPU memory → small batch size  
**Solution**: Accumulate gradients over multiple mini-batches

```python
accumulation_steps = 4  # Effective batch = 32 * 4 = 128
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()
```

### 4. Gradient Clipping

**Problem**: Exploding gradients in deep networks  
**Solution**: Clip gradient norm

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**When to use**: Always! Essential for stable transformer training.

---

## 🚀 Inference Strategies

### Comparison Table

| Method | Speed | Diversity | Quality | Best For |
|--------|-------|-----------|---------|----------|
| **Greedy** | ⚡⚡⚡ Fastest | ❌ None | 👍 Good | Quick demos, deterministic |
| **Temperature** | ⚡⚡⚡ Fast | 🎨 Adjustable | 👍👍 Good | Controlled creativity |
| **Top-k** | ⚡⚡ Fast | 🎨🎨 Medium | 👍👍 Good | General purpose |
| **Top-p** | ⚡⚡ Fast | 🎨🎨🎨 High | 👍👍 Very Good | Creative writing |
| **Beam Search** | ⚡ Slow | ❌ Low | 👍👍👍 Best | Translation, summarization |

### 1. Greedy Decoding
```python
next_token = torch.argmax(probs, dim=-1)  # Pick highest probability
```
**Pros**: Fast, deterministic  
**Cons**: Can miss better sequences

### 2. Temperature Sampling
```python
probs = softmax(logits / temperature)
# temperature < 1: Conservative (factual)
# temperature > 1: Creative (stories)
```

### 3. Top-k Sampling
```python
# Keep only top k tokens, sample from them
top_k_logits, indices = torch.topk(logits, k)
```

### 4. Top-p (Nucleus) Sampling
```python
# Keep tokens until cumulative probability >= p
# Adaptive: fewer tokens when confident, more when uncertain
```

### 5. Beam Search
```python
# Keep top-k candidates at each step
# More systematic search, better quality
# k times slower
```

**Length Penalty**:
```python
score = log_prob / (length ** penalty)
# penalty > 1: favor longer sequences
```

---

## 🎨 Advanced Topics

### 1. Attention Variants
- **Flash Attention**: 2-4x faster, memory-efficient, exact
- **Multi-Query Attention (MQA)**: Single K,V for all heads → faster inference
- **Grouped-Query Attention (GQA)**: Middle ground between MHA and MQA

### 2. Position Embeddings
- **RoPE** (Rotary): Better length extrapolation (used in LLaMA)
- **ALiBi**: Add linear bias to attention scores, no embeddings needed

### 3. Sparse Attention
**Problem**: Full attention is O(n²)  
**Solutions**: Local attention, strided attention, Longformer, BigBird

### 4. Model Scaling Laws
```
Loss = (N_0 / N)^α + (D_0 / D)^β
```
- Larger models are more sample-efficient
- Chinchilla scaling: Balance model size and data size

### 5. Parameter-Efficient Fine-Tuning (PEFT)

**LoRA** (Low-Rank Adaptation):
```
W_new = W_pretrained + AB  # A:(d,r), B:(r,d), r<<d
Only train A and B → 0.1% of parameters!
```

**Other methods**: Prefix tuning, adapters, prompt tuning

### 6. Memory Optimization
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: FP16 operations → 2x speedup, 2x memory reduction
- **Parameter Sharing**: Share weights across layers (ALBERT)

---

## 🔧 Hyperparameter Reference

### Model Architecture

| Parameter | Small | Medium | Large | Notes |
|-----------|-------|--------|-------|-------|
| **d_model** | 128-256 | 512-768 | 1024-2048 | Embedding dimension |
| **num_heads** | 4-8 | 8-12 | 16-32 | Must divide d_model |
| **num_layers** | 2-4 | 6-12 | 24-96 | More = better but slower |
| **d_ff** | 512-1024 | 2048-3072 | 4096-8192 | Typically 4×d_model |
| **dropout** | 0.1 | 0.1 | 0.1-0.3 | More for larger models |

### Training

| Parameter | Conservative | Standard | Aggressive |
|-----------|-------------|----------|------------|
| **Learning Rate** | 1e-5 | 1e-4 | 5e-4 |
| **Warmup Steps** | 1000 | 4000 | 10000 |
| **Batch Size** | 16-32 | 64-128 | 256-512 |
| **Grad Clip** | 0.5 | 1.0 | 5.0 |

### Inference

| Parameter | Conservative | Balanced | Creative |
|-----------|-------------|----------|----------|
| **Temperature** | 0.7 | 1.0 | 1.5 |
| **Top-k** | 10 | 50 | 100 |
| **Top-p** | 0.75 | 0.9 | 0.95 |
| **Beam Width** | 3 | 5 | 10 |

---

## 🎓 Customization Ideas

### Beginner Modifications
```python
# In any model file, change these:
D_MODEL = 128       # Try: 64, 256, 512
NUM_LAYERS = 2      # Try: 1, 4, 6
NUM_HEADS = 4       # Try: 2, 8 (must divide d_model)
BATCH_SIZE = 32     # Try: 16, 64
LEARNING_RATE = 1e-4  # Try: 5e-5, 5e-4
```

### Intermediate Modifications
- Implement different positional encodings (learnable, RoPE)
- Add dropout in different locations
- Try different activation functions (GELU, Swish)
- Implement weight tying (share embeddings with output layer)

### Advanced Modifications
- Implement Flash Attention
- Add sparse attention patterns
- Implement mixture of experts (MoE)
- Add LoRA for fine-tuning
- Implement model quantization

---

## 🔧 Troubleshooting

### Training Issues

**Loss not decreasing?**
- ✓ Check learning rate (1e-4 is safe)
- ✓ Verify data preprocessing (tokens, masks)
- ✓ Reduce model size (start with d_model=128)
- ✓ Check for NaN gradients (`torch.isnan(loss)`)

**Training too slow?**
- ✓ Enable mixed precision (FP16)
- ✓ Increase batch size
- ✓ Use gradient accumulation
- ✓ Profile with `torch.profiler`

**CUDA out of memory?**
```python
# Solutions (in order of preference):
1. Reduce batch_size: 32 → 16 → 8
2. Reduce sequence length: 512 → 256
3. Enable gradient checkpointing
4. Reduce d_model: 512 → 256
5. Use gradient accumulation
```

### Inference Issues

**Generating gibberish?**
- ✓ Train longer (loss < 1.0 for toy dataset)
- ✓ Lower temperature (1.0 → 0.7)
- ✓ Use beam search (width=5)
- ✓ Check tokenization/detokenization

**Generation too slow?**
- ✓ Use greedy decoding for prototyping
- ✓ Reduce beam width (10 → 3)
- ✓ Use caching (store past key/values)
- ✓ Consider model quantization

**Repetitive outputs?**
- ✓ Use top-p/top-k sampling
- ✓ Increase temperature
- ✓ Add repetition penalty
- ✓ Use beam search with diversity

### Model Issues

**Attention weights are uniform?**
- ✓ Check masking logic
- ✓ Verify positional encodings
- ✓ Try different initialization
- ✓ Train longer

**Overfitting quickly?**
- ✓ Increase dropout (0.1 → 0.3)
- ✓ Add weight decay (L2 reg)
- ✓ Use more data
- ✓ Reduce model capacity

---

## 📚 Learning Milestones

**After 1 hour**, you should understand:
- ✅ The three architecture types and when to use each
- ✅ Basic attention mechanism
- ✅ How to run and modify the code

**After 3 hours**, you should understand:
- ✅ Multi-head attention implementation
- ✅ Masking strategies (padding, causal, cross)
- ✅ Training loop with teacher forcing
- ✅ Basic inference strategies

**After 6 hours**, you should understand:
- ✅ All code in detail
- ✅ Cross-attention in encoder-decoder
- ✅ Advanced inference (beam search, sampling)
- ✅ Hyperparameter tuning

**After 10 hours**, you should be able to:
- ✅ Implement custom architectures
- ✅ Debug training issues
- ✅ Optimize for production
- ✅ Explain concepts to others

---

## 📖 Further Reading

### 📄 Essential Papers (Read in order)
1. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (2017) - The original Transformer
   - *Difficulty*: Medium | *Priority*: Must Read
   
2. **[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** (2018)
   - *Difficulty*: Easy | *Use Case*: Understanding tasks
   
3. **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** (GPT-2, 2019)
   - *Difficulty*: Easy | *Use Case*: Generation
   
4. **[Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)** (2019)
   - *Difficulty*: Medium | *Use Case*: Seq2seq tasks
   
5. **[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)** (2023)
   - *Difficulty*: Advanced | *Modern best practices*

### 📚 Detailed Guides
- **[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)** - Best visual introduction
- **[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)** - Line-by-line PyTorch
- **[Lil'Log: The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)** - Comprehensive variants

### 🎥 Video Resources
- **Andrej Karpathy's "Let's build GPT"** - From scratch tutorial
- **Stanford CS224N** - NLP with Deep Learning course
- **Hugging Face Course** - Practical transformers

### 🛠️ Practical Resources
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - Production library
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** - Minimal GPT implementation
- **[minGPT](https://github.com/karpathy/minGPT)** - Educational GPT

---

## 🎯 Quick Command Reference

```bash
# Training
python encoder_only.py          # Train BERT-style (classification)
python decoder_only.py           # Train GPT-style (generation)
python encoder_decoder.py        # Train T5-style (translation)
python train_all.py              # Advanced training pipeline

# Visualization
python visualizations.py         # Generate all diagrams
python compare_architectures.py  # Understand differences

# Quick demos
python quick_start.py            # 5-minute overview
python toy_dataset.py            # Inspect dataset

# Common modifications
# In any training file:
# - Change BATCH_SIZE = 32 → your value
# - Change NUM_EPOCHS = 20 → your value  
# - Change D_MODEL = 256 → your value
# Then rerun!
```

---

## 🤝 Contributing

This is an educational repository. Contributions welcome!

**How to help**:
- 🐛 Report bugs or unclear comments
- 📝 Improve documentation
- 💡 Suggest additional concepts to cover
- ✨ Add more visualizations
- 🌍 Translate comments to other languages

**Guidelines**:
- Keep code heavily commented (40%+ ratio)
- Prioritize clarity over performance
- Include visual examples where possible
- Test on toy dataset (fast iteration)

---

## 🙏 Acknowledgments

**Inspired by**:
- Andrej Karpathy's educational philosophy
- Harvard's Annotated Transformer
- Jay Alammar's visual explanations

**Built for**:
- ML students preparing for interviews
- Engineers transitioning to LLM work
- Anyone curious about how ChatGPT works under the hood

---

## 📄 License

**MIT License** - Use freely for learning, teaching, and commercial projects.

---

## 📞 Questions?

**Getting stuck?** Check:
1. Comments in the code (1,670+ lines!)
2. Troubleshooting section above
3. Generated visualizations (`python visualizations.py`)
4. Output from `python compare_architectures.py`

**Still confused?** The code is the documentation! Every line is explained.

---

**Happy Learning! 🚀**

*Remember: The best way to learn is to read the code, run it, break it, fix it, and make it your own!*
