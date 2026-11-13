<div align="center">

# 🧠 GPT-2 (345M) from First Principles in PyTorch

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-GPT--2%20Medium%20(345M)-792EE5?style=for-the-badge)](https://github.com/openai/gpt-2)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS%20%7C%20CUDA-green?style=for-the-badge)]()
[![Zero Blackbox](https://img.shields.io/badge/Blackbox%20Libs-0%25%20(No%20HuggingFace)-red?style=for-the-badge)]()

**A completely transparent, from-scratch implementation of the 345M parameter GPT-2 Medium architecture built in pure PyTorch.**

[Overview](#-1-what-we-built) • [How It Was Built](#-2-how-we-built-it) • [Why We Built It & Potential](#-3-why-we-built-it--real-world-potential) • [Setup Guide](#-4-quickstart--setup-guide) • [Architecture Deep Dive](ARCHITECTURE.md)

</div>

---

## 📖 1. What We Built

Most modern language model tutorials rely on high-level wrappers like HuggingFace `AutoModelForCausalLM` or pre-packaged training libraries that hide the real engineering.

**This project takes the opposite approach.** We constructed the entire **GPT-2 Medium (345M parameter)** architecture from first principles using only base PyTorch tensor operations:

* **345 Million Parameters:** Configured with 24 transformer blocks, 16 attention heads, and an embedding dimension of $d_{\text{model}} = 1024$.
* **Zero Modeling Abstractions:** Every linear projection, scaled dot-product calculation, causal attention mask, Pre-LayerNorm transformation, and residual highway is written explicitly.
* **Direct OpenAI Checkpoint Ingestion:** Includes a custom parser that reads raw TensorFlow checkpoint binaries from OpenAI and translates the parameter tensors into our custom PyTorch module.
* **Full Autoregressive Generation Engine:** An end-to-end decoding pipeline featuring vectorized Top-K sampling, temperature scaling, and deterministic seeding.

### 🎬 Real Generation Output

Here is what our custom model produces when given raw text prompts:

```text
Prompt: "I love cooking"
Settings: temperature=0.8, top_k=50, max_new_tokens=35

Output:
"I love cooking and I've had a few great experiences with folks who have cooked 
a healthy food that is healthy, fun, and delicious. My goal is to teach more 
ways of making healthy meals! Thank you to..."
```

```text
Prompt: "Every effort moves you"
Settings: temperature=0.8, top_k=50, max_new_tokens=35

Output:
"Every effort moves you forward in your mission. As if your own strength was 
a weapon: 'And yet I can see no end to the road ahead...'"
```

---

## 🛠️ 2. How We Built It

Instead of relying on third-party abstractions, the system is assembled like fine mechanical clockwork:

```
Token IDs (B, T)
      │
      ▼
┌─────────────────────────┐
│ Token + Pos Embeddings  │ ──► Dimensions: (B, T, 1024)
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│   Transformer Block 1   │ ──► Pre-LN ──► Multi-Head Attention (16 heads × 64d)
│            │            │ ──► Pre-LN ──► GELU Feed-Forward (1024 ──► 4096 ──► 1024)
│         . . .           │
│   Transformer Block 24  │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│     Final LayerNorm     │ ──► Dimensions: (B, T, 1024)
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│     Linear LM Head      │ ──► Dimensions: (B, T, 50257)
└─────────────────────────┘
      │
      ▼
Autoregressive Sampling (Top-K + Temperature) ──► Next Token ID
```

### Core Engineering Highlights:

* **Strict Causal Multi-Head Attention:** Built with an upper-triangular mask setting future attention scores to $-\infty$, ensuring no token can attend to subsequent positions during training or inference.
* **Pre-Layer Normalization & Residual Streams:** Normalization is applied *prior* to attention and MLP blocks, providing a clean additive residual highway that prevents vanishing gradients across all 24 layers.
* **Smooth GELU Activation:** Handcrafted approximation of the Gaussian Error Linear Unit ($x \cdot \Phi(x)$), delivering smooth non-linear transitions unlike standard ReLU.
* **TensorFlow Checkpoint Bridge:** A specialized extractor (`gpt_download3.py`) that lists, slices, and transposes raw checkpoint arrays (e.g., splitting combined $Q, K, V$ weights) directly into PyTorch parameters.
* **Vectorized Top-K Decoding:** Properly ranked 2D tensor operations that broadcast correctly across arbitrary batch sizes ($B \ge 1$) for fast text synthesis.

> 📚 **Want the full math & tensor dimension breakdown?** Read our comprehensive [ARCHITECTURE.md](ARCHITECTURE.md) covering every equation, weight shape, and parameter derivation.

---

## 💡 3. Why We Built It & Real-World Potential

While today's frontier models reach hundreds of billions of parameters, massive models are often too slow, costly, and resource-intensive for specialized, privacy-sensitive tasks.

A compact, highly optimized 345M parameter model built from first principles offers compelling real-world advantages:

```
                         REAL-WORLD VALUE
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
 1. Edge & On-Device     2. Domain-Specific       3. Research &
    Deployment              Classifiers / Tasks      Custom Prototyping
    • < 1.5 GB memory       • Support ticket routing • Custom attention layers
    • Runs on laptop/Mac    • Spam / fraud detect    • Quantization kernels
    • Zero API fees         • Medical / Legal NLP    • Architecture tests
```

1. **Ultra-Low Latency & Edge AI:**  
   The entire 345M model takes only ~1.4 GB of RAM in FP32 (and under 400 MB when quantized). It runs blazingly fast on consumer hardware—like Mac Apple Silicon or edge devices—with **zero API fees** and **zero internet latency**.

2. **100% Data Privacy & Air-Gapped Environments:**  
   In regulated sectors (finance, healthcare, defense), sending proprietary documents to third-party cloud APIs is often legally prohibited. This model runs entirely offline on local hardware.

3. **Specialized Downstream Tasks:**  
   By replacing the final linear head, this architecture can be fine-tuned into high-accuracy, ultra-fast domain models for:
   * **Automated Customer Support Routing** (categorizing inbound enterprise tickets).
   * **Spam & Phishing Detection** in live streaming pipelines.
   * **Constrained DSL & Code Autocompletion** (SQL queries, custom regex, or medical shorthand).

4. **Transparent Research Sandbox:**  
   Because every line of code is accessible and unhidden by frameworks, this codebase serves as an ideal testbed for experimenting with novel attention mechanisms (FlashAttention, Linear Attention), state-space models, or custom quantization algorithms.

---

## 🚀 4. Quickstart & Setup Guide

### Prerequisites
* Python 3.10+
* PyTorch 2.0+
* JupyterLab or Jupyter Notebook

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install tiktoken "tensorflow>=2.15.0" "tqdm>=4.66" matplotlib numpy jupyterlab
```

---

### Step 1: Project Organization
Ensure the pre-trained weights are located under `gpt2/345M/`:

```text
LLM-from-scratch/
├── ARCHITECTURE.md          # Technical architecture & mathematical breakdown
├── README.md                # Project overview and guide
├── gptLLM.ipynb             # Main end-to-end training and inference notebook
├── practiceNotebook.ipynb   # Modular practice notebook
├── gpt_download3.py         # OpenAI TensorFlow checkpoint extraction script
├── the-verdict.txt          # Sample text corpus
└── gpt2/
    └── 345M/                # Extracted OpenAI weights
        ├── checkpoint
        ├── encoder.json
        ├── hparams.json
        ├── model.ckpt.data-00000-of-00001
        ├── model.ckpt.index
        ├── model.ckpt.meta
        └── vocab.bpe
```

---

### Step 2: 60-Second Inference Snippet

Run this snippet in Python or Jupyter to load the model and generate text immediately:

```python
import torch
import tiktoken
from gptLLM import GPTModel, GPT_CONFIG_345M, generate, text_to_token_ids, token_ids_to_text

# 1. Automatic hardware dispatch (Apple Silicon MPS on Mac, CUDA on Nvidia, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Configure for pretrained 345M weights (requires qkv_bias=True)
CONFIG = GPT_CONFIG_345M.copy()
CONFIG.update({"context_length": 1024, "qkv_bias": True})

# 3. Instantiate model & load weights
model = GPTModel(CONFIG)
model.load_state_dict(torch.load("gpt2_345M_pretrained.pth", map_location=device))
model.to(device)
model.eval()

# 4. Generate text!
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "I love cooking"

token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer, device).to(device),
    max_new_tokens=40,
    context_size=CONFIG["context_length"],
    top_k=50,
    temperature=0.8
)

print(token_ids_to_text(token_ids, tokenizer))
```

---

## 💻 Hardware Acceleration Notes

* **Apple Silicon (Mac M-Series):** Automatically binds to `torch.device("mps")`. On devices like the Mac M5 Air with 24GB Unified Memory, tensor operations run natively on the GPU with full hardware acceleration and instant weight loading.
* **Nvidia GPUs:** Automatically binds to `torch.device("cuda")` with standard cuDNN acceleration.
* **CPU Fallback:** Seamlessly operates in standard FP32 mode if no GPU accelerator is detected.

---

## 🌟 5. Acknowledgments & Closing Thoughts

> *"What I cannot create, I do not understand."*  
> — **Richard Feynman**

Building a large language model from scratch is one of the most rewarding journeys in artificial intelligence. Demystifying the layers beneath the APIs turns mysterious "black-box AI" into pure, predictable engineering.

* **Original Paper:** [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., OpenAI, 2019)
* **Inspired by:** The educational teachings of Sebastian Raschka (*Build a Large Language Model from Scratch*) and Andrej Karpathy.

---

<div align="center">
⭐ <b>If you found this project helpful, give it a star on GitHub!</b> ⭐
</div>
