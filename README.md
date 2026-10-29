# 🧠 GPT-2 (345M) Implementation from Scratch

An end-to-end, from-scratch implementation of the **GPT-2 Medium (345M parameters)** architecture in PyTorch, featuring custom Multi-Head Causal Attention, Pre-LayerNorm residual blocks, tokenization with `tiktoken`, autoregressive generation with Top-K & temperature sampling, and automated OpenAI TensorFlow weight loading.

> 📖 **Deep Dive Documentation:** For a complete mathematical and architectural breakdown of every layer, parameter count, and tensor dimension, see [ARCHITECTURE.md](file:///Users/akashyadav/Documents/Development/Jupyter%20Notebook/LLM-from-scratch/ARCHITECTURE.md).

---

## ⚡ Key Highlights & Updates

* **345M Parameter Model (GPT-2 Medium):** Configured with 24 transformer layers, 16 attention heads, and an embedding dimension of 1024 ($d_{model} = 1024$).
* **True Causal Masking:** Fixed attention score masking (`masked_fill`) ensuring strict causality during training and coherent autoregressive text generation.
* **Vectorized Top-K Sampling:** Fully vectorized sampling in `generate()` supporting arbitrary batch sizes ($B \ge 1$) without shape broadcasting mismatches.
* **Apple Silicon & Hardware Acceleration:** Built-in dynamic device dispatch supporting **Apple Silicon MPS** (`mps`), Nvidia CUDA (`cuda`), and CPU fallback. Works effortlessly on Mac M-series laptops (e.g., M5 Air with 24GB Unified Memory).

---

## 📋 Prerequisites

Ensure you have Python 3.10+ installed along with Jupyter Notebook:

```bash
pip install torch torchvision torchaudio
pip install tiktoken "tensorflow>=2.15.0" "tqdm>=4.66" matplotlib numpy jupyterlab
```

> **Note on Apple Silicon (Mac M-series):** PyTorch natively leverages Apple's Metal Performance Shaders (MPS) for hardware acceleration on unified memory without needing any external CUDA drivers.

---

## 🚀 Setup Instructions

### 1. Project Files Structure

Ensure your project root is organized as follows:

```text
LLM-from-scratch/
├── ARCHITECTURE.md          # Complete technical architecture specification
├── README.md                # Project setup and user guide
├── gptLLM.ipynb             # Main end-to-end training & generation notebook
├── practiceNotebook.ipynb   # Step-by-step modular practice notebook
├── gpt_download3.py         # OpenAI TensorFlow checkpoint extraction script
├── the-verdict.txt          # Sample text corpus for educational fine-tuning
└── gpt2/
    └── 345M/                # Extracted OpenAI GPT-2 345M weights
        ├── checkpoint
        ├── encoder.json
        ├── hparams.json
        ├── model.ckpt.data-00000-of-00001
        ├── model.ckpt.index
        ├── model.ckpt.meta
        └── vocab.bpe
```

---

### 2. Download Pretrained GPT-2 (345M) Weights

The model can load pre-trained OpenAI weights directly into your custom PyTorch architecture.

1. Download the **GPT-2 345M weights** from [Kaggle](https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights) or OpenAI's official repository.
2. Extract the archive into a folder named `345M`.
3. Place this folder under `gpt2/345M/` in the project root.

✅ Ensure the path is exactly `gpt2/345M/` and contains `checkpoint`, `hparams.json`, and the `model.ckpt.*` files.

---

### 3. Running the Main Notebook

Launch JupyterLab or Jupyter Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Open [gptLLM.ipynb](file:///Users/akashyadav/Documents/Development/Jupyter%20Notebook/LLM-from-scratch/gptLLM.ipynb) and run the cells sequentially:

1. **Architecture Initialization:** Defines `GPT_CONFIG_345M` and builds the `GPTModel` class from scratch.
2. **Data Pipeline:** Tokenizes `the-verdict.txt` into sliding window batches using a causal dataset loader (`GPTDatasetV1`).
3. **Training Demo:** Runs a demonstration training loop showcasing how cross-entropy loss drops over epochs.
4. **Weight Transfer:** Loads official 345M parameters from `gpt2/345M/` into your custom PyTorch `GPTModel`.
5. **Autoregressive Generation:** Generates coherent continuations from user prompts using top-$k$ and temperature scaling.

---

## 💻 Hardware & Performance Notes

* **Apple Silicon (Mac M5 Air 24GB Unified RAM):**
  The model will automatically utilize `torch.device("mps")`. 
  With 24GB unified memory, the 345M parameter model (~1.4 GB in FP32) fits with massive headroom for inference and fine-tuning.
* **Nvidia GPUs:**
  If running on Linux/Windows with CUDA, PyTorch will automatically bind to `torch.device("cuda")`.
* **CPU Fallback:**
  If neither MPS nor CUDA is available, computations cleanly execute on the CPU.

---

## 🔬 Model Configurations Reference

You can switch between model sizes at any time in the notebook:

```python
# GPT-2 Small (124M parameters)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# GPT-2 Medium (345M parameters) - Active Model
GPT_CONFIG_345M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```
