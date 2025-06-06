# 🧠 GPT-2 Notebook Guide

This project involves running a Jupyter Notebook to perform tasks using GPT-2 (124M parameters). Follow the steps below to get everything set up correctly and running smoothly.

---


## 📋 Prerequisites

Make sure you have the following installed:
- Python 3.7+
- Jupyter Notebook


---


## 🚀 Setup Instructions

### 1. **Open the Notebook**

Open the main gptLLM.ipynb notebook file using Jupyter Notebook:



### 2. **Run Cells**
Run all the cells up to and including the Training Loop. These cells prepare the environment and show case a small DEMO of LLM which uses the-verdict.txt and how training loop helps in generating better result.

### Step 3: Download GPT-2 (124M) Weights

- Visit [Kaggle](https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights) and download the **GPT-2 124M weights**.
- Extract the archive after downloading. You should get a folder named `124M` containing files like:
    checkpoint encoder.json hparams.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta vocab.bpe

### 📁 Step 4: Organize the Weights
In the root of this project, create a folder named `gpt2`, and place the extracted `124M` folder inside it.

✅ **Make sure the path is exactly `gpt2/124M/` and that the filenames are correct.**


### ▶️ Step 5: Continue Running the Notebook

After setting up the weights, go back to the notebook and **continue running the remaining cells after cell #140**.


---
### ▶NOTE : 
I have used CUDA everywhere but in Training loop I have used CPU because of low GPU Memory. If you want to use CUDA simply replace device = torch.device("cpu") with device = torch.device("cuda")


This will:

- ✅ Load GPT-2 with the 124M parameters  
- 🚀 Generate improved results based on the model
