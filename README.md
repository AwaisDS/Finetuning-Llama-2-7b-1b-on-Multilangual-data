# Fine-Tuning Llama-2 (7B / 1B) on Multilingual Data

This repository contains an end-to-end guide and Jupyter Notebook (`Fine_tune_Llama_2.ipynb`) demonstrating how to fine-tune Meta's **Llama 2** on multilingual dataset(s) using **Parameter-Efficient Fine-Tuning (PEFT)** with **QLoRA (4-bit Quantization)**.

---

## 📌 Project Overview

Fine-tuning large language models (LLMs) from scratch is computationally expensive. This notebook leverages **QLoRA (Quantized Low-Rank Adaptation)** to drastically reduce memory usage, allowing a 7B-parameter model to be fine-tuned on consumer GPUs or standard cloud environments (e.g., Google Colab T4/A100).

### Key Features
- **4-bit NormalFloat (NF4) Quantization:** Using `bitsandbytes` to load model weights in 4-bit precision.
- **LoRA Adapter Tuning:** Training low-rank decomposition matrices while freezing base model weights.
- **Multilingual Instruction Tuning:** Adapting instruction/response pairs across multiple languages.
- **Hugging Face Ecosystem:** Built with `transformers`, `peft`, `trl` (`SFTTrainer`), and `datasets`.

---

## 🛠 Prerequisites & Setup

### 1. Hardware Requirements
- **GPU:** Minimum 12 GB VRAM (e.g., NVIDIA T4, V100, A100).
- **RAM:** Minimum 12 GB system memory.

### 2. Hugging Face Access
Meta's Llama 2 models require accepted license terms:
1. Request access on [Meta's Llama 2 page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and the corresponding [Hugging Face model page](https://huggingface.co/meta-llama/Llama-2-7b-hf).
2. Authenticate in your environment via the CLI or notebook:
   ```python
   from huggingface_hub import login
   login("YOUR_HF_TOKEN")
