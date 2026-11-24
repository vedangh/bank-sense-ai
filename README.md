# Self-Attention Resolves "Bank" – Contextual Word Embeddings & Word Sense Disambiguation

An academic, full-stack Machine Learning and Natural Language Processing project demonstrating how Transformer-based self-attention solves lexical polysemy for the English word **"bank"** (Financial Institution vs. River Land), in sharp contrast to static Word2Vec embeddings.

---

## 🌟 Project Highlights

- 🧠 **Transformer Architecture:** Pretrained BERT (`bert-base-uncased`) with genuine multi-head self-attention weight extraction.
- 🔤 **Static Embedding Baseline:** PyTorch CBOW Word2Vec architecture trained directly on the corpus to demonstrate vector invariance.
- 📊 **Empirical Dataset:** 2,100 balanced, cleaned sentence pairs (1,050 Financial, 1,050 River), partitioned into Stratified Train (70%), Validation (15%), and Test (15%) splits.
- 🎯 **Word Sense Disambiguation:** Fine-tuned Sequence Classifier achieving **100.0% Accuracy and F1-Score** on the held-out 315-sentence test set.
- 📈 **Interpretability & Visualization:** Interactive Plotly Self-Attention Heatmaps (Layers 1–12, Heads 0–11), Contextual Token Importance rankings, Confusion Matrix with TP/TN/FP/FN analysis, and 2D/3D PCA embedding projections.
- 💻 **Zero Paid APIs:** 100% locally executed open-source stack (PyTorch CPU/GPU, HuggingFace Transformers, Streamlit, Scikit-Learn, Plotly). Fully compatible with Python 3.10–3.14 on Windows.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Windows 10/11 (or Linux / macOS)

### 2. Installation
Open your terminal (PowerShell or Command Prompt) in the project directory and install the requirements:

```bash
pip install -r Bank_Sense_AI/requirements.txt
```

### 3. Running the Streamlit Application
Launch the web dashboard with a single command from this root folder:

```bash
streamlit run app.py
```

Or run directly inside `Bank_Sense_AI`:
```bash
cd Bank_Sense_AI
streamlit run app.py
```

Or simply double-click `run.bat`.

The application will open in your default browser at `http://localhost:8501`.
