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
pip install -r requirements.txt
```

### 3. Running the Streamlit Application
Launch the web dashboard with a single command:

```bash
streamlit run app.py
```

Or on Windows, simply double-click `run.bat`.

The application will open in your default browser at `http://localhost:8501`.

---

## 🏛️ Project Architecture

```
Bank_Sense_AI/
│
├── app.py                      # Main 8-Page Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── run.bat                     # Windows one-click launcher
│
├── data/
│   ├── bank_sense_dataset.csv  # 2,100 balanced sentences (1,050 financial, 1,050 river)
│   ├── train.csv               # 70% Stratified Training Split (1,470 samples)
│   ├── val.csv                 # 15% Stratified Validation Split (315 samples)
│   └── test.csv                # 15% Stratified Held-out Test Split (315 samples)
│
├── models/
│   ├── word2vec_model/         # Trained Word2Vec weights, metadata & vocabulary
│   ├── bert_model/             # Fine-tuned BERT weights & training history
│   ├── evaluation_metrics.json # Calculated test metrics (Accuracy, F1, Confusion Matrix)
│   └── pca_embeddings.json     # Precomputed 2D/3D PCA coordinates of test embeddings
│
├── src/
│   ├── data_preprocessing.py   # Text normalization, tokenization, validation
│   ├── dataset_loader.py       # Dataset loaders, split managers, custom upload
│   ├── train_word2vec.py       # PyTorch CBOW neural Word2Vec implementation
│   ├── train_bert.py           # BERT fine-tuning & sequence classification manager
│   ├── embeddings.py           # Target token locator, contextual & static extractors
│   ├── attention_visualization.py # Multi-head attention extraction & Plotly heatmaps
│   ├── prediction.py           # Unified inference & interpretability engine
│   └── evaluation.py           # Metric calculation, confusion matrix, PCA projections
│
└── reports/
    └── project_report.md       # Comprehensive academic research report
```

---

## 🖥️ 8-Page Dashboard Navigation

1. **Page 1 – Home Dashboard:** High-level project summary, dataset KPIs, architecture badges, and problem overview.
2. **Page 2 – Dataset Explorer:** Search and filter 2,100 records, inspect class distribution, sentence length histograms, CSV download, and custom CSV uploader.
3. **Page 3 – Model Training:** Interactive training interfaces for Word2Vec and BERT with loss and accuracy curves.
4. **Page 4 – Bank Meaning Prediction:** Real-time sense classification (FINANCIAL vs. RIVER), confidence score, class probabilities, attention anchors, and Word2Vec contrast.
5. **Page 5 – Self-Attention Visualization:** Interactive Query-Key-Value attention heatmaps for any sentence across individual layers and heads.
6. **Page 6 – Model Comparison:** Comprehensive architectural comparison and empirical cosine similarity calculations demonstrating static invariance vs contextual flexibility.
7. **Page 7 – Evaluation & Metrics:** Held-out test set accuracy, precision, recall, F1, interactive Confusion Matrix, classification report, and 2D PCA sense cluster scatter plot.
8. **Page 8 – Algorithm Explanation:** In-depth academic breakdown of polysemy, transformer mathematics, and system methodology.

---

## 📊 Evaluation Results (Held-Out Test Set)

| Metric | Financial Class | River Class | Macro Average | Weighted Average |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | — | — | **100.0%** | **100.0%** |
| **Precision** | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| **Recall** | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| **F1-Score** | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| **Support** | 158 | 157 | 315 | 315 |

### Confusion Matrix Breakdown
- **True Positives (Financial → Financial):** 158
- **True Negatives (River → River):** 157
- **False Positives (River → Financial):** 0
- **False Negatives (Financial → River):** 0

---

## 🔬 Core Concept: Why Word2Vec Fails & Self-Attention Succeeds

### Static Word2Vec
$$\mathbf{v}_{\text{bank}} = \text{EmbeddingTable}[\text{idx}]$$
Every occurrence of `"bank"` receives the **exact same 100-dimensional vector**, causing semantic conflation.

### Transformer Self-Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
In *"I deposited my salary in the bank"*, the token `"bank"` attends strongly to *"salary"* and *"deposited"*, dynamically pulling its representation into the financial subspace. In *"The fisherman sat beside the bank of the river"*, `"bank"` attends to *"river"* and *"fisherman"*, shifting into the riparian subspace.
