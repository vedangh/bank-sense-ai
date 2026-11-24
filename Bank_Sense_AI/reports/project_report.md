# Academic Project Report

# Self-Attention Resolves "Bank" – Contextual Word Embeddings and Word Sense Disambiguation

**Course:** Natural Language Processing & Machine Learning Capstone  
**Author:** AI/ML Research Team  
**Date:** Academic Year 2026–2027  
**Technology Stack:** Python 3.14 / 3.12, PyTorch, HuggingFace Transformers, Streamlit, Scikit-Learn, Plotly  

---

## 1. Abstract

Lexical polysemy—the capacity for a single lexical unit to carry distinct semantic interpretations depending upon surrounding sentential context—presents a foundational challenge in computational linguistics. The English noun *"bank"* constitutes the classic benchmark of homonymy/polysemy, possessing both a commercial financial interpretation (*"I deposited my salary in the bank"*) and a riparian hydrological interpretation (*"The fisherman sat beside the bank of the river"*).

Classical distributed semantic models, notably Word2Vec (Mikolov et al., 2013), project each vocabulary item into a singular static vector $\mathbf{v} \in \mathbb{R}^d$. This structural limitation collapses competing semantic senses into an uninterpretable compromise vector. Conversely, the Transformer self-attention mechanism (Vaswani et al., 2017) computes context-dependent dynamic representations by routing contextual token interactions through query, key, and value matrices.

This project delivers a complete, reproducible, fully locally executed machine learning system that empirically demonstrates how self-attention resolves polysemy. Using a curated, balanced dataset of 2,100 validated sentences, we train both a static Continuous Bag of Words (CBOW) Word2Vec baseline and a fine-tuned BERT sequence classifier. On a held-out test set of 315 balanced sentences, the contextual model achieved **100.0% accuracy, precision, recall, and F1-score**, yielding a clean topological separation in 2D/3D PCA projection space. An interactive, publication-grade 8-page Streamlit web dashboard is implemented to visualize real-time inference, token importance, and authentic multi-head attention heatmaps.

---

## 2. Introduction & Problem Statement

### 2.1 The Linguistic Phenomenon of Polysemy
Human natural language is inherently ambiguous and non-injective: a finite set of phonetic and orthographic symbols maps to an infinite set of semantic propositions. In computational semantics, Word Sense Disambiguation (WSD) is the task of programmatically determining which sense of a word is activated in a given sentence.

Consider the word **"bank"**:
- **Sense $S_1$ (Financial):** A financial institution that accepts deposits, extends loans, and facilitates fiscal transactions.
- **Sense $S_2$ (River):** The sloping terrestrial embankment bordering a river, stream, or body of water.

### 2.2 The Collapse of Static Embeddings
Static distributed representations represent the lexicon as a parameter matrix $\mathbf{W}_{\text{embed}} \in \mathbb{R}^{|V| \times d}$. The vector lookup function is context-insensitive:

$$\mathbf{v}_{\text{bank}} = \mathbf{W}_{\text{embed}}[\text{index}(\text{"bank"})]$$

When exposed to corpora containing both senses, the gradient updates pull $\mathbf{v}_{\text{bank}}$ toward financial tokens ($\nabla_{\text{finance}}$) and river tokens ($\nabla_{\text{river}}$), settling at their geometric mean:

$$\mathbf{v}_{\text{bank}} \approx \alpha \mathbf{v}_{\text{financial}} + (1 - \alpha) \mathbf{v}_{\text{river}}$$

This conflation corrupts downstream tasks, as the vector is unable to align cleanly with either conceptual cluster.

### 2.3 The Contextual Paradigm Shift
Bidirectional Encoder Representations from Transformers (BERT; Devlin et al., 2018) replaces the lookup table with a deep encoder function parameterized by self-attention layers:

$$\mathbf{h}_{\text{bank}}^{(L)} = f_{\text{BERT}}(\text{"bank"} \mid \text{sentence context})$$

Where $\mathbf{h}_{\text{bank}}^{(L)}$ is dynamically conditioned upon every other token in the sentence.

---

## 3. Mathematical Methodology

### 3.1 Static Word2Vec (Continuous Bag of Words)
The CBOW architecture predicts a target word $w_t$ given a symmetric context window of radius $c$:

$$\mathcal{C} = \{w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+c}\}$$

The context vector is the mean of the input embeddings:

$$\mathbf{v}_{\mathcal{C}} = \frac{1}{2c} \sum_{w \in \mathcal{C}} \mathbf{W}_{\text{in}}[w]$$

The conditional probability of the target word is computed via softmax over the output projection $\mathbf{W}_{\text{out}}$:

$$P(w_t \mid \mathcal{C}) = \frac{\exp(\mathbf{W}_{\text{out}}[w_t]^T \mathbf{v}_{\mathcal{C}})}{\sum_{k=1}^{|V|} \exp(\mathbf{W}_{\text{out}}[w_k]^T \mathbf{v}_{\mathcal{C}})}$$

The objective is to minimize cross-entropy loss over the corpus:

$$\mathcal{L}_{\text{CBOW}} = -\sum_{t=1}^{T} \log P(w_t \mid \mathcal{C}_t)$$

Crucially, once training concludes, $\mathbf{W}_{\text{in}}[\text{"bank"}]$ is frozen and immutable regardless of whether the query sentence mentions mortgages or fishing rods.

### 3.2 Transformer Multi-Head Self-Attention
In BERT, the input sequence of $n$ tokens is first mapped to initial embeddings $\mathbf{H}^{(0)} \in \mathbb{R}^{n \times d_{\text{model}}}$ (where $d_{\text{model}} = 768$).

At each layer $l \in [1, 12]$, multi-head self-attention computes $h = 12$ independent attention heads with projection dimension $d_k = d_v = \frac{d_{\text{model}}}{h} = 64$.

For each head $i \in [1, h]$:
1. **Query projection:** $\mathbf{Q}_i = \mathbf{H}^{(l-1)} \mathbf{W}_i^Q$
2. **Key projection:** $\mathbf{K}_i = \mathbf{H}^{(l-1)} \mathbf{W}_i^K$
3. **Value projection:** $\mathbf{V}_i = \mathbf{H}^{(l-1)} \mathbf{W}_i^V$

Pairwise attention weights are computed using the scaled dot-product:

$$\mathbf{A}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}$$

Where $A_{i, j, k}$ represents the probability weight that token $j$ assigns to token $k$. The contextualized output for head $i$ is:

$$\text{Head}_i = \mathbf{A}_i \mathbf{V}_i \in \mathbb{R}^{n \times d_k}$$

The heads are concatenated and linearly transformed:

$$\text{MultiHead}(\mathbf{H}^{(l-1)}) = \left[\text{Head}_1 \mathbin{\Vert} \text{Head}_2 \mathbin{\Vert} \dots \mathbin{\Vert} \text{Head}_h\right] \mathbf{W}^O$$

Following residual connections and Layer Normalization:

$$\mathbf{H}^{(l)} = \text{LayerNorm}(\mathbf{H}^{(l-1)} + \text{MultiHead}(\mathbf{H}^{(l-1)}))$$
$$\mathbf{H}^{(l)} = \text{LayerNorm}(\mathbf{H}^{(l)} + \text{FFN}(\mathbf{H}^{(l)}))$$

---

## 4. Dataset Engineering & Preparation

A primary vulnerability of experimental NLP evaluations is class imbalance and sample artifacting. To ensure rigorous, reproducible training, a custom corpus was engineered with the following specifications:

| Parameter | Specification |
| :--- | :--- |
| **Total Sentences** | 2,100 unique sentences |
| **Class Distribution** | 1,050 Financial (50.0%), 1,050 River (50.0%) |
| **Target Keyword** | Exact occurrence of whole word `"bank"` or `"banks"` in every sample |
| **Deduplication** | Strict string deduplication |
| **Stratified Split** | **Train:** 1,470 (70%) \| **Val:** 315 (15%) \| **Test:** 315 (15%) |
| **Average Word Count** | 12.4 words per sentence |

Sentences were engineered to encompass active, passive, fronted prepositional, and subordinate clause syntactic variants with realistic lexical collocations.

---

## 5. Experimental Results & Evaluation

### 5.1 Held-Out Test Set Performance
The fine-tuned BERT model was evaluated on the 315 held-out test samples (158 Financial, 157 River) with zero exposure during training:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Financial** | 1.0000 | 1.0000 | 1.0000 | 158 |
| **River** | 1.0000 | 1.0000 | 1.0000 | 157 |
| **Macro Average** | **1.0000** | **1.0000** | **1.0000** | **315** |
| **Weighted Average** | **1.0000** | **1.0000** | **1.0000** | **315** |
| **Overall Accuracy** | — | — | **100.0%** | **315** |

### 5.2 Confusion Matrix Breakdown
- **True Positives (Financial $\rightarrow$ Financial):** 158
- **True Negatives (River $\rightarrow$ River):** 157
- **False Positives (River $\rightarrow$ Financial):** 0
- **False Negatives (Financial $\rightarrow$ River):** 0

### 5.3 Empirical Cosine Similarity Demonstration

Cosine similarity between the extracted vectors $\mathbf{u}, \mathbf{v}$ is given by:

$$\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

| Vector Pair Evaluated | Model Type | Cosine Similarity | Interpretation |
| :--- | :---: | :---: | :--- |
| $\text{"bank"}_{\text{fin1}} \leftrightarrow \text{"bank"}_{\text{fin2}}$ | BERT (Contextual) | **0.8412** | High affinity within same semantic domain |
| $\text{"bank"}_{\text{riv1}} \leftrightarrow \text{"bank"}_{\text{riv2}}$ | BERT (Contextual) | **0.8179** | High affinity within same semantic domain |
| $\text{"bank"}_{\text{fin1}} \leftrightarrow \text{"bank"}_{\text{riv1}}$ | BERT (Contextual) | **0.6204** | Marked topological separation across senses |
| $\text{"bank"} \leftrightarrow \text{"bank"}$ | Word2Vec (Static) | **1.0000** | Total invariance; completely insensitive to context |

### 5.4 Principal Component Analysis (PCA)
PCA reduction of the 768-dimensional contextual vectors of `"bank"` onto the first two principal components accounts for **59.21% of the total variance** (PC1: 55.07%, PC2: 4.14%). The 2D scatter plot demonstrates two completely disjoint, linearly separable clusters.

---

## 6. Software Architecture & GUI Implementation

The project includes an 8-page interactive Streamlit web dashboard:
1. **Home Dashboard:** Executive overview and key metrics.
2. **Dataset Explorer:** Full dataset filtering, stats, CSV download, and custom CSV uploader.
3. **Model Training:** Interactive training interfaces for Word2Vec and BERT with loss/accuracy curves.
4. **Bank Meaning Prediction:** User input validation, sense prediction, confidence score, and token importance ranking.
5. **Self-Attention Visualization:** Interactive Query-Key-Value attention heatmaps (all 12 layers and 12 heads).
6. **Model Comparison:** Side-by-side architectural analysis and live cosine similarity calculation.
7. **Evaluation & Metrics:** Interactive Confusion Matrix, Classification Report, and 2D PCA cluster visualization.
8. **Algorithm Explanation:** Detailed pedagogical breakdown of polysemy and transformer mathematics.

---

## 7. Conclusion

This research project demonstrates that the attention mechanism in deep Transformers resolves the fundamental flaw of static word embeddings. By allowing token representations to be dynamically steered by their surrounding context, BERT creates cleanly separated semantic manifolds for polysemous words. All experimental code, trained model checkpoints, and evaluation artifacts are stored locally and are fully reproducible on consumer CPU hardware.
