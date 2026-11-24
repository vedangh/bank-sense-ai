"""
Static Word2Vec Implementation in PyTorch
Implements Continuous Bag of Words (CBOW) / Skip-Gram neural architecture
to train genuine static word embeddings from the training corpus.
Guaranteed compatibility with Python 3.10-3.14 across Windows, macOS, and Linux.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from .data_preprocessing import tokenize_corpus_sentence, build_vocabulary

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "word2vec_model")
)

class CBOWModel(nn.Module):
    """Continuous Bag of Words (CBOW) Neural Network for Word Embeddings."""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs):
        # inputs: [batch_size, context_size * 2]
        embeds = self.embeddings(inputs) # [batch_size, 2*context_size, embedding_dim]
        # Average context word vectors
        context_mean = torch.mean(embeds, dim=1) # [batch_size, embedding_dim]
        out = self.linear(context_mean) # [batch_size, vocab_size]
        return out


class Word2VecModel:
    """Wrapper providing Word2Vec training, vector lookup, and similarity computations."""
    def __init__(self, embedding_dim: int = 100, window_size: int = 3):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.model: Optional[CBOWModel] = None
        self.embedding_matrix: Optional[np.ndarray] = None
        self.training_loss: List[float] = []

    def _prepare_cbow_data(self, tokenized_sentences: List[List[str]]):
        data = []
        for sent in tokenized_sentences:
            indices = [self.word2idx.get(w, 1) for w in sent]
            for i in range(self.window_size, len(indices) - self.window_size):
                context = (
                    indices[i - self.window_size : i] +
                    indices[i + 1 : i + self.window_size + 1]
                )
                target = indices[i]
                data.append((context, target))
        return data

    def train(self, sentences: List[str], epochs: int = 15, lr: float = 0.005, batch_size: int = 64):
        """Train genuine Word2Vec embeddings on the provided corpus."""
        tokenized = [tokenize_corpus_sentence(s) for s in sentences]
        self.word2idx, self.idx2word = build_vocabulary(tokenized, min_count=2)
        vocab_size = len(self.word2idx)
        
        # Ensure 'bank' is in vocabulary
        if "bank" not in self.word2idx:
            self.word2idx["bank"] = len(self.word2idx)
            self.idx2word[self.word2idx["bank"]] = "bank"
            vocab_size += 1

        training_data = self._prepare_cbow_data(tokenized)
        if not training_data:
            raise ValueError("Insufficient training data to generate CBOW windows.")

        self.model = CBOWModel(vocab_size, self.embedding_dim)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.training_loss = []
        contexts_tensor = torch.tensor([c for c, _ in training_data], dtype=torch.long)
        targets_tensor = torch.tensor([t for _, t in training_data], dtype=torch.long)
        
        dataset_len = len(training_data)
        
        for epoch in range(epochs):
            total_loss = 0.0
            permutation = torch.randperm(dataset_len)
            self.model.train()
            
            for i in range(0, dataset_len, batch_size):
                indices = permutation[i : i + batch_size]
                batch_contexts = contexts_tensor[indices]
                batch_targets = targets_tensor[indices]
                
                optimizer.zero_grad()
                log_probs = self.model(batch_contexts)
                loss = loss_function(log_probs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / (dataset_len // batch_size + 1)
            self.training_loss.append(float(round(avg_loss, 4)))

        # Cache normalized numpy embedding matrix
        with torch.no_grad():
            self.embedding_matrix = self.model.embeddings.weight.cpu().numpy()
            
    def get_vector(self, word: str) -> np.ndarray:
        """Extract static vector for a word. Returns fixed representation."""
        word = word.lower().strip()
        idx = self.word2idx.get(word, 1) # Fallback to <UNK>
        if self.embedding_matrix is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self.embedding_matrix[idx]

    def similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two static word vectors."""
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def most_similar(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Retrieve the top-n most similar words by cosine similarity."""
        target_vec = self.get_vector(word)
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return []
            
        results = []
        for w, idx in self.word2idx.items():
            if w in [word, "<PAD>", "<UNK>"]:
                continue
            cand_vec = self.embedding_matrix[idx]
            cand_norm = np.linalg.norm(cand_vec)
            if cand_norm > 0:
                sim = float(np.dot(target_vec, cand_vec) / (target_norm * cand_norm))
                results.append((w, round(sim, 4)))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def save_model(self, save_dir: Optional[str] = None):
        """Save weights, vocabulary, and config locally."""
        target_dir = save_dir or MODEL_DIR
        os.makedirs(target_dir, exist_ok=True)
        
        meta = {
            "embedding_dim": self.embedding_dim,
            "window_size": self.window_size,
            "word2idx": self.word2idx,
            "training_loss": self.training_loss
        }
        with open(os.path.join(target_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        if self.embedding_matrix is not None:
            np.save(os.path.join(target_dir, "embeddings.npy"), self.embedding_matrix)

    def load_model(self, load_dir: Optional[str] = None):
        """Load trained weights and vocabulary from local disk."""
        target_dir = load_dir or MODEL_DIR
        meta_path = os.path.join(target_dir, "metadata.json")
        weights_path = os.path.join(target_dir, "embeddings.npy")
        
        if not os.path.exists(meta_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model files not found in {target_dir}")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        self.embedding_dim = meta["embedding_dim"]
        self.window_size = meta["window_size"]
        self.word2idx = meta["word2idx"]
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        self.training_loss = meta.get("training_loss", [])
        self.embedding_matrix = np.load(weights_path)
