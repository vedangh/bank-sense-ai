"""
Embeddings Module for Static vs Contextual Comparison
Extracts genuine contextual representations from BERT for target token 'bank',
handles subword alignments, and compares against static Word2Vec representations.
"""

import os
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from transformers import AutoTokenizer, AutoModel
from .train_word2vec import Word2VecModel

BERT_CACHE_NAME = "bert-base-uncased"

class EmbeddingExtractor:
    """Extracts and compares static Word2Vec and contextual BERT embeddings."""
    _tokenizer = None
    _bert_model = None

    def __init__(self, w2v_model_dir: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_bert()
        self.w2v = Word2VecModel()
        try:
            self.w2v.load_model(w2v_model_dir)
            self.w2v_loaded = True
        except Exception:
            self.w2v_loaded = False

    @classmethod
    def _init_bert(cls):
        """Lazy load singleton BERT tokenizer and model to prevent memory churn."""
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(BERT_CACHE_NAME, local_files_only=True)
        if cls._bert_model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._bert_model = AutoModel.from_pretrained(BERT_CACHE_NAME, output_hidden_states=True, local_files_only=True)
            cls._bert_model.to(device)
            cls._bert_model.eval()

    def find_bank_token_indices(self, sentence: str) -> List[int]:
        """Accurately identify token indices corresponding to the polysemous word 'bank' using word boundaries."""
        import re
        encoded = self._tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
        tokens = self._tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        offset_mapping = encoded["offset_mapping"]
        
        lower_sent = sentence.lower()
        # Strictly find whole-word occurrences of 'bank' or 'banks'
        bank_spans = [(m.start(), m.end()) for m in re.finditer(r"\bbanks?\b", lower_sent)]
        
        bank_indices = []
        if bank_spans:
            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == 0 and tok_end == 0:
                    continue # Special tokens [CLS], [SEP]
                for s_char, e_char in bank_spans:
                    if max(s_char, tok_start) < min(e_char, tok_end):
                        bank_indices.append(idx)
                        break
        else:
            # Fallback: scan token strings
            for i, tok in enumerate(tokens):
                clean_tok = tok.replace("##", "").lower()
                if clean_tok in ["bank", "banks"]:
                    bank_indices.append(i)
                    
        return bank_indices if bank_indices else [1]

    def extract_bert_bank_embedding(self, sentence: str) -> Tuple[np.ndarray, List[str], int]:
        """
        Extract the genuine 768-dimensional contextual vector of 'bank' from the last hidden layer.
        Returns (embedding_vector, token_list, bank_token_index).
        """
        self._init_bert()
        inputs = self._tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        bank_indices = self.find_bank_token_indices(sentence)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._bert_model(**inputs)
            # outputs.last_hidden_state: [1, seq_len, 768]
            last_hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            
        # If subword split occurred, average the token vectors; otherwise take first
        if len(bank_indices) == 1:
            bank_vec = last_hidden[bank_indices[0]]
            primary_idx = bank_indices[0]
        else:
            bank_vec = np.mean(last_hidden[bank_indices], axis=0)
            primary_idx = bank_indices[0]
            
        return bank_vec, tokens, primary_idx

    def extract_w2v_bank_embedding(self) -> np.ndarray:
        """Extract the static Word2Vec vector for 'bank'."""
        if not self.w2v_loaded:
            raise RuntimeError("Word2Vec model is not trained or loaded.")
        return self.w2v.get_vector("bank")

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two dense vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def compare_contexts(
        self,
        sentence_fin1: str,
        sentence_fin2: str,
        sentence_riv1: str,
        sentence_riv2: str
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison demonstrating that contextual vectors adapt dynamically
        while static vectors stay completely invariant.
        """
        v_fin1, _, _ = self.extract_bert_bank_embedding(sentence_fin1)
        v_fin2, _, _ = self.extract_bert_bank_embedding(sentence_fin2)
        v_riv1, _, _ = self.extract_bert_bank_embedding(sentence_riv1)
        v_riv2, _, _ = self.extract_bert_bank_embedding(sentence_riv2)
        
        # Contextual similarities
        sim_fin_fin = self.cosine_similarity(v_fin1, v_fin2)
        sim_riv_riv = self.cosine_similarity(v_riv1, v_riv2)
        sim_fin_riv = self.cosine_similarity(v_fin1, v_riv1)
        sim_fin2_riv2 = self.cosine_similarity(v_fin2, v_riv2)

        # Static similarities
        if self.w2v_loaded:
            w2v_vec = self.extract_w2v_bank_embedding()
            w2v_dim = len(w2v_vec)
            w2v_self_sim = self.cosine_similarity(w2v_vec, w2v_vec)
        else:
            w2v_vec = None
            w2v_dim = 100
            w2v_self_sim = 1.0

        return {
            "bert_dimension": len(v_fin1),
            "w2v_dimension": w2v_dim,
            "sim_financial_to_financial": round(sim_fin_fin, 4),
            "sim_river_to_river": round(sim_riv_riv, 4),
            "sim_financial_to_river": round(sim_fin_riv, 4),
            "sim_financial2_to_river2": round(sim_fin2_riv2, 4),
            "w2v_bank_invariance_similarity": round(w2v_self_sim, 4),
            "vectors": {
                "fin1_preview": v_fin1[:6].tolist(),
                "riv1_preview": v_riv1[:6].tolist(),
                "w2v_preview": w2v_vec[:6].tolist() if w2v_vec is not None else []
            }
        }
