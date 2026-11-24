"""
Self-Attention Extraction and Visualization Module
Extracts authentic multi-head attention weights from BERT Transformer layers.
Provides mathematical breakdowns, attention matrices, and interactive Plotly heatmaps.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
from .embeddings import EmbeddingExtractor

BERT_CACHE_NAME = "bert-base-uncased"

class AttentionExtractor:
    """Extracts authentic attention weights from BERT transformer layers."""
    _tokenizer = None
    _model = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()
        self.embed_extractor = EmbeddingExtractor()

    @classmethod
    def _init_model(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(BERT_CACHE_NAME, local_files_only=True)
        if cls._model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._model = AutoModel.from_pretrained(BERT_CACHE_NAME, output_attentions=True, local_files_only=True)
            cls._model.to(device)
            cls._model.eval()

    def get_attention_data(self, sentence: str) -> Dict[str, Any]:
        """
        Extract genuine multi-layer, multi-head attention tensors for a sentence.
        Returns full attention weights, token lists, and bank token focus.
        """
        self._init_model()
        inputs = self._tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        bank_indices = self.embed_extractor.find_bank_token_indices(sentence)
        bank_idx = bank_indices[0] if bank_indices else 1
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # outputs.attentions: tuple of 12 layers, each [1, 12, seq_len, seq_len]
            attentions = [layer.squeeze(0).cpu().numpy() for layer in outputs.attentions]

        num_layers = len(attentions)
        num_heads = attentions[0].shape[0]
        seq_len = len(tokens)

        # Compute summary attention matrices
        # Last layer attention (Layer 12, index 11)
        last_layer_heads = attentions[-1] # [12, seq_len, seq_len]
        last_layer_avg = np.mean(last_layer_heads, axis=0) # [seq_len, seq_len]

        # Overall model attention averaged across all 12 layers and 12 heads
        all_layers_avg = np.mean([np.mean(layer, axis=0) for layer in attentions], axis=0)

        # Context words that 'bank' attends to (row bank_idx)
        bank_attends_to = last_layer_avg[bank_idx]
        # Context words that attend to 'bank' (col bank_idx)
        attends_to_bank = last_layer_avg[:, bank_idx]

        # Rank context tokens (ignoring [CLS], [SEP], and 'bank' itself)
        ranked_context = []
        for i, tok in enumerate(tokens):
            if i != bank_idx and tok not in ["[CLS]", "[SEP]", ".", ",", "?", "!"]:
                clean_tok = tok.replace("##", "")
                score = float(bank_attends_to[i])
                ranked_context.append({"token": clean_tok, "position": i, "attention_weight": round(score, 4)})
                
        ranked_context.sort(key=lambda x: x["attention_weight"], reverse=True)

        return {
            "sentence": sentence,
            "tokens": tokens,
            "bank_token_idx": bank_idx,
            "bank_token_str": tokens[bank_idx],
            "num_layers": num_layers,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "last_layer_avg_matrix": last_layer_avg,
            "all_layers_avg_matrix": all_layers_avg,
            "layer_12_heads": last_layer_heads,
            "ranked_context": ranked_context,
            "bank_attends_to_vector": bank_attends_to.tolist(),
            "attends_to_bank_vector": attends_to_bank.tolist(),
        }

    @staticmethod
    def create_heatmap_figure(
        matrix: np.ndarray,
        tokens: List[str],
        title: str = "Self-Attention Weight Matrix",
        highlight_idx: Optional[int] = None
    ) -> go.Figure:
        """Create a clean, publication-grade interactive Plotly heatmap with robust token indexing."""
        seq_len = len(tokens)
        indices = list(range(seq_len))
        clean_tokens = [f"<b>{t}</b>" if i == highlight_idx else t for i, t in enumerate(tokens)]
        
        # Format custom hover text
        hover_text = []
        for i in range(seq_len):
            row_hover = []
            for j in range(seq_len):
                row_hover.append(
                    f"Query Token: {tokens[i]} (pos {i})<br>"
                    f"Key Token: {tokens[j]} (pos {j})<br>"
                    f"Attention Weight: {matrix[i][j]:.4f}"
                )
            hover_text.append(row_hover)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=indices,
                y=indices,
                text=hover_text,
                hoverinfo="text",
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Weight")
            )
        )
        
        fig.update_layout(
            title={
                "text": title,
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 16, "family": "Inter, Arial, sans-serif", "color": "#0F2027"}
            },
            xaxis=dict(
                title="Key Tokens (Context Attended To)",
                tickmode="array",
                tickvals=indices,
                ticktext=clean_tokens,
                tickangle=45
            ),
            yaxis=dict(
                title="Query Tokens (Attending From)",
                tickmode="array",
                tickvals=indices,
                ticktext=clean_tokens,
                autorange="reversed"
            ),
            width=680,
            height=580,
            margin=dict(l=60, r=40, t=60, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def create_token_importance_bar(ranked_context: List[Dict[str, Any]], title: str = "Contextual Words Attending to 'Bank'") -> go.Figure:
        """Create an interactive bar chart of the top contextual words."""
        top_words = ranked_context[:8]
        tokens = [w["token"] for w in top_words][::-1]
        scores = [w["attention_weight"] for w in top_words][::-1]

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=tokens,
                orientation="h",
                marker=dict(
                    color=scores,
                    colorscale="Teal",
                    showscale=False
                ),
                text=[f"{s:.4f}" for s in scores],
                textposition="outside",
                hovertemplate="Word: %{y}<br>Attention Weight: %{x:.4f}<extra></extra>"
            )
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 14, "color": "#0F2027"}},
            xaxis_title="Attention Probability Weight",
            yaxis_title="Context Token",
            height=340,
            margin=dict(l=60, r=40, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
