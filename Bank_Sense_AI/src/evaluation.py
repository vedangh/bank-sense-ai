"""
Model Evaluation and Metrics Module
Calculates authentic test set metrics: Accuracy, Precision, Recall, F1,
Confusion Matrix with TP/TN/FP/FN breakdown, Classification Report,
and PCA/t-SNE contextual embedding projections.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from .train_bert import BertClassifierManager, LABEL2ID, ID2LABEL
from .embeddings import EmbeddingExtractor

class ModelEvaluator:
    """Computes comprehensive evaluation metrics and visualizations on held-out test data."""
    def __init__(self):
        self.classifier = BertClassifierManager()
        self.extractor = EmbeddingExtractor()

    def evaluate_test_dataset(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run genuine inference across held-out test set and calculate all validation metrics.
        """
        y_true = [LABEL2ID[l] for l in test_df["label"]]
        y_pred = []
        y_probs_fin = []
        y_probs_riv = []

        for sent in test_df["sentence"]:
            res = self.classifier.predict_sentence(sent)
            pred_id = LABEL2ID[res["predicted_label"]]
            y_pred.append(pred_id)
            y_probs_fin.append(res["prob_financial"])
            y_probs_riv.append(res["prob_river"])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Standard metrics
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        prec_fin = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        rec_fin = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_fin = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

        prec_riv = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_riv = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_riv = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # [[TP, FN], [FP, TN]] considering Financial as positive class (0)
        tp = int(cm[0, 0]) # True Financial
        fn = int(cm[0, 1]) # Financial misclassified as River
        fp = int(cm[1, 0]) # River misclassified as Financial
        tn = int(cm[1, 1]) # True River

        clf_report_dict = classification_report(
            y_true, y_pred,
            target_names=["financial", "river"],
            output_dict=True,
            zero_division=0
        )
        clf_report_text = classification_report(
            y_true, y_pred,
            target_names=["financial", "river"],
            zero_division=0
        )

        return {
            "accuracy": round(float(acc), 4),
            "macro_avg": {
                "precision": round(float(prec_macro), 4),
                "recall": round(float(rec_macro), 4),
                "f1_score": round(float(f1_macro), 4)
            },
            "weighted_avg": {
                "precision": round(float(prec_weighted), 4),
                "recall": round(float(rec_weighted), 4),
                "f1_score": round(float(f1_weighted), 4)
            },
            "class_metrics": {
                "financial": {
                    "precision": round(float(prec_fin), 4),
                    "recall": round(float(rec_fin), 4),
                    "f1_score": round(float(f1_fin), 4),
                    "support": int((y_true == 0).sum())
                },
                "river": {
                    "precision": round(float(prec_riv), 4),
                    "recall": round(float(rec_riv), 4),
                    "f1_score": round(float(f1_riv), 4),
                    "support": int((y_true == 1).sum())
                }
            },
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "total_test_samples": len(y_true)
            },
            "classification_report_dict": clf_report_dict,
            "classification_report_text": clf_report_text
        }

    def compute_pca_embeddings(self, test_df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """
        Extract genuine BERT contextual vectors for 'bank' across test samples
        and perform 2D and 3D PCA dimension reduction to visualize sense clustering.
        """
        half = sample_size // 2
        fin_samples = test_df[test_df["label"] == "financial"].sample(
            n=min(half, len(test_df[test_df["label"] == "financial"])), random_state=42
        )
        riv_samples = test_df[test_df["label"] == "river"].sample(
            n=min(half, len(test_df[test_df["label"] == "river"])), random_state=42
        )
        sampled_df = pd.concat([fin_samples, riv_samples]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        vectors = []
        labels = []
        sentences = []

        for _, row in sampled_df.iterrows():
            sent = str(row["sentence"])
            lbl = str(row["label"])
            vec, _, _ = self.extractor.extract_bert_bank_embedding(sent)
            vectors.append(vec)
            labels.append(lbl)
            sentences.append(sent)

        vectors = np.array(vectors) # [N, 768]
        pca_2d = PCA(n_components=2, random_state=42)
        coords_2d = pca_2d.fit_transform(vectors)

        pca_3d = PCA(n_components=3, random_state=42)
        coords_3d = pca_3d.fit_transform(vectors)

        explained_variance_2d = [round(float(v) * 100, 2) for v in pca_2d.explained_variance_ratio_]
        explained_variance_3d = [round(float(v) * 100, 2) for v in pca_3d.explained_variance_ratio_]

        return {
            "coords_2d": coords_2d.tolist(),
            "coords_3d": coords_3d.tolist(),
            "labels": labels,
            "sentences": sentences,
            "var_explained_2d": explained_variance_2d,
            "var_explained_3d": explained_variance_3d
        }

    @staticmethod
    def plot_confusion_matrix(cm_dict: Dict[str, Any]) -> go.Figure:
        """Create an annotated, publication-quality Plotly Confusion Matrix."""
        cm = cm_dict["matrix"]
        labels = ["Financial (Predicted)", "River (Predicted)"]
        y_labels = ["Financial (Actual)", "River (Actual)"]

        # Cell annotations with educational descriptions
        tp = cm_dict["true_positives"]
        fn = cm_dict["false_negatives"]
        fp = cm_dict["false_positives"]
        tn = cm_dict["true_negatives"]

        annotations_text = [
            [f"<b>{tp}</b><br>(True Positive)", f"<b>{fn}</b><br>(False Negative)"],
            [f"<b>{fp}</b><br>(False Positive)", f"<b>{tn}</b><br>(True Negative)"]
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=y_labels,
                colorscale="Blues",
                showscale=True,
                hoverongaps=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            )
        )

        for i in range(2):
            for j in range(2):
                fig.add_annotation(
                    x=labels[j],
                    y=y_labels[i],
                    text=annotations_text[i][j],
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > (tp + tn) / 4 else "#0F2027", size=14)
                )

        fig.update_layout(
            title={"text": "Test Set Confusion Matrix", "x": 0.5, "font": {"size": 16, "color": "#0F2027"}},
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            yaxis=dict(autorange="reversed"),
            width=540,
            height=460,
            margin=dict(l=60, r=40, t=60, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def plot_pca_scatter_2d(pca_data: Dict[str, Any]) -> go.Figure:
        """Create a 2D scatter plot of contextual BERT embeddings of 'bank'."""
        coords = np.array(pca_data["coords_2d"])
        labels = pca_data["labels"]
        sentences = pca_data["sentences"]
        var = pca_data["var_explained_2d"]

        df_plot = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Sense": [l.capitalize() for l in labels],
            "Sentence": [s[:80] + "..." if len(s) > 80 else s for s in sentences]
        })

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Sense",
            hover_data=["Sentence"],
            color_discrete_map={"Financial": "#0052cc", "River": "#00875a"},
            title=f"PCA of Contextual 'Bank' Embeddings (PC1: {var[0]}%, PC2: {var[1]}%)"
        )
        fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")))
        fig.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
        )
        return fig

    @staticmethod
    def plot_pca_scatter_3d(pca_data: Dict[str, Any]) -> go.Figure:
        """Create a 3D interactive scatter plot of contextual BERT embeddings of 'bank'."""
        coords = np.array(pca_data["coords_3d"])
        labels = pca_data["labels"]
        sentences = pca_data["sentences"]
        var = pca_data.get("var_explained_3d", [55.0, 4.1, 2.5])

        df_plot = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "PC3": coords[:, 2],
            "Sense": [l.capitalize() for l in labels],
            "Sentence": [s[:80] + "..." if len(s) > 80 else s for s in sentences]
        })

        fig = px.scatter_3d(
            df_plot,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Sense",
            hover_data=["Sentence"],
            color_discrete_map={"Financial": "#0052cc", "River": "#00875a"},
            title=f"3D PCA of Contextual 'Bank' Embeddings (PC1: {var[0]}%, PC2: {var[1]}%, PC3: {var[2]}%)"
        )
        fig.update_traces(marker=dict(size=5, opacity=0.85))
        fig.update_layout(
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig
