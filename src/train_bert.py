"""
BERT Fine-Tuning Module for Bank Sense Classification
Implements genuine fine-tuning for sequence classification on top of BERT representations.
Tracks training and validation loss/accuracy across epochs and persists weights locally.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
from .dataset_loader import load_splits

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "bert_model")
)
BERT_CACHE_NAME = "bert-base-uncased"
LABEL2ID = {"financial": 0, "river": 1}
ID2LABEL = {0: "financial", 1: "river"}

class BankDataset(Dataset):
    """PyTorch Dataset for sentence classification."""
    def __init__(self, sentences: List[str], labels: List[int], tokenizer, max_len: int = 64):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sent = str(self.sentences[item])
        label = self.labels[item]
        encoding = self.tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


class BertClassifierManager:
    """Manages training, saving, loading, and evaluation of the fine-tuned BERT classifier."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_CACHE_NAME, local_files_only=True)
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 3e-5,
        freeze_base: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fine-tune BERT for sequence classification.
        freeze_base: Freezes lower 10 layers for blazing fast CPU execution while fine-tuning
        the upper attention layer, pooling head, and classifier.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BERT_CACHE_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=True
        )
        self.model.to(self.device)

        # Freeze lower layers for CPU efficiency if requested
        if freeze_base:
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.model.bert.encoder.layer[:10]:
                for param in layer.parameters():
                    param.requires_grad = False

        train_dataset = BankDataset(
            train_df["sentence"].tolist(),
            [LABEL2ID[l] for l in train_df["label"]],
            self.tokenizer
        )
        val_dataset = BankDataset(
            val_df["sentence"].tolist(),
            [LABEL2ID[l] for l in val_df["label"]],
            self.tokenizer
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )

        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_val / total_val

            self.history["train_loss"].append(round(avg_train_loss, 4))
            self.history["val_loss"].append(round(avg_val_loss, 4))
            self.history["train_acc"].append(round(train_accuracy, 4))
            self.history["val_acc"].append(round(val_accuracy, 4))

        return self.history

    def save_model(self, save_dir: Optional[str] = None):
        """Save fine-tuned weights, tokenizer, and training history."""
        target_dir = save_dir or MODEL_DIR
        os.makedirs(target_dir, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(target_dir)
            self.tokenizer.save_pretrained(target_dir)
            with open(os.path.join(target_dir, "training_history.json"), "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

    def load_model(self, load_dir: Optional[str] = None):
        """Load fine-tuned BERT checkpoint."""
        target_dir = load_dir or MODEL_DIR
        if not os.path.exists(os.path.join(target_dir, "config.json")):
            raise FileNotFoundError(f"BERT checkpoint config not found in {target_dir}")
        self.model = AutoModelForSequenceClassification.from_pretrained(target_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        hist_path = os.path.join(target_dir, "training_history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)

    def predict_sentence(self, sentence: str) -> Dict[str, Any]:
        """Run actual inference on an input sentence."""
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "prob_financial": round(float(probs[0]), 4),
            "prob_river": round(float(probs[1]), 4),
            "logits": logits.squeeze(0).cpu().numpy().tolist()
        }
