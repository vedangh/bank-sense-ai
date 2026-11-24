"""
Dataset Loader Module for Bank Sense Disambiguation
Handles loading, splitting, custom uploads, and statistics.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from .data_preprocessing import clean_sentence, validate_bank_sentence

DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)
DEFAULT_CSV_PATH = os.path.join(DEFAULT_DATA_DIR, "bank_sense_dataset.csv")

def load_full_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the complete balanced dataset."""
    path = csv_path or DEFAULT_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    df = pd.read_csv(path)
    df["sentence"] = df["sentence"].astype(str).apply(clean_sentence)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df

def load_splits(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test splits. Re-creates them if missing."""
    dir_path = data_dir or DEFAULT_DATA_DIR
    train_path = os.path.join(dir_path, "train.csv")
    val_path = os.path.join(dir_path, "val.csv")
    test_path = os.path.join(dir_path, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        df = load_full_dataset(os.path.join(dir_path, "bank_sense_dataset.csv"))
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
    for split_df in [train_df, val_df, test_df]:
        split_df["sentence"] = split_df["sentence"].astype(str).apply(clean_sentence)
        split_df["label"] = split_df["label"].astype(str).str.lower().str.strip()
        
    return train_df, val_df, test_df

def load_custom_dataset(file_or_path: Any) -> Tuple[pd.DataFrame, str]:
    """Load and validate an uploaded custom CSV file."""
    try:
        df = pd.read_csv(file_or_path)
    except Exception as e:
        return pd.DataFrame(), f"Failed to read CSV: {str(e)}"
        
    req_cols = {"sentence", "label"}
    if not req_cols.issubset(set(df.columns)):
        return pd.DataFrame(), f"CSV must contain 'sentence' and 'label' columns. Found: {list(df.columns)}"
        
    df = df.dropna(subset=["sentence", "label"]).copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_sentence)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    
    # Filter valid labels
    valid_labels = {"financial", "river"}
    df = df[df["label"].isin(valid_labels)]
    if df.empty:
        return pd.DataFrame(), "No valid rows found. Labels must be either 'financial' or 'river'."
        
    # Check bank presence
    df = df[df["sentence"].apply(lambda s: validate_bank_sentence(s)[0])]
    if df.empty:
        return pd.DataFrame(), "No valid sentences found containing the word 'bank'."
        
    df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    return df, f"Successfully loaded {len(df)} validated records."

def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for a dataset dataframe."""
    total = len(df)
    counts = df["label"].value_counts().to_dict()
    fin_count = counts.get("financial", 0)
    riv_count = counts.get("river", 0)
    
    lengths = df["sentence"].apply(lambda s: len(s.split()))
    avg_len = round(lengths.mean(), 2) if total > 0 else 0
    max_len = int(lengths.max()) if total > 0 else 0
    min_len = int(lengths.min()) if total > 0 else 0
    
    return {
        "total_records": total,
        "financial_count": fin_count,
        "river_count": riv_count,
        "balance_ratio": round(fin_count / max(1, riv_count), 2),
        "avg_sentence_length": avg_len,
        "min_sentence_length": min_len,
        "max_sentence_length": max_len,
        "financial_percent": round(fin_count / max(1, total) * 100, 1),
        "river_percent": round(riv_count / max(1, total) * 100, 1)
    }
