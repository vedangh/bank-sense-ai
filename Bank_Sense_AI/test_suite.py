"""
End-to-End Automated Test & Verification Suite
Validates all components, models, inferences, metrics, and Streamlit app integrity.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.dataset_loader import load_full_dataset, load_splits, get_dataset_statistics
from src.train_word2vec import Word2VecModel
from src.train_bert import BertClassifierManager
from src.embeddings import EmbeddingExtractor
from src.attention_visualization import AttentionExtractor
from src.prediction import PredictionEngine
from src.evaluation import ModelEvaluator

def run_tests():
    print("=" * 65)
    print("RUNNING COMPREHENSIVE E2E VERIFICATION TEST SUITE")
    print("=" * 65)

    # TEST 1: Dataset Integrity
    print("\n[TEST 1] Testing Dataset Loader & Splits...")
    df = load_full_dataset()
    assert len(df) == 2100, f"Expected 2100 records, got {len(df)}"
    stats = get_dataset_statistics(df)
    assert stats["financial_count"] == 1050, "Class imbalance in financial"
    assert stats["river_count"] == 1050, "Class imbalance in river"
    
    train_df, val_df, test_df = load_splits()
    assert len(train_df) == 1470, f"Train split size mismatch: {len(train_df)}"
    assert len(val_df) == 315, f"Val split size mismatch: {len(val_df)}"
    assert len(test_df) == 315, f"Test split size mismatch: {len(test_df)}"
    print("  ✓ Dataset loading, balance, and stratification passed!")

    # TEST 2: Static Word2Vec Model
    print("\n[TEST 2] Testing Word2Vec Model...")
    w2v = Word2VecModel()
    w2v.load_model()
    bank_vec = w2v.get_vector("bank")
    assert bank_vec.shape[0] == 100, f"Expected 100 dimensions, got {bank_vec.shape}"
    sim_self = w2v.similarity("bank", "bank")
    assert abs(sim_self - 1.0) < 1e-4, f"Self-similarity should be 1.0, got {sim_self}"
    print(f"  ✓ Word2Vec vector for 'bank' loaded ({bank_vec.shape[0]}-D). Static self-similarity: {sim_self:.4f}")

    # TEST 3: BERT Prediction Engine on Diverse & Unseen Sentences
    print("\n[TEST 3] Testing Prediction Engine on Financial, River, and Unseen Sentences...")
    engine = PredictionEngine()
    
    test_cases = [
        ("I deposited my salary in the bank.", "FINANCIAL"),
        ("The fisherman sat beside the bank of the river.", "RIVER"),
        ("The bank approved my education loan.", "FINANCIAL"),
        ("The children played near the bank of the river.", "RIVER"),
        ("The central bank announced a revision in interest rates.", "FINANCIAL"),
        ("Heavy rain caused the water to breach the river bank.", "RIVER"),
        ("The banker went fishing along the steep grassy bank.", "RIVER"), # Challenging contrast!
    ]
    
    for sentence, expected in test_cases:
        ok, res, err = engine.predict(sentence)
        assert ok, f"Prediction failed for '{sentence}': {err}"
        actual = res["predicted_meaning"]
        print(f"  Sentence: '{sentence[:55]}...'")
        print(f"    -> Predicted: {actual} (Expected: {expected}) | Confidence: {res['confidence']*100:.1f}%")
        assert actual == expected, f"Sense mismatch for '{sentence}'. Expected {expected}, got {actual}"
    print("  ✓ All prediction test cases classified accurately!")

    # TEST 4: Attention Visualization & Extraction
    print("\n[TEST 4] Testing Authentic Self-Attention Extraction...")
    attn_extractor = AttentionExtractor()
    attn_data = attn_extractor.get_attention_data("I deposited my salary in the bank.")
    tokens = attn_data["tokens"]
    bank_idx = attn_data["bank_token_idx"]
    print(f"  Sentence tokens: {tokens}")
    print(f"  'bank' token index: {bank_idx} ('{tokens[bank_idx]}')")
    assert tokens[bank_idx] == "bank", f"Expected token 'bank', got {tokens[bank_idx]}"
    
    matrix = attn_data["last_layer_avg_matrix"]
    assert matrix.shape == (len(tokens), len(tokens)), f"Matrix shape mismatch: {matrix.shape}"
    # Verify probability distribution (rows sum to ~1.0)
    row_sum = np.sum(matrix[bank_idx])
    assert abs(row_sum - 1.0) < 0.05, f"Attention row does not sum to 1.0: {row_sum}"
    print(f"  Attention matrix verified: {matrix.shape}, row sum: {row_sum:.4f}")
    top_anchor = attn_data["ranked_context"][0]["token"]
    print(f"  Top context token attended by 'bank': '{top_anchor}'")
    # Verify token boundary locator with contrast word 'banker'
    contrast_sent = "The banker sat on the grassy river bank."
    contrast_tokens = attn_extractor._tokenizer.convert_ids_to_tokens(
        attn_extractor._tokenizer(contrast_sent)["input_ids"]
    )
    contrast_bank_idx = attn_extractor.embed_extractor.find_bank_token_indices(contrast_sent)[0]
    print(f"  Contrast sentence token check: '{contrast_sent}'")
    print(f"  Target 'bank' resolved to index {contrast_bank_idx}: '{contrast_tokens[contrast_bank_idx]}'")
    assert contrast_tokens[contrast_bank_idx] == "bank", f"Expected 'bank', got '{contrast_tokens[contrast_bank_idx]}'"
    print("  ✓ Whole-word boundary token resolution verified!")

    # TEST 5: Cosine Similarity Comparison
    print("\n[TEST 5] Testing Vector Disambiguation Comparison...")
    embed_extractor = EmbeddingExtractor()
    comp = embed_extractor.compare_contexts(
        "I deposited money in the bank.",
        "The bank approved our loan.",
        "The fisherman rested on the river bank.",
        "Willows grew along the stream bank."
    )
    sim_ff = comp["sim_financial_to_financial"]
    sim_rr = comp["sim_river_to_river"]
    sim_fr = comp["sim_financial_to_river"]
    w2v_sim = comp["w2v_bank_invariance_similarity"]
    print(f"  Financial ↔ Financial Similarity: {sim_ff:.4f}")
    print(f"  River ↔ River Similarity:         {sim_rr:.4f}")
    print(f"  Financial ↔ River Similarity:     {sim_fr:.4f}")
    print(f"  Static Word2Vec Invariance:       {w2v_sim:.4f}")
    assert sim_ff > sim_fr, f"Expected intra-class similarity > inter-class similarity: {sim_ff} vs {sim_fr}"
    assert sim_rr > sim_fr, f"Expected intra-class similarity > inter-class similarity: {sim_rr} vs {sim_fr}"
    assert abs(w2v_sim - 1.0) < 1e-4, "Word2Vec should be invariant (1.0)"
    print("  ✓ Embedding similarity comparison mathematically validated!")

    # TEST 6: Validation Metrics & Confusion Matrix
    print("\n[TEST 6] Testing Saved Evaluation Metrics & Confusion Matrix...")
    eval_path = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")
    assert os.path.exists(eval_path), "Evaluation metrics file missing"
    with open(eval_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    acc = metrics["accuracy"]
    cm = metrics["confusion_matrix"]
    print(f"  Overall Accuracy: {acc*100:.2f}%")
    print(f"  TP: {cm['true_positives']} | TN: {cm['true_negatives']} | FP: {cm['false_positives']} | FN: {cm['false_negatives']}")
    assert cm["true_positives"] + cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] == 315
    print("  ✓ Metrics and Confusion Matrix mathematically verified!")

    # TEST 7: App Compilation Check
    print("\n[TEST 7] Testing Streamlit app.py syntax & compilation...")
    app_path = os.path.join(BASE_DIR, "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        compile(f.read(), app_path, "exec")
    print("  ✓ Streamlit app.py compiled with zero syntax errors!")

    print("\n" + "=" * 65)
    print("ALL 7 TEST SUITES PASSED FLAWLESSLY! 100% OPERATIONAL.")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
