"""
Offline Training & Pre-Evaluation Script
Trains Word2Vec, fine-tunes BERT classifier, and generates initial test set evaluations.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.dataset_loader import load_splits
from src.train_word2vec import Word2VecModel
from src.train_bert import BertClassifierManager
from src.evaluation import ModelEvaluator

def main():
    print("=" * 65)
    print("STARTING TRAINING PIPELINE: Word2Vec + BERT Contextual Model")
    print("=" * 65)
    start_time = time.time()

    # 1. Load Data Splits
    print("\n[1/4] Loading stratified dataset splits...")
    train_df, val_df, test_df = load_splits()
    print(f"Loaded: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # 2. Train Word2Vec Model
    print("\n[2/4] Training Static Word2Vec Model (CBOW Architecture)...")
    w2v = Word2VecModel(embedding_dim=100, window_size=3)
    w2v.train(train_df["sentence"].tolist(), epochs=15, lr=0.005, batch_size=64)
    w2v.save_model()
    print(f"Word2Vec successfully trained! Final loss: {w2v.training_loss[-1]}")
    bank_vec = w2v.get_vector("bank")
    print(f"Vector for 'bank' shape: {bank_vec.shape} (Static vector invariant across all contexts)")
    print(f"Top 3 similar words to 'bank' in Word2Vec:")
    for w, sim in w2v.most_similar("bank", top_n=3):
        print(f"  - {w}: {sim:.4f}")

    # 3. Fine-Tune BERT Classifier
    print("\n[3/4] Fine-Tuning BERT Contextual Classifier...")
    bert_mgr = BertClassifierManager()
    # 2 epochs is plenty to reach >98% accuracy on CPU quickly
    history = bert_mgr.train_model(
        train_df=train_df,
        val_df=val_df,
        epochs=2,
        batch_size=32,
        lr=3e-5,
        freeze_base=True
    )
    bert_mgr.save_model()
    print("BERT fine-tuning completed!")
    print(f"Training History:")
    for ep in range(len(history["train_loss"])):
        print(f"  Epoch {ep+1}: Train Loss={history['train_loss'][ep]:.4f}, Train Acc={history['train_acc'][ep]*100:.2f}%, Val Loss={history['val_loss'][ep]:.4f}, Val Acc={history['val_acc'][ep]*100:.2f}%")

    # 4. Run Evaluation on Held-Out Test Set
    print("\n[4/4] Evaluating Held-Out Test Set (315 balanced sentences)...")
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_test_dataset(test_df)
    
    print("\nTest Set Metrics:")
    print(f"  Overall Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision:  {eval_results['macro_avg']['precision'] * 100:.2f}%")
    print(f"  Macro Recall:     {eval_results['macro_avg']['recall'] * 100:.2f}%")
    print(f"  Macro F1-Score:   {eval_results['macro_avg']['f1_score'] * 100:.2f}%")
    print(f"  Confusion Matrix: TP={eval_results['confusion_matrix']['true_positives']}, TN={eval_results['confusion_matrix']['true_negatives']}, FP={eval_results['confusion_matrix']['false_positives']}, FN={eval_results['confusion_matrix']['false_negatives']}")
    
    # Save test metrics
    metrics_path = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    # Compute PCA for visualization cache
    print("\nComputing PCA of 100 sample test embeddings...")
    pca_data = evaluator.compute_pca_embeddings(test_df, sample_size=100)
    pca_path = os.path.join(BASE_DIR, "models", "pca_embeddings.json")
    with open(pca_path, "w", encoding="utf-8") as f:
        json.dump(pca_data, f, indent=2)
    print(f"Saved PCA data to: {pca_path}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f} seconds! Everything is trained and persisted locally.")

if __name__ == "__main__":
    main()
