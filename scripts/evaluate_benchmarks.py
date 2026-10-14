"""
Standalone Fast CLI Benchmark & Inference Evaluator
Usage:
    python scripts/evaluate_benchmarks.py --mode benchmark
    python scripts/evaluate_benchmarks.py --mode predict --text "The system shall encrypt passwords with bcrypt."
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import generate_benchmark_corpus, set_seed
from src.features import MultiGranularityFeatureExtractor
from src.models import CalibratedCostSensitiveEnsemble
from src.evaluation import evaluate_proposed_model

def main():
    parser = argparse.ArgumentParser(description="MultiGran-NFR Benchmark & Inference Tool")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "predict"],
                        help="Execution mode: 'benchmark' for 10-fold CV or 'predict' for single sample inference.")
    parser.add_argument("--text", type=str, default="All REST endpoints shall authenticate tokens via OAuth2.",
                        help="Requirement text for prediction mode.")
    args = parser.parse_args()

    set_seed(42)
    df = generate_benchmark_corpus()
    categories = sorted(df['category'].unique())
    label_to_id = {c: i for i, c in enumerate(categories)}
    id_to_label = {i: c for i, c in enumerate(categories)}

    if args.mode == "benchmark":
        print("=" * 60)
        print("Executing 10-Fold Stratified Zero-Leakage Benchmark...")
        print("=" * 60)
        X = df['requirement'].values
        y_indices = df['category'].map(label_to_id).values
        results = evaluate_proposed_model(X, y_indices, n_splits=10)
        print(f"Mean Accuracy:  {results['accuracy_mean'] * 100:.2f}% (+/- {results['accuracy_std'] * 100:.2f}%)")
        print(f"Mean Precision: {results['precision_mean'] * 100:.2f}% (+/- {results['precision_std'] * 100:.2f}%)")
        print(f"Mean Recall:    {results['recall_mean'] * 100:.2f}% (+/- {results['recall_std'] * 100:.2f}%)")
        print(f"Mean F1-Score:  {results['f1_mean'] * 100:.2f}% (+/- {results['f1_std'] * 100:.2f}%)")
        print("=" * 60)
    elif args.mode == "predict":
        extractor = MultiGranularityFeatureExtractor(max_features=1200)
        X_vec = extractor.fit_transform(df['requirement'].values)
        model = CalibratedCostSensitiveEnsemble(random_state=42)
        model.fit(X_vec, df['category'].map(label_to_id).values)

        test_vec = extractor.transform([args.text])
        pred_id = model.predict(test_vec)[0]
        probs = model.predict_proba(test_vec)[0]

        print("\n" + "=" * 60)
        print(f"Requirement: \"{args.text}\"")
        print(f"Predicted Class: {id_to_label[pred_id]}")
        print("Class Probabilities:")
        for i, c in enumerate(categories):
            print(f"  - {c:20s}: {probs[i] * 100:5.2f}%")
        print("=" * 60)

if __name__ == "__main__":
    main()
