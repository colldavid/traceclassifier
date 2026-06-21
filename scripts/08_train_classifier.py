"""Train a classifier to predict capitulation from thinking traces.

Loads data/main_pipeline_results.json (or a partial run), extracts
interpretable features from each thinking trace, trains a logistic
regression, and reports accuracy + feature importances.

Target: binary — resisted (1) vs capitulated/hedged (0).
        3-class mode available via --multiclass flag.

Args:
  --data PATH      Path to results JSON (default: data/main_pipeline_results.json)
  --multiclass     3-class mode: resisted / hedged / capitulated
  --condition N    Train only on condition N (1, 2, or 3). Default: all.
  --save PATH      Save trained model to this path (joblib). Default: none.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features import extract_with_meta, FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_INPUT = os.path.join(DATA_DIR, "main_pipeline_results.json")

ALL_FEATURE_NAMES = FEATURE_NAMES + ["condition", "n_documents", "correct_in_thinking", "wrong_in_thinking"]


def load_dataset(path: str, condition: int | None, multiclass: bool):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    # Only rows with a judge label
    rows = [r for r in rows if r.get("judge_label") in ("resisted", "capitulated", "hedged")]

    if condition is not None:
        rows = [r for r in rows if r.get("condition") == condition]

    X = np.array([list(extract_with_meta(r).values()) for r in rows])

    if multiclass:
        label_map = {"resisted": 2, "hedged": 1, "capitulated": 0}
        label_names = ["capitulated", "hedged", "resisted"]
    else:
        label_map = {"resisted": 1, "capitulated": 0, "hedged": 0}
        label_names = ["cap/hedged", "resisted"]

    y = np.array([label_map[r["judge_label"]] for r in rows])

    return X, y, label_names, rows


def print_feature_importances(model, feature_names, label_names, top_n=10):
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        # Tree-based: feature_importances_ (higher = more important, no direction)
        importances = clf.feature_importances_
        print(f"\n  Top {top_n} features by importance (tree-based, no direction):")
        ranked = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)
        for idx, imp in ranked[:top_n]:
            print(f"    {feature_names[idx]:<30} {imp:.4f}")
    elif hasattr(clf, "coef_"):
        coef = clf.coef_
        if coef.shape[0] == 1:
            importances = coef[0]
            print(f"\n  Top {top_n} features by absolute coefficient weight (resisted vs cap/hedged):")
            ranked = sorted(enumerate(importances), key=lambda x: abs(x[1]), reverse=True)
            for idx, weight in ranked[:top_n]:
                direction = "-> RESIST" if weight > 0 else "-> CAPITULATE"
                print(f"    {feature_names[idx]:<30} {weight:+.3f}  {direction}")
        else:
            for class_idx, class_name in enumerate(label_names):
                importances = coef[class_idx]
                print(f"\n  Top features for class '{class_name}':")
                ranked = sorted(enumerate(importances), key=lambda x: abs(x[1]), reverse=True)
                for idx, weight in ranked[:5]:
                    print(f"    {feature_names[idx]:<30} {weight:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_INPUT)
    parser.add_argument("--multiclass", action="store_true")
    parser.add_argument("--condition", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--model", default="logistic", choices=["logistic", "gbm", "rf"])
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    X, y, label_names, rows = load_dataset(args.data, args.condition, args.multiclass)

    print(f"Dataset: {len(rows)} rows")
    print(f"Label distribution: {dict(Counter(r['judge_label'] for r in rows))}")
    if args.condition:
        print(f"Condition filter: C{args.condition} only")
    print(f"Mode: {'3-class' if args.multiclass else 'binary (resisted vs cap+hedged)'}")
    print(f"Model: {args.model}")
    print(f"Features: {len(ALL_FEATURE_NAMES)}")
    print()

    if args.model == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif args.model == "gbm":
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model = Pipeline([("clf", clf)])
    elif args.model == "rf":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        model = Pipeline([("clf", clf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated accuracy
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    if not args.multiclass:
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        print(f"5-fold CV F1:       {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Full confusion matrix via cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=cv)
    print()
    print("Classification report (cross-validated predictions):")
    print(classification_report(y, y_pred, target_names=label_names, digits=3))

    print("Confusion matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  (rows=actual, cols=predicted | classes: {label_names})")
    for i, row_vals in enumerate(cm):
        print(f"  {label_names[i]:<14}: {row_vals}")
    print()

    # Fit on full data for feature importances
    model.fit(X, y)
    print_feature_importances(model, ALL_FEATURE_NAMES, label_names)

    # Per-condition breakdown (only in all-conditions mode)
    if args.condition is None:
        print()
        print("Per-condition accuracy (CV):")
        for cond in (1, 2, 3):
            mask = np.array([r["condition"] == cond for r in rows])
            if mask.sum() < 10:
                continue
            Xc, yc = X[mask], y[mask]
            cv_c = StratifiedKFold(n_splits=min(5, int(mask.sum() // 2)), shuffle=True, random_state=42)
            sc = cross_val_score(model, Xc, yc, cv=cv_c, scoring="accuracy")
            print(f"  C{cond} (n={mask.sum():4d}): {sc.mean():.3f} ± {sc.std():.3f}")

    if args.save:
        import joblib
        joblib.dump(model, args.save)
        print(f"\nModel saved to {args.save}")


if __name__ == "__main__":
    main()
