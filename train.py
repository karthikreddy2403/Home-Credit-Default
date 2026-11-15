"""Fit and export the seven-table LightGBM model used by the interactive demo."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import build_features

ROOT = Path(__file__).resolve().parent
SEED = 42
HISTORY_PROFILES = {
    "short": "Short installment history",
    "regular": "Mostly on-time installments",
    "late": "Frequently late installments",
}


def clean_numbers(values):
    return [None if pd.isna(value) else float(value) for value in values]


def compact_tree(node):
    if "leaf_value" in node:
        return float(node["leaf_value"])
    if node["decision_type"] != "<=":
        raise ValueError("This exporter requires numeric splits")
    return [
        int(node["split_feature"]),
        float(node["threshold"]),
        bool(node["default_left"]),
        node["missing_type"],
        compact_tree(node["left_child"]),
        compact_tree(node["right_child"]),
    ]


def classifier(leaves, workers):
    return LGBMClassifier(
        objective="binary",
        n_estimators=130,
        learning_rate=0.06,
        num_leaves=leaves,
        min_child_samples=75,
        reg_lambda=2.0,
        colsample_bytree=0.85,
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        n_jobs=workers,
        verbosity=-1,
    )


def choose_leaves(x, y, workers):
    # Select hyperparameters entirely within the training partition.
    indices = np.arange(len(y))
    if len(indices) > 60000:
        indices, _ = train_test_split(
            indices, train_size=60000, stratify=y, random_state=SEED
        )
    subset, target = x.iloc[indices], y.iloc[indices]
    splits = list(
        StratifiedKFold(3, shuffle=True, random_state=SEED).split(subset, target)
    )
    scores = {}
    for leaves in (15, 31):
        fold_scores = []
        for train, valid in splits:
            model = classifier(leaves, workers).fit(
                subset.iloc[train], target.iloc[train]
            )
            predicted = model.predict_proba(subset.iloc[valid])[:, 1]
            fold_scores.append(float(roc_auc_score(target.iloc[valid], predicted)))
        scores[str(leaves)] = fold_scores
        print(
            f"{leaves} leaves: mean validation ROC AUC {np.mean(fold_scores):.4f}",
            flush=True,
        )
    best = max(scores, key=lambda leaves: np.mean(scores[leaves]))
    return int(best), scores


def training_profiles(x):
    count = x["INST_COUNT"].fillna(0)
    late_share = x["INST_WAS_LATE_mean"]
    masks = {
        "short": count <= 4,
        "regular": (count > 4) & (late_share <= 0.10),
        "late": (count > 4) & (late_share >= 0.35),
    }
    profiles = {}
    for key, mask in masks.items():
        group = x.loc[mask]
        # A median vector summarizes a group. Sparse history remains missing.
        median = group.median(skipna=True).where(group.notna().mean() >= 0.5)
        profiles[key] = {
            "label": HISTORY_PROFILES[key],
            "training_rows": len(group),
            "features": clean_numbers(median),
        }
    return profiles


def load_features(args):
    if args.features_cache and args.features_cache.is_file():
        table = pd.read_parquet(args.features_cache)
        return table.drop(columns=["TARGET"]), table["TARGET"]
    x, y = build_features(args.data)
    if args.features_cache:
        args.features_cache.parent.mkdir(parents=True, exist_ok=True)
        x.assign(TARGET=y).to_parquet(args.features_cache, compression="zstd")
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--features-cache",
        type=Path,
        help="Optional local cache tied to one input dataset.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    x, y = load_features(args)
    if not x.index.is_unique or not x.index.equals(y.index):
        raise ValueError("Expected unique applicant IDs aligned with their labels.")
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=SEED
    )
    train, target = x.iloc[train_idx], y.iloc[train_idx]
    test, test_target = x.iloc[test_idx], y.iloc[test_idx]
    best, validation = choose_leaves(train, target, args.workers)
    model = classifier(best, args.workers).fit(train, target)
    probabilities = model.predict_proba(test)[:, 1]
    metrics = {
        "dataset_rows": len(x),
        "feature_count": x.shape[1],
        "training_rows": len(train),
        "test_rows": len(test),
        "test_prevalence": float(test_target.mean()),
        "test_roc_auc": float(roc_auc_score(test_target, probabilities)),
        "cv_scores_by_num_leaves": validation,
        "selected_num_leaves": best,
        "seed": SEED,
    }
    dump = model.booster_.dump_model()
    if dump.get("average_output"):
        raise ValueError("This exporter expects a boosted sum of tree scores.")
    exported = {
        "feature_names": list(train.columns),
        "profiles": training_profiles(train),
        "trees": [compact_tree(t["tree_structure"]) for t in dump["tree_info"]],
        "metrics": metrics,
    }
    (ROOT / "docs/model-data.js").write_text(
        "globalThis.BenchmarkModel = "
        + json.dumps(exported, allow_nan=False, separators=(",", ":"))
        + ";\n"
    )
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Synthetic parity cases exercise missing-value branches without exporting any individual record.
    checks = []
    for profile in exported["profiles"].values():
        checks.append(profile["features"])
    checks.append([None] * x.shape[1])
    for shift in (-0.15, 0.15):
        v = list(exported["profiles"]["regular"]["features"])
        v[train.columns.get_loc("EXT_SOURCE_2")] = 0.55 + shift
        checks.append(v)
    matrix = pd.DataFrame(checks, columns=train.columns, dtype="float32")
    expected = model.predict_proba(matrix)[:, 1].tolist()
    cases = [{"values": row, "score": p} for row, p in zip(checks, expected)]
    (ROOT / "tests/reference.json").write_text(
        json.dumps(cases, allow_nan=False) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
