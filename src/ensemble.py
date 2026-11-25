"""Select an ensemble on development folds, review it, and apply frozen weights."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .data import ID, PREDICTION, TARGET, checksum, get_splits, load_data
from .results import (
    identity, load_run, metrics, read_json, read_predictions, run_directory, save_run, write_json,
)
from .train import data_options
from .validate_submission import prediction_frame

METHODS = ("probability", "logit", "rank")


def blend(scores, method, weight, folds=None):
    if not 0 < weight < 1 or method not in METHODS:
        raise ValueError("Blend requires a valid method and a weight between 0 and 1")
    if method == "logit":
        margin = logit(np.clip(scores, 1e-7, 1 - 1e-7))
        return expit(margin[:, 0] * weight + margin[:, 1] * (1 - weight))
    if method == "rank":
        if folds is None:
            scores = rankdata(scores, axis=0, method="average") / len(scores)
        else:
            # A validation row's rank is based on its own fold's predictions.
            ranked = np.empty(scores.shape)
            for fold in np.unique(folds):
                mask = folds == fold
                ranked[mask] = rankdata(scores[mask], axis=0, method="average") / mask.sum()
            scores = ranked
    return scores[:, 0] * weight + scores[:, 1] * (1 - weight)


def members(paths, data, folds, split_path, phase):
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("Select two distinct model runs")
    metadata = [load_run(path, identity(data, split_path, phase)) for path in paths]
    fold_column = "dev" if phase == "dev" else "full"
    mask = folds[fold_column].ge(0)
    ids = pd.Index(folds.loc[mask, ID], name=ID)
    oofs = [
        read_predictions(path, meta, ids, "oof.csv", ["fold", TARGET])
        for path, meta in zip(paths, metadata, strict=True)
    ]
    for path, oof in zip(paths, oofs, strict=True):
        if not (
            np.array_equal(oof[TARGET], data.y[mask.to_numpy()])
            and np.array_equal(oof["fold"], folds.loc[mask, fold_column])
        ):
            raise ValueError(f"{path} does not match the reference labels or folds")
    test_ids = pd.Index(data.test[ID], name=ID)
    tests = [
        read_predictions(path, meta, test_ids, "test.csv", [])
        for path, meta in zip(paths, metadata, strict=True)
    ]
    return metadata, oofs, tests


def score_matrix(frames):
    return np.column_stack([frame[PREDICTION].to_numpy() for frame in frames])


def selection(oof, scores):
    folds = oof["fold"].to_numpy()
    baseline_metrics = [
        metrics(oof.assign(prediction=scores[:, index])) for index in range(2)
    ]
    stronger = max(baseline_metrics, key=lambda row: row["fold_mean"])
    rows = []
    for method in METHODS:
        for weight in (0.50, 0.25, 0.75):
            result = metrics(oof.assign(prediction=blend(scores, method, weight, folds)))
            delta = result["fold_mean"] - stronger["fold_mean"]
            positive = sum(
                score > stronger["fold_auc"][fold] for fold, score in result["fold_auc"].items()
            )
            rows.append({
                "method": method, "weight": weight, "delta_fold_mean": delta,
                "folds_improved": positive, **result,
            })
    ranked = sorted(rows, key=lambda row: row["fold_mean"], reverse=True)
    return ranked[0], {"members": baseline_metrics, "blends": ranked}


def ensemble_run(args, data, folds, split_path):
    phase = "final" if args.command == "apply" else args.phase
    meta, oofs, tests = members(args.members, data, folds, split_path, phase)
    member_hashes = [checksum(path / "run.json") for path in args.members]
    if phase == "final":
        recipe = read_json(args.recipe)
        if (
            recipe.get("phase") != "dev" or recipe.get("data_fingerprint") != data.fingerprint
            or recipe.get("split_sha256") != checksum(split_path)
            or recipe.get("member_run_sha256") != [item.get("development_run_sha256") for item in meta]
        ):
            raise ValueError("The recipe does not match these final runs and their development parents")
        method, weight = recipe["method"], recipe["weight"]
    elif args.command == "search":
        winner, grid = selection(oofs[0], score_matrix(oofs))
        method, weight = winner["method"], winner["weight"]
    else:
        method, weight = args.method, args.weight
    recipe = {
        **identity(data, split_path, phase), "method": method, "weight": weight,
        "member_run_sha256": member_hashes,
        "members": [item["experiment"] for item in meta],
    }
    oof = oofs[0].assign(
        prediction=blend(score_matrix(oofs), method, weight, oofs[0]["fold"].to_numpy()),
    )
    test = prediction_frame(data.test[ID], blend(score_matrix(tests), method, weight))
    holdout = None
    if phase == "dev":
        ids = pd.Index(folds.loc[folds["dev"].eq(-1), ID], name=ID)
        components = [
            read_predictions(path, info, ids, "holdout.csv", [])
            for path, info in zip(args.members, meta, strict=True)
        ]
        holdout = prediction_frame(ids, blend(score_matrix(components), method, weight))
    with run_directory(args.out / phase, args.name) as directory:
        if args.command == "search":
            write_json(directory / "comparison.json", grid)
        write_json(directory / "recipe.json", recipe)
        save_run(directory, data, oof, test, holdout, {
            **recipe, "complete": True, "experiment": args.name,
        })
    print(f"{method} first-member weight {weight:.2f} · OOF AUC {metrics(oof)['auc']:.6f}")
    print(args.out / phase / args.name)


def review(args, data, folds, split_path):
    mask = folds["dev"].eq(-1).to_numpy()
    y = data.y[mask]
    ids = pd.Index(data.train.loc[mask, ID], name=ID)
    expected = identity(data, split_path, "dev")
    metadata = [load_run(path, expected) for path in args.members]
    predictions = [
        read_predictions(path, info, ids, "holdout.csv", [])[PREDICTION].to_numpy()
        for path, info in zip(args.members, metadata, strict=True)
    ]
    baseline, candidate = predictions
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    differences = np.empty(args.bootstrap)
    rng = np.random.default_rng(args.split_seed)
    for index in range(args.bootstrap):
        chosen = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in class_indices])
        differences[index] = roc_auc_score(y[chosen], candidate[chosen]) - roc_auc_score(y[chosen], baseline[chosen])
    scores = [float(roc_auc_score(y, values)) for values in predictions]
    interval = np.quantile(differences, [0.025, 0.975]).tolist()
    report = {
        **expected, "holdout_rows": len(y), "members": [item["experiment"] for item in metadata],
        "member_run_sha256": [checksum(path / "run.json") for path in args.members],
        "auc": scores, "candidate_minus_baseline": scores[1] - scores[0], "ci95": interval,
        "bootstrap_replicates": args.bootstrap, "selected": metadata[int(interval[0] > 0)]["experiment"],
    }
    with run_directory(args.out / "review", args.name) as directory:
        write_json(directory / "review.json", report)
    print(f"Holdout AUC {scores[0]:.6f} vs {scores[1]:.6f} · delta 95% CI [{interval[0]:+.6f}, {interval[1]:+.6f}]")
    print(f"Selected {report['selected']} · {args.out / 'review' / args.name / 'review.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("blend", "search", "review", "apply"):
        command = sub.add_parser(name)
        data_options(command)
        command.add_argument("members", nargs=2, type=Path, metavar="RUN")
        command.add_argument("--name", required=True)
        if name == "blend":
            command.add_argument("--phase", choices=("baseline", "dev"), default="dev")
            command.add_argument("--method", choices=METHODS, default="probability")
            command.add_argument("--weight", type=float, default=0.5)
        elif name == "search":
            command.set_defaults(phase="dev")
        elif name == "review":
            command.add_argument("--bootstrap", type=int, default=1000)
        else:
            command.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.folds < 2 or args.split_seed < 0:
            raise ValueError("--folds must be at least 2 and --split-seed must be nonnegative")
        data = load_data(args.data_dir)
        folds, split_path = get_splits(data, args.out, args.folds, args.split_seed)
        if args.command == "review":
            if args.bootstrap < 2:
                raise ValueError("--bootstrap must be at least 2")
            review(args, data, folds, split_path)
        else:
            ensemble_run(args, data, folds, split_path)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
