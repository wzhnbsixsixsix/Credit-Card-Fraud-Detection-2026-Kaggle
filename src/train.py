"""Audit data and run cross-validated fraud models."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .data import ID, PREDICTION, TARGET, audit, checksum, get_splits, load_data
from .features import external_frame, fold_features, prepare_frame
from .models import fit_fold
from .results import identity, load_run, metrics, read_json, run_directory, save_run, write_json
from .validate_submission import prediction_frame, probabilities

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments.json"


def experiment(path, name):
    spec = read_json(path)
    if name not in spec["experiments"]:
        raise ValueError(f"Unknown experiment {name}. Available: {', '.join(spec['experiments'])}")
    chosen = spec["experiments"][name]
    return {**chosen, "params": {**spec[chosen["model"]], **chosen.get("params", {})}}


def fit_experiment(args, data, folds, split_path, directory, config):
    start = perf_counter()
    settings = identity(data, split_path, args.phase)
    if bool(config.get("external")) != bool(args.external_data):
        raise ValueError("Use --external-data only with a configured external-data experiment")
    external_hash = checksum(args.external_data) if args.external_data else None
    schema = data.train.dtypes
    train = prepare_frame(data.train, data.root / "train.csv", data.features, schema, config)
    test = prepare_frame(data.test, data.root / "test.csv", data.features, schema, config)
    external = external_frame(args.external_data, data, config, train, test) if args.external_data else None
    is_dev = args.phase == "dev"
    fold_values = folds["dev" if is_dev else "full"].to_numpy()
    holdout_indices = np.flatnonzero(fold_values == -1)
    use_folds = range(args.folds) if args.fold is None else [args.fold]
    complete = args.fold is None
    fixed_rounds = None
    if args.from_run:
        if args.phase != "final":
            raise ValueError("--from-run applies to final training")
        dev = load_run(args.from_run, {**settings, "phase": "dev"})
        if (
            dev.get("experiment") != args.experiment or dev.get("configuration") != config
            or dev.get("external_data_sha256") != external_hash
            or dev.get("model_seed") != args.model_seed or dev.get("device") != args.device
        ):
            raise ValueError("Final configuration, external data, seed or device differs from the development run")
        recorded_rounds = list(dev["rounds"].values())
        if recorded_rounds and recorded_rounds[0] is not None:
            fixed_rounds = int(np.median(recorded_rounds))
    elif args.phase == "final":
        raise ValueError("Final training requires --from-run pointing to a completed development run")

    oof = np.full(len(data.train), np.nan)
    test_sum = np.zeros(len(test))
    holdout_sum = np.zeros(len(holdout_indices))
    rounds_by_fold = {}
    for fold in use_folds:
        train_indices = np.flatnonzero((fold_values >= 0) & (fold_values != fold))
        valid_indices = np.flatnonzero(fold_values == fold)
        prediction_sets = [test]
        if is_dev:
            prediction_sets.append(train.iloc[holdout_indices])
        frames, y = fold_features(
            train.iloc[train_indices], train.iloc[valid_indices], prediction_sets,
            data.y[train_indices], config, external, args.model_seed + fold,
        )
        predictions, rounds = fit_fold(
            frames, y, data.y[valid_indices], config, seed=args.model_seed + fold,
            threads=args.threads, device=args.device, fixed_rounds=fixed_rounds,
        )
        oof[valid_indices] = probabilities(predictions[0])
        test_sum += probabilities(predictions[1]) / len(use_folds)
        if is_dev:
            holdout_sum += probabilities(predictions[2]) / len(use_folds)
        rounds_by_fold[str(fold)] = rounds
        mask = fold_values == fold
        fold_frame = pd.DataFrame({"fold": fold, TARGET: data.y[mask], PREDICTION: oof[mask]})
        print(f"Fold {fold + 1}/{args.folds}  AUC {metrics(fold_frame)['auc']:.6f}  rounds {rounds}", flush=True)
        del frames, predictions

    mask = np.isin(fold_values, use_folds)
    oof_frame = pd.DataFrame({
        ID: data.train.loc[mask, ID], "fold": fold_values[mask],
        TARGET: data.y[mask], PREDICTION: probabilities(oof[mask]),
    })
    test_frame = prediction_frame(data.test[ID], np.clip(test_sum, 0, 1)) if complete else None
    holdout_frame = None
    if is_dev and complete:
        holdout_frame = prediction_frame(data.train.iloc[holdout_indices][ID].reset_index(drop=True), np.clip(holdout_sum, 0, 1))
    save_run(directory, data, oof_frame, test_frame, holdout_frame, {
        **settings, "complete": complete, "experiment": args.experiment,
        "configuration": config, "split_seed": args.split_seed, "model_seed": args.model_seed,
        "threads": args.threads, "device": args.device, "rounds": rounds_by_fold,
        "external_data_sha256": external_hash, "external_rows_used": len(external[0]) if external else None,
        "fixed_rounds": fixed_rounds,
        "development_run_sha256": checksum(args.from_run / "run.json") if args.from_run else None,
        "elapsed_seconds": round(perf_counter() - start, 2),
    })
    print(f"OOF AUC {metrics(oof_frame)['auc']:.6f}", flush=True)


def data_options(parser):
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="Validate data, save folds and report feature summaries")
    data_options(audit_parser)
    train_parser = sub.add_parser("fit", help="Train a configured experiment and save its predictions")
    data_options(train_parser)
    train_parser.add_argument("experiment")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--phase", choices=("dev", "baseline", "final"), default="dev")
    train_parser.add_argument("--fold", type=int, help="Fit just this zero-based fold for a smoke check")
    train_parser.add_argument("--name", help="Optional output name. Default: experiment name")
    train_parser.add_argument("--model-seed", type=int, default=42)
    train_parser.add_argument("--threads", type=int, default=4)
    train_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train_parser.add_argument("--external-data", type=Path)
    train_parser.add_argument("--from-run", type=Path)
    args = parser.parse_args()

    try:
        if args.folds < 2 or args.split_seed < 0:
            raise ValueError("--folds must be at least 2 and --split-seed must be nonnegative")
        data = load_data(args.data_dir)
        folds, path = get_splits(data, args.out, args.folds, args.split_seed)
        if args.command == "audit":
            destination = args.out / "audit"
            name = f"{data.fingerprint[:12]}-{args.folds}-{args.split_seed}"
            if (destination / name / "audit.json").is_file():
                print(destination / name / "audit.json")
                return
            with run_directory(destination, name) as directory:
                write_json(directory / "audit.json", audit(data, folds))
            print(destination / name / "audit.json")
            print(path)
            return
        if args.threads < 1 or args.model_seed < 0 or (args.fold is not None and not 0 <= args.fold < args.folds):
            raise ValueError("Invalid thread count, model seed or fold number")
        config = experiment(args.config, args.experiment)
        if args.device == "cuda" and config["model"] == "lightgbm":
            raise ValueError("This pipeline uses the CPU build of LightGBM; omit --device cuda")
        name = args.name or (args.experiment if args.fold is None else f"{args.experiment}-fold{args.fold}")
        with run_directory(args.out / args.phase, name) as directory:
            fit_experiment(args, data, folds, path, directory, config)
        print(args.out / args.phase / name)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
