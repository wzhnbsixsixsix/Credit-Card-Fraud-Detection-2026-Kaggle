"""Persist complete runs and validate prediction alignment across experiments."""

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import platform
import re
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import ID, PREDICTION, TARGET, aligned, checksum, read_csv
from .validate_submission import probabilities, write_submission


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def run_directory(root, name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", name):
        raise ValueError("Run name must use letters, digits, dots, underscores or hyphens")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        raise ValueError(f"Run already exists: {target}. Choose a different name.")
    with TemporaryDirectory(prefix=".run-", dir=root) as work:
        # Publish the complete directory in one step. Failed runs leave no visible result.
        temporary = Path(work) / "result"
        temporary.mkdir()
        yield temporary
        temporary.rename(target)


def identity(data, folds_path, phase):
    return {
        "phase": phase,
        "data_fingerprint": data.fingerprint,
        "file_sha256": data.hashes,
        "split_sha256": checksum(folds_path),
    }


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "pytabkit", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        git, dirty = None, None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git, "git_dirty": dirty,
        "python": platform.python_version(), "machine": platform.machine(), "packages": packages,
    }


def metrics(oof):
    y = oof[TARGET].to_numpy()
    scores = probabilities(oof[PREDICTION])
    aucs = {
        str(int(fold)): float(roc_auc_score(group[TARGET], group[PREDICTION]))
        for fold, group in oof.groupby("fold")
    }
    top_count = max(1, int(np.ceil(len(y) * 0.01)))
    top_rows = np.argsort(scores, kind="stable")[-top_count:]
    return {
        "auc": float(roc_auc_score(y, scores)),
        "fold_auc": aucs, "fold_mean": float(np.mean(list(aucs.values()))),
        "fold_std": float(np.std(list(aucs.values()), ddof=0)),
        "average_precision": float(average_precision_score(y, scores)),
        "precision_at_top_1pct": float(np.mean(y[top_rows])),
    }


def save_run(directory, data, oof, test, holdout, metadata):
    oof.to_csv(directory / "oof.csv", index=False)
    manifest = {"oof.csv": checksum(directory / "oof.csv")}
    if test is not None:
        test.to_csv(directory / "test.csv", index=False)
        metadata["submission"] = write_submission(test, data.sample, directory / "submission.csv")
        manifest.update({name: checksum(directory / name) for name in ("test.csv", "submission.csv")})
    if holdout is not None:
        holdout.to_csv(directory / "holdout.csv", index=False)
        manifest["holdout.csv"] = checksum(directory / "holdout.csv")
    write_json(directory / "run.json", {
        **environment(), **metadata, "metrics": metrics(oof), "prediction_sha256": manifest,
    })


def load_run(path, expected, *, complete=True):
    meta = read_json(path / "run.json")
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{path} used different source data, folds or phase")
    if complete and not meta.get("complete"):
        raise ValueError(f"{path} is a partial run")
    return meta


def read_predictions(path, meta, ids, filename, columns):
    file = path / filename
    if checksum(file) != meta["prediction_sha256"].get(filename):
        raise ValueError(f"Hash mismatch for {file}")
    predictions = aligned(read_csv(file), ids, str(file))
    if list(predictions.columns) != [ID, *columns, PREDICTION]:
        raise ValueError(f"Unexpected columns in {file}")
    probabilities(predictions[PREDICTION])
    return predictions
