"""Input validation, reproducible splits, and the competition data audit."""

from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ID = "id"
TARGET = "is_fraud"
PREDICTION = "prediction"
FILES = ("train.csv", "test.csv", "sample_submission.csv")
SIGNALS = (
    "merchant_risk_score", "velocity_score", "cvv_retry_count", "is_new_merchant",
    "amount_usd", "txn_count_last_24h", "auth_method", "channel",
)


def checksum(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_csv(path, **kwargs):
    return pd.read_csv(path, dtype={ID: "string"}, low_memory=False, **kwargs)


def check_ids(frame, context):
    if ID not in frame or frame.empty:
        raise ValueError(f"{context} must have rows and an id column")
    ids = pd.Index(frame[ID], name=ID)
    if ids.hasnans or not ids.is_unique or frame[ID].str.strip().eq("").any():
        raise ValueError(f"{context} contains missing, blank or duplicate IDs")
    return ids


def aligned(frame, ids, context):
    source = check_ids(frame, context)
    if len(source) != len(ids) or not ids.isin(source).all():
        raise ValueError(f"{context} IDs do not match the reference IDs")
    return frame.set_index(ID).loc[ids].reset_index()


def labels(frame, context):
    if TARGET not in frame:
        raise ValueError(f"{context} is missing {TARGET}")
    values = pd.to_numeric(frame[TARGET], errors="coerce")
    if values.isna().any() or set(values.unique()) != {0, 1}:
        raise ValueError(f"{context} must contain both binary labels, 0 and 1")
    return values.to_numpy(dtype=np.int8)


@dataclass
class CompetitionData:
    root: Path
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    features: list[str]
    hashes: dict[str, str]
    y: np.ndarray

    @property
    def fingerprint(self):
        return sha256("".join(self.hashes.values()).encode()).hexdigest()


def load_data(root):
    root = Path(root)
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(
            f"{root} is missing {', '.join(missing)}. Download the three files from "
            "Kaggle competition playground-series-s6e11."
        )
    train, test, sample = (read_csv(root / name) for name in FILES)
    train_ids = check_ids(train, "train.csv")
    test_ids = check_ids(test, "test.csv")
    sample_ids = check_ids(sample, "sample_submission.csv")
    if len(train_ids.intersection(test_ids)):
        raise ValueError("Train and test IDs overlap")
    if list(sample.columns) != [ID, TARGET]:
        raise ValueError("sample_submission.csv must contain id,is_fraud")
    if len(sample_ids) != len(test_ids) or not sample_ids.isin(test_ids).all():
        raise ValueError("Sample and test IDs do not match")
    features = [column for column in train if column not in (ID, TARGET)]
    if not features or set(test.columns) != {ID, *features}:
        raise ValueError("Train and test feature columns do not match, or test contains labels")
    return CompetitionData(
        root, train, test, sample, features,
        {name: checksum(root / name) for name in FILES}, labels(train, "train.csv"),
    )


def get_splits(data, output, n_folds, seed):
    directory = Path(output) / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v1-{data.fingerprint}-{n_folds}-{seed}.csv"
    ids = pd.Index(data.train[ID], name=ID)
    if not path.exists():
        if n_folds < 2 or np.bincount(data.y).min() < n_folds + 2:
            raise ValueError("Too few examples per class for the requested folds and holdout")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        full = np.full(len(ids), -1, dtype=np.int16)
        dev = full.copy()
        # Stable ID order keeps row assignments reproducible if a CSV is reordered.
        order = ids.argsort()
        for fold, (_, valid) in enumerate(splitter.split(order, data.y[order])):
            full[order[valid]] = fold
        keep = order[full[order] != 0]
        if np.bincount(data.y[keep]).min() < n_folds:
            raise ValueError("Too few examples per class remain after reserving the holdout")
        for fold, (_, valid) in enumerate(splitter.split(keep, data.y[keep])):
            dev[keep[valid]] = fold
        with NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
            temp = Path(stream.name)
            try:
                pd.DataFrame({ID: ids, "full": full, "dev": dev}).to_csv(stream, index=False)
                stream.flush()
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
    folds = aligned(read_csv(path), ids, "saved folds")
    if set(folds.columns) != {ID, "full", "dev"}:
        raise ValueError("Saved folds must contain id,full,dev")
    if (
        set(folds["full"]) != set(range(n_folds))
        or set(folds["dev"]) != {-1, *range(n_folds)}
        or not folds["full"].eq(0).equals(folds["dev"].eq(-1))
    ):
        raise ValueError("Saved folds have invalid assignments or an inconsistent holdout")
    for column in ("full", "dev"):
        for fold, group in folds.groupby(column).groups.items():
            if fold >= 0 and np.unique(data.y[group]).size != 2:
                raise ValueError(f"Saved {column} fold {fold} must contain both labels")
    return folds, path


def category_values(series):
    # Prefix real strings to distinguish actual values from the missing sentinel.
    return ("v:" + series.astype("string")).fillna("missing")


def model_frame(frame, features, schema):
    columns = {}
    for name in features:
        values = frame[name]
        kind = schema[name]
        if pd.api.types.is_bool_dtype(kind):
            mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
            columns[name] = values.astype("string").str.lower().map(mapping).astype("float32")
        elif pd.api.types.is_numeric_dtype(kind):
            columns[name] = pd.to_numeric(values, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).astype("float32")
        else:
            columns[name] = category_values(values)
    return pd.DataFrame(columns, index=frame.index)


def audit(data, folds):
    dev = folds["dev"].ge(0)
    tr = data.train.loc[dev, data.features]
    te = data.test[data.features]
    y_dev = data.y[dev.to_numpy()]
    rows = []
    for name in data.features:
        a, b = tr[name], te[name]
        item = {
            "feature": name, "train_type": str(a.dtype), "test_type": str(b.dtype),
            "train_missing": int(a.isna().sum()), "test_missing": int(b.isna().sum()),
            "train_unique": int(a.nunique()), "test_unique": int(b.nunique()),
        }
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            b = pd.to_numeric(b, errors="coerce")
            for split, series in (("train", a), ("test", b)):
                item[f"{split}_infinite"] = int(series.isin([np.inf, -np.inf]).sum())
                values = series.replace([np.inf, -np.inf], np.nan).dropna()
                item[f"{split}_p01_p50_p99"] = values.quantile([0.01, 0.5, 0.99]).tolist() if len(values) else None
        else:
            item["test_unseen_rows"] = int((b.notna() & ~b.isin(a.dropna())).sum())
        rows.append(item)

    hash_train = pd.util.hash_pandas_object(tr, index=False)
    hash_test = pd.util.hash_pandas_object(te, index=False)
    conflicts = pd.DataFrame({"hash": hash_train.to_numpy(), TARGET: y_dev}).groupby("hash")[TARGET].nunique()
    ordered = np.argsort(pd.to_numeric(data.train.loc[dev, ID], errors="coerce").to_numpy())
    if pd.to_numeric(data.train.loc[dev, ID], errors="coerce").isna().any():
        ordered = data.train.loc[dev, ID].argsort().to_numpy()
    segment_rates = [float(y_dev[group].mean()) for group in np.array_split(ordered, min(10, len(ordered)))]

    signals = {}
    for name in (name for name in SIGNALS if name in tr):
        a, b = tr[name], te[name]
        if pd.api.types.is_numeric_dtype(a) and a.nunique() > 12:
            numeric_a = a.replace([np.inf, -np.inf], np.nan)
            quantiles = np.unique(numeric_a.dropna().quantile(np.linspace(0, 1, 11)))
            if len(quantiles) >= 2:
                quantiles[[0, -1]] = [-np.inf, np.inf]
                a = pd.cut(numeric_a, quantiles, include_lowest=True)
                b = pd.cut(pd.to_numeric(b, errors="coerce"), quantiles, include_lowest=True)
        grouped = pd.DataFrame({"value": a.astype("string").fillna("missing").to_numpy(), TARGET: y_dev})
        stats = grouped.groupby("value")[TARGET].agg(train_count="size", fraud_rate="mean")
        stats["test_count"] = b.astype("string").fillna("missing").value_counts()
        stats["test_count"] = stats["test_count"].fillna(0).astype(int)
        signals[name] = stats.reset_index().to_dict("records")

    return {
        "rows": {"train": len(data.train), "test": len(data.test), "development": len(tr), "holdout": int((~dev).sum())},
        "development_fraud_rate": float(y_dev.mean()), "file_sha256": data.hashes,
        "features": rows, "development_id_segment_fraud_rates": segment_rates,
        "development_duplicate_rows": int(hash_train.duplicated().sum()),
        "development_conflicting_feature_keys": int(conflicts.gt(1).sum()),
        "test_rows_matching_development": int(hash_test.isin(hash_train).sum()),
        "signals": signals,
    }
