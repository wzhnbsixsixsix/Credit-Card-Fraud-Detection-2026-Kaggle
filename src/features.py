"""Fold-local feature experiments. Labels enter only through training rows."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import TARGET, labels, model_frame

DIGIT_SOURCES = (
    "amount_usd", "merchant_risk_score", "velocity_score", "distance_from_home_km",
    "hours_since_last_txn", "account_balance_usd",
)
INTERACTIONS = (
    ("merchant_risk_score", "is_new_merchant"),
    ("velocity_score", "txn_count_last_24h"),
    ("cvv_retry_count", "auth_method"),
)
ENCODING_PAIRS = (("auth_method", "channel"), ("merchant_category", "card_type"))


def prepare_frame(frame, path, columns, schema, config):
    result = model_frame(frame, columns, schema)
    if not config.get("digits"):
        return result

    sources = [name for name in DIGIT_SOURCES if name in columns]
    if not sources:
        raise ValueError("This experiment needs at least one supported decimal feature")
    csv_text = pd.read_csv(path, usecols=sources, dtype="string")
    for name in sources:
        # Extract the CSV spelling directly, before a float parser can change it.
        parts = csv_text[name].str.strip().str.extract(
            r"^[+-]?(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$"
        )
        whole = parts[0].fillna("")
        decimal = parts[1].fillna("")
        significant = (whole + decimal).str.lstrip("0")
        prefix = f"digit__{name}__"
        position = whole.str.len() - ((whole + decimal).str.len() - significant.str.len())
        exponent = pd.to_numeric(parts[2], errors="coerce").fillna(0)
        result[f"{prefix}scale"] = category_token((position + exponent).where(significant.ne("")))
        result[f"{prefix}precision"] = category_token(decimal.str.len().where(parts[0].notna()))
        for i in range(1, 6):
            result[f"{prefix}significant_{i}"] = category_token(significant.str.get(i - 1))
        result[f"{prefix}last_decimal"] = category_token(decimal.str.get(-1))
        numeric = pd.to_numeric(csv_text[name], errors="coerce").astype("float64")
        # Use a common cents grid to group infrequent values by magnitude and digit.
        cents = np.rint(numeric.abs().where(numeric.abs() < 1e15) * 100).astype("Int64")
        integer = (cents // 100).astype("string")
        result[f"{prefix}rounded_cents"] = category_token(cents % 100)
        result[f"{prefix}rounded_integer"] = category_token(integer)
        result[f"{prefix}length_leading"] = category_token(
            integer.str.len().astype("string") + "_" + integer.str[:2]
        )
    if config["digits"] == "only":
        return result.filter(regex=r"^digit__")
    return result


def category_token(values):
    return values.astype("string").fillna("missing")


def numeric_bins(series, quantiles):
    return np.unique(series.dropna().quantile(quantiles).dropna().to_numpy())[1:-1]


def discrete(series, edges):
    return pd.Series(
        np.where(series.isna(), -1, np.searchsorted(edges, series.to_numpy(), side="right")),
        index=series.index,
    )


def shuffle_digits(frames, seed):
    rng = np.random.default_rng(seed)
    for name in (name for name in DIGIT_SOURCES if name in frames[0]):
        edges = numeric_bins(frames[0][name], np.linspace(0, 1, 11))
        columns = list(frames[0].filter(like=f"digit__{name}__"))
        for frame in frames:
            order = np.arange(len(frame))
            bins = discrete(frame[name], edges)
            for indices in bins.groupby(bins).indices.values():
                order[indices] = rng.permutation(indices)
            for column in columns:
                frame[column] = frame[column].array.take(order)


def add_interactions(frames):
    fit = frames[0]
    for left, right in INTERACTIONS:
        if left not in fit or right not in fit:
            continue
        names = (left, right)
        edges = {
            name: numeric_bins(fit[name], np.linspace(0, 1, 9))
            for name in names if pd.api.types.is_numeric_dtype(fit[name]) and fit[name].nunique() > 12
        }
        for frame in frames:
            keys = [
                category_token(discrete(frame[name], edges[name]) if name in edges else frame[name])
                for name in names
            ]
            # A stable pair hash avoids collisions from string delimiters in category values.
            frame[f"pair__{left}__{right}"] = pd.util.hash_pandas_object(
                pd.concat(keys, axis=1), index=False
            ).astype("string")


def encoding_keys(frame, columns, pairs):
    result = {name: frame[name] for name in columns}
    for left, right in pairs:
        result[f"{left}__{right}"] = pd.util.hash_pandas_object(frame[[left, right]], index=False)
    return result


def risk_map(key, y, smoothing=25):
    prior = float(np.mean(y))
    stats = pd.Series(y, index=key.index).groupby(key, dropna=False).agg(["sum", "size"])
    return (stats["sum"] + smoothing * prior) / (stats["size"] + smoothing), prior


def add_encodings(frames, y, mode, seed):
    train = frames[0]
    columns = [name for name in train if not pd.api.types.is_numeric_dtype(train[name])]
    pairs = [pair for pair in ENCODING_PAIRS if all(name in columns for name in pair)]
    keys = [encoding_keys(frame, columns, pairs) for frame in frames]
    inner = []
    if mode == "target" and columns:
        n_splits = min(3, int(np.bincount(y).min()))
        if n_splits < 2:
            raise ValueError("Target encoding needs at least two examples per class in the training fold")
        inner = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y))

    for name, key in keys[0].items():
        output = f"{mode}__{name}"
        if mode == "frequency":
            counts = key.value_counts()
            for frame, column_keys in zip(frames, keys, strict=True):
                frame[output] = np.log1p(column_keys[name].map(counts).fillna(0)).astype("float32")
            continue
        cross_fit = np.empty(len(y), dtype=np.float32)
        for fit, valid in inner:
            mapping, prior = risk_map(key.iloc[fit], y[fit])
            cross_fit[valid] = key.iloc[valid].map(mapping).fillna(prior)
        train[output] = cross_fit
        mapping, prior = risk_map(key, y)
        for frame, column_keys in zip(frames[1:], keys[1:], strict=True):
            frame[output] = column_keys[name].map(mapping).fillna(prior).astype("float32")


def external_frame(path, data, config, train, test):
    frame = pd.read_csv(path, low_memory=False)
    missing = set(data.features).difference(frame)
    if missing:
        raise ValueError(f"External data is missing features: {', '.join(sorted(missing))}")
    reference = prepare_frame(frame, path, data.features, data.train.dtypes, config)
    keys = pd.util.hash_pandas_object(reference, index=False)
    # Exact feature matches must not supply reference labels to validation or test rows.
    keep = ~keys.isin(pd.util.hash_pandas_object(train, index=False))
    keep &= ~keys.isin(pd.util.hash_pandas_object(test, index=False))
    if not keep.any():
        raise ValueError("External data has no rows after excluding exact feature matches")
    reference = reference.loc[keep].reset_index(drop=True)
    y = labels(frame.loc[keep], path.name) if config["external"] != "count" else None
    return reference, y


def add_external_lookups(frames, external, mode):
    reference, y = external
    for name in reference:
        if pd.api.types.is_numeric_dtype(reference[name]):
            edges = numeric_bins(reference[name], np.linspace(0, 1, 33))
            source = discrete(reference[name], edges)
            keys = [discrete(frame[name], edges) for frame in frames]
        else:
            source = reference[name]
            keys = [frame[name] for frame in frames]
        counts = source.value_counts()
        if mode == "risk":
            risks, prior = risk_map(source, y)
        for frame, key in zip(frames, keys, strict=True):
            frame[f"external_count__{name}"] = np.log1p(key.map(counts).fillna(0)).astype("float32")
            if mode == "risk":
                frame[f"external_risk__{name}"] = key.map(risks).fillna(prior).astype("float32")


def fold_features(train, valid, others, y, config, external, seed):
    frames = [frame.copy().reset_index(drop=True) for frame in (train, valid, *others)]
    if config.get("digits") == "shuffle":
        shuffle_digits(frames, seed)
    if config.get("interactions"):
        add_interactions(frames)
    if mode := config.get("encoding"):
        add_encodings(frames, y, mode, seed)
    if external is not None:
        if config["external"] == "rows":
            frames[0] = pd.concat([frames[0], external[0]], ignore_index=True)
            y = np.concatenate((y, external[1]))
        else:
            add_external_lookups(frames, external, config["external"])
    return frames, y
