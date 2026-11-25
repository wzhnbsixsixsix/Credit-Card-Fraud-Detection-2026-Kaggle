"""Model-specific preprocessing and fitting on one fold."""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from threadpoolctl import threadpool_limits


def fit_logistic(frames, y, categories, params, threads):
    params = params.copy()
    bins = params.pop("n_bins")
    train = frames[0]
    discrete = [
        name for name in train if name not in categories and train[name].nunique() <= 12
    ]
    continuous = [name for name in train if name not in categories and name not in discrete]
    one_hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=12, dtype=np.float32)
    preprocessing = ColumnTransformer(
        [
            ("category", one_hot, categories + discrete),
            (
                "quantiles",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    KBinsDiscretizer(
                        n_bins=bins, strategy="quantile", quantile_method="linear", subsample=None,
                    ),
                ),
                continuous,
            ),
        ],
        sparse_threshold=1.0,
    )
    model = make_pipeline(preprocessing, LogisticRegression(**params))
    with threadpool_limits(limits=threads):
        # Repeated values can collapse adjacent quantile boundaries.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Bins whose width are too small.*")
            model.fit(train, y)
        predictions = [model.predict_proba(frame)[:, 1] for frame in frames[1:]]
    return predictions, None


def fit_fold(frames, y, y_valid, config, *, seed, threads, device, fixed_rounds=None):
    name = config["model"]
    params = config["params"].copy()
    train, valid, *others = frames
    categories = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
    if name == "logistic":
        return fit_logistic(frames, y, categories, params, threads)

    patience = params.pop("patience", None)
    rounds = None
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, early_stopping
        except (ImportError, OSError) as error:
            raise ValueError("LightGBM import failed. Install requirements and libomp on macOS.") from error
        # LightGBM can reject feature names containing certain JSON punctuation.
        names = [f"f{index}" for index in range(len(train.columns))]
        vocab = {column: pd.Index(train[column].unique()) for column in categories}
        for frame in frames:
            for column, values in vocab.items():
                frame[column] = pd.Categorical(frame[column], categories=values)
            frame.columns = names
        if fixed_rounds is not None:
            params["n_estimators"] = fixed_rounds
        model = LGBMClassifier(
            **params, objective="binary", metric="auc", random_state=seed, n_jobs=threads,
            device_type="gpu" if device == "cuda" else "cpu", verbosity=-1,
        )
        options = {"eval_set": [(valid, y_valid)], "callbacks": [early_stopping(patience, verbose=False)]}
        model.fit(train, y, **(options if fixed_rounds is None else {}))
        rounds = model.best_iteration_ or model.n_estimators_
    elif name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except (ImportError, OSError) as error:
            raise ValueError("CatBoost import failed. Install requirements for this platform.") from error
        for frame in frames:
            for column in categories:
                frame[column] = frame[column].astype(str)
        if fixed_rounds is not None:
            params["iterations"] = fixed_rounds
        model = CatBoostClassifier(
            **params, eval_metric="AUC", random_seed=seed, thread_count=threads,
            task_type="GPU" if device == "cuda" else "CPU", verbose=False, allow_writing_files=False,
        )
        options = {"eval_set": (valid, y_valid), "early_stopping_rounds": patience, "use_best_model": True}
        model.fit(train, y, cat_features=categories, **(options if fixed_rounds is None else {}))
        rounds = model.tree_count_
    elif name == "realmlp":
        try:
            from pytabkit import RealMLP_TD_Classifier
        except (ImportError, OSError) as error:
            raise ValueError("Install requirements-neural.txt to run RealMLP.") from error
        numeric = [column for column in train if column not in categories]
        fill_values = train[numeric].median().fillna(0)
        for frame in frames:
            frame[numeric] = frame[numeric].fillna(fill_values)
            for column in categories:
                frame[column] = frame[column].astype("category")
        epochs = params.pop("epochs")
        # With no validation split, the single refitting pass sees every outer training row.
        model = RealMLP_TD_Classifier(
            **params, n_epochs=epochs, n_cv=1, val_fraction=0.0 if fixed_rounds is not None else 0.2,
            stop_epoch=fixed_rounds, n_threads=threads, device=device, random_state=seed,
            val_metric_name="1-auc_ovr",
        )
        if fixed_rounds is None:
            model.fit(train, y, X_val=valid, y_val=y_valid)
            best_epoch = model.fit_params_["stop_epoch"]
            rounds = best_epoch["1-auc_ovr"] if isinstance(best_epoch, dict) else best_epoch
        else:
            model.fit(train, y)
            rounds = fixed_rounds
    else:
        raise ValueError(f"Unsupported model {name}")
    return [model.predict_proba(frame)[:, 1] for frame in (valid, *others)], int(rounds)
