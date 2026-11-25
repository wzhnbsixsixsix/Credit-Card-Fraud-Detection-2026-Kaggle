"""Write or validate a scored CSV against the sample's exact ID order."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID, PREDICTION, TARGET, aligned, check_ids, checksum, read_csv


def probabilities(values, *, nonconstant=False):
    try:
        scores = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Predictions must be numeric") from error
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Predictions must be a nonempty vector of finite values")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("Predictions must be between 0 and 1")
    if nonconstant and np.ptp(scores) == 0:
        raise ValueError("Every prediction is identical")
    return scores


def prediction_frame(ids, scores):
    return pd.DataFrame({ID: ids, PREDICTION: probabilities(scores)})


def validate(frame, sample):
    if list(frame.columns) != [ID, TARGET] or list(sample.columns) != [ID, TARGET]:
        raise ValueError("Submission and sample headers must be id,is_fraud")
    sample_ids = check_ids(sample, "sample")
    ids = check_ids(frame, "submission")
    if not ids.equals(sample_ids):
        raise ValueError("Submission IDs must match the sample, in the same order")
    scores = probabilities(frame[TARGET], nonconstant=True)
    return {"rows": len(scores), "quantiles": np.quantile(scores, [0, 0.01, 0.5, 0.99, 1]).tolist()}


def write_submission(predictions, sample, path):
    ordered = aligned(predictions, check_ids(sample, "sample"), "test predictions")
    submission = ordered[[ID, PREDICTION]].rename(columns={PREDICTION: TARGET})
    validate(submission, sample)
    submission.to_csv(path, index=False)
    return validate(read_csv(path), sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--sample", type=Path, default=Path("data/raw/sample_submission.csv"))
    args = parser.parse_args()
    try:
        report = validate(read_csv(args.submission), read_csv(args.sample))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"{report['rows']} rows · quantiles {report['quantiles']} · SHA-256 {checksum(args.submission)}")


if __name__ == "__main__":
    main()
