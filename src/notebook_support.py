"""Local inputs and prediction previews for the S6E11 notebooks."""

from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata

from .data import FILES, ID, TARGET, aligned, check_ids, checksum, read_csv
from .validate_submission import probabilities, validate

REFERENCE_HASHES = {
    "verified": "04c90efb29839b9b87b97563f8246764234aa9064372c2c9d4f51e4211167da8",
    "ag": "5a2c6e7bacb74249d4c1ffce8f17195e2ba7f767cc898eeb54776863b6969f6a",
    "ac": "f3acc736909177529f34d3d314d0077d1eb0f31d392cdf2a0132fb4c377bd484",
}


def competition_data_dir(project_dir):
    directory = Path(project_dir) / "data" / "raw"
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} 缺少比赛文件：{', '.join(missing)}")
    return directory


def reference_files(directory):
    by_hash = {digest: name for name, digest in REFERENCE_HASHES.items()}
    references = {}
    for file in sorted(directory.rglob("*.csv")):
        if file.is_file() and (name := by_hash.get(checksum(file))):
            references[name] = file
    return references


def zero_risk_zones(test):
    flags = [
        "is_new_merchant", "is_foreign_transaction", "used_vpn", "ip_country_mismatch",
        "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
    ]
    clean = test.cvv_retry_count.eq(0) & test.prior_disputes.eq(0) & ~test[flags].astype(bool).any(axis=1)
    velocity, risk = test.velocity_score, test.merchant_risk_score
    return (
        risk.lt(13.1)
        | (velocity.le(4.1) & risk.le(20.0))
        | (clean & velocity.le(4.1) & risk.le(33.8))
        | (clean & velocity.le(6.0) & risk.le(25.0))
    ).to_numpy()


def rank_fraction(values):
    return (rankdata(values) - 0.5) / len(values)


def continue_ranks(sample, zones, ag_path, ac_path):
    ids = check_ids(sample, "sample")
    margins = []
    for path in (ag_path, ac_path):
        predictions = aligned(read_csv(path), ids, str(path))
        margins.append(logit(rank_fraction(probabilities(predictions[TARGET]))))
    ag, ac = margins
    gap = ag - ac
    cap = np.quantile(np.abs(gap[~zones]), 0.99)
    scores = rank_fraction(ag + 0.78 * np.clip(gap, -cap, cap))
    scores[zones] -= 2.0
    return sample[[ID]].assign(**{TARGET: rank_fraction(scores)})


def sota_preview(project_dir, sample, test):
    project_dir = Path(project_dir)
    test = aligned(test, check_ids(sample, "sample"), "test.csv")
    zones = zero_risk_zones(test)
    paths = reference_files(project_dir / "data" / "reference" / "sota-v18")
    if "verified" in paths:
        mode = "verified"
        frame = read_csv(paths["verified"])
    elif {"ag", "ac"} <= paths.keys():
        mode = "continued"
        frame = continue_ranks(sample, zones, paths["ag"], paths["ac"])
    else:
        baseline = project_dir / "artifacts" / "baseline" / "e01-lgb2" / "submission.csv"
        if not baseline.is_file():
            return None, "missing", None, int(zones.sum())
        frame = read_csv(baseline)
        validate(frame, sample)
        return frame, "baseline", baseline, int(zones.sum())

    validate(frame, sample)
    output = project_dir / "artifacts" / "notebooks" / "sota-v18" / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            if mode == "verified":
                copyfile(paths["verified"], temp)
            else:
                frame.to_csv(temp, index=False)
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
    return frame, mode, output, int(zones.sum())
