import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ID, PREDICTION, TARGET, get_splits, load_data, read_csv
from src.ensemble import blend
from src.features import add_encodings
from src.validate_submission import validate, write_submission


class PipelineTests(unittest.TestCase):
    def test_submission_is_reordered_by_identifier(self):
        sample = pd.DataFrame({ID: pd.array(["004", "001"], dtype="string"), TARGET: [0.5, 0.5]})
        predicted = pd.DataFrame({ID: sample[ID].iloc[::-1].array, PREDICTION: [0.9, 0.1]})
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "submission.csv"
            write_submission(predicted, sample, output)
            result = read_csv(output)
            self.assertEqual(result[ID].tolist(), ["004", "001"])
            self.assertEqual(result[TARGET].tolist(), [0.1, 0.9])
        with self.assertRaisesRegex(ValueError, "same order"):
            validate(result.iloc[::-1], sample)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(result.assign(id="004").astype({ID: "string"}), sample)
        with self.assertRaisesRegex(ValueError, "finite"):
            validate(result.assign(is_fraud=[0.5, np.nan]), sample)

    def test_holdout_is_excluded_from_development_folds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            n = 70
            train = pd.DataFrame({ID: range(n), "amount_usd": range(n), TARGET: np.tile([0, 1], n // 2)})
            test = pd.DataFrame({ID: [100, 101], "amount_usd": [102.0, 118.0]})
            for name, frame in {
                "train.csv": train, "test.csv": test, "sample_submission.csv": test[[ID]].assign(is_fraud=0.5),
            }.items():
                frame.to_csv(root / name, index=False)
            data = load_data(root)
            folds, path = get_splits(data, root / "artifacts", 5, 42)
            self.assertTrue(folds["full"].eq(0).equals(folds["dev"].eq(-1)))
            self.assertEqual(folds["dev"].eq(-1).sum(), 14)
            self.assertTrue(path.exists())
            loaded, _ = get_splits(data, root / "artifacts", 5, 42)
            pd.testing.assert_frame_equal(loaded, folds)

    def test_target_encoding_excludes_own_label_and_handles_unseen_categories(self):
        y = np.tile([0, 1], 12)
        train = pd.DataFrame({"cat": pd.array([f"unique-{i}" for i in range(len(y))], dtype="string")})
        valid = pd.DataFrame({"cat": pd.array(["new", "unique-1"], dtype="string")})
        add_encodings([train, valid], y, "target", 42)
        np.testing.assert_allclose(train["target__cat"], 0.5)
        self.assertAlmostEqual(valid["target__cat"].iloc[0], 0.5)
        self.assertGreater(valid["target__cat"].iloc[1], 0.5)

    def test_rank_blend_uses_same_fold_and_averages_ties(self):
        scores = np.array([[0.1, 0.2], [0.1, 0.3], [0.8, 0.9], [0.9, 0.8]])
        actual = blend(scores, "rank", 0.5, np.array([0, 0, 1, 1]))
        np.testing.assert_allclose(actual, [0.625, 0.875, 0.75, 0.75])


if __name__ == "__main__":
    unittest.main()
