from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.training.config_io import load_experiment_env_config


class TrainingConfigIoTests(unittest.TestCase):
    def test_experiment_extends_deep_merges_env_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "env_config:\n"
                "  step_ratio: 6\n"
                "  guidance:\n"
                "    enabled: true\n"
                "    scale: 0.2\n",
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                "extends: base.yaml\n"
                "env_config:\n"
                "  guidance:\n"
                "    scale: 0.4\n",
                encoding="utf-8",
            )

            config = load_experiment_env_config("child.yaml", root)

            self.assertEqual(config["step_ratio"], 6)
            self.assertTrue(config["guidance"]["enabled"])
            self.assertEqual(config["guidance"]["scale"], 0.4)

    def test_experiment_extends_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
            (root / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cyclic experiment extends"):
                load_experiment_env_config("a.yaml", root)


if __name__ == "__main__":
    unittest.main()
