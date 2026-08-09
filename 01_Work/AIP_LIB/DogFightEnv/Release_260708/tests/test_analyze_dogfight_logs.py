import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.log_analysis.analyze_dogfight_logs import (
    analyze_summary,
    build_summary,
    classify_result,
    discover_summaries,
)


class ClassificationTests(unittest.TestCase):
    def test_separates_target_destroyed_from_target_crash(self):
        self.assertEqual(
            classify_result(
                {"outcome": "win", "end_condition": "target destroyed"}
            ),
            "self_shot_kill",
        )
        self.assertEqual(
            classify_result(
                {"outcome": "win", "end_condition": "target altitude below min"}
            ),
            "target_crash",
        )

    def test_classifies_ownship_crash_and_timeout(self):
        self.assertEqual(
            classify_result(
                {"outcome": "crash", "end_condition": "ownship altitude below min"}
            ),
            "ownship_crash",
        )
        self.assertEqual(
            classify_result({"outcome": "timeout", "end_condition": "max time out"}),
            "timeout",
        )


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.episode_dir = self.root / "stage_02_iter_000029" / "episode_01"
        self.episode_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_csv(self, path: Path, rows: list[list[object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Time",
                    "Longitude",
                    "Latitude",
                    "Altitude",
                    "Roll (deg)",
                    "Pitch (deg)",
                    "Yaw (deg)",
                    "Health",
                ]
            )
            writer.writerows(rows)

    def test_analyzes_health_geometry_and_wez(self):
        prefix = "2026_8_5_15_3_9"
        summary_path = self.episode_dir / f"{prefix}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "outcome": "win",
                    "end_condition": "target destroyed",
                    "ownship_health": 0.9,
                    "target_health": 0.0,
                }
            ),
            encoding="utf-8",
        )
        ownship_path = self.episode_dir / f"{prefix}_ownship_(F-16)[Blue].csv"
        target_path = self.episode_dir / f"{prefix}_target_(F-16)[Red].csv"
        self._write_csv(
            ownship_path,
            [
                [0.0, 128.0, 37.0, 7000.0, 5.0, 0.0, 0.0, 1.0],
                [0.1, 128.0, 37.0, 7000.0, 10.0, 0.0, 0.0, 0.9],
            ],
        )
        self._write_csv(
            target_path,
            [
                [0.0, 128.0, 37.006, 7000.0, 0.0, 0.0, 0.0, 0.5],
                [0.1, 128.0, 37.006, 7000.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        result = analyze_summary("PPO_V14", summary_path, 152.4, 914.4, 2.0)

        self.assertEqual(result.stage, 2)
        self.assertEqual(result.iteration, 29)
        self.assertEqual(result.episode, 1)
        self.assertEqual(result.result_class, "self_shot_kill")
        self.assertAlmostEqual(result.damage_dealt_percent, 100.0)
        self.assertAlmostEqual(result.damage_received_percent, 10.0)
        self.assertEqual(result.calculated_wez_steps, 2)
        self.assertAlmostEqual(result.calculated_wez_max_continuous_s, 0.2)
        self.assertAlmostEqual(result.max_abs_roll_deg, 10.0)

    def test_discovers_and_summarizes_episodes(self):
        summaries = []
        for index, (outcome, end_condition) in enumerate(
            [
                ("win", "target destroyed"),
                ("win", "target altitude below min"),
                ("timeout", "max time out"),
            ]
        ):
            path = self.episode_dir / f"case_{index}_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "outcome": outcome,
                        "end_condition": end_condition,
                        "ownship_health": 1.0,
                        "target_health": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            summaries.append(path)

        self.assertEqual(discover_summaries(self.root), sorted(summaries))
        results = [
            analyze_summary("MODEL", path, 152.4, 914.4, 2.0)
            for path in summaries
        ]
        metrics = build_summary(results)["labels"]["MODEL"]
        self.assertEqual(metrics["episodes"], 3)
        self.assertEqual(metrics["self_shot_kills"], 1)
        self.assertEqual(metrics["target_crashes"], 1)
        self.assertEqual(metrics["timeouts"], 1)
        self.assertEqual(metrics["stages"]["2"]["episodes"], 3)


if __name__ == "__main__":
    unittest.main()
