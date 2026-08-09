#!/usr/bin/env python3
"""Analyze AI Pilot dogfight summaries and flight CSV logs.

The script intentionally uses only the Python standard library so it can run
on both macOS and Windows without the simulator DLLs or extra dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_WEZ_MIN_RANGE_M = 152.4
DEFAULT_WEZ_MAX_RANGE_M = 914.4
DEFAULT_WEZ_ANGLE_DEG = 2.0
EARTH_METERS_PER_DEGREE = 111_320.0


@dataclass
class FlightGeometry:
    duration_s: float | None = None
    min_altitude_m: float | None = None
    max_abs_roll_deg: float | None = None
    max_abs_pitch_deg: float | None = None
    calculated_min_distance_m: float | None = None
    calculated_mean_distance_m: float | None = None
    calculated_min_ata_deg: float | None = None
    calculated_ata_at_min_distance_deg: float | None = None
    calculated_wez_steps: int | None = None
    calculated_wez_total_s: float | None = None
    calculated_wez_max_continuous_s: float | None = None


@dataclass
class EpisodeResult:
    label: str
    stage: int | None
    iteration: int | None
    episode: int | None
    result_class: str
    outcome: str
    end_condition: str
    ownship_health: float | None
    target_health: float | None
    damage_dealt_percent: float | None
    damage_received_percent: float | None
    episode_steps: int | None
    duration_s: float | None
    min_distance_m: float | None
    mean_distance_m: float | None
    final_ata_deg: float | None
    final_aa_deg: float | None
    official_wez_steps: int | None
    official_wez_entered: bool | None
    calculated_min_distance_m: float | None
    calculated_mean_distance_m: float | None
    calculated_min_ata_deg: float | None
    calculated_ata_at_min_distance_deg: float | None
    calculated_wez_steps: int | None
    calculated_wez_total_s: float | None
    calculated_wez_max_continuous_s: float | None
    min_altitude_m: float | None
    max_abs_roll_deg: float | None
    max_abs_pitch_deg: float | None
    summary_path: str
    ownship_log: str
    target_log: str


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: object) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _first_value(mapping: dict[str, object], names: Sequence[str]) -> object:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return None


def classify_result(summary: dict[str, object]) -> str:
    """Separate actual kills from aircraft crashes and timeouts."""
    outcome = str(summary.get("outcome", "")).strip().lower()
    end_condition = str(summary.get("end_condition", "")).strip().lower()

    if "target altitude below" in end_condition:
        return "target_crash"
    if "ownship altitude below" in end_condition:
        return "ownship_crash"
    if "target destroyed" in end_condition:
        return "self_shot_kill"
    if "ownship destroyed" in end_condition:
        return "combat_loss"
    if "target" in end_condition and "fdm" in end_condition:
        return "target_fdm_failure"
    if ("ownship" in end_condition and "fdm" in end_condition) or (
        "fdm" in end_condition and "target" not in end_condition
    ):
        return "ownship_fdm_failure"
    if "time" in end_condition or outcome in {"timeout", "truncated"}:
        return "timeout"
    if outcome == "crash":
        return "ownship_crash"
    if outcome == "win":
        return "win_unverified"
    if outcome in {"loss", "lose"}:
        return "loss_unverified"
    if outcome == "draw":
        return "draw"
    return "unknown"


def parse_location(path: Path) -> tuple[int | None, int | None, int | None]:
    text = str(path)
    stage_match = re.search(r"stage[_-]?(\d+)", text, re.IGNORECASE)
    iteration_match = re.search(r"(?:iter|iteration)[_-]?(\d+)", text, re.IGNORECASE)
    episode_match = re.search(r"(?:episode|game)[_-]?(\d+)", text, re.IGNORECASE)
    return (
        int(stage_match.group(1)) if stage_match else None,
        int(iteration_match.group(1)) if iteration_match else None,
        int(episode_match.group(1)) if episode_match else None,
    )


def find_flight_logs(summary_path: Path) -> tuple[Path | None, Path | None]:
    prefix = summary_path.name.removesuffix("_summary.json")
    siblings = list(summary_path.parent.glob("*.csv"))

    def choose(keyword: str) -> Path | None:
        exact = [p for p in siblings if p.name.startswith(prefix) and keyword in p.name.lower()]
        fallback = [p for p in siblings if keyword in p.name.lower()]
        candidates = exact or fallback
        return sorted(candidates)[0] if len(candidates) == 1 else None

    return choose("ownship"), choose("target")


def _read_flight_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _column(row: dict[str, str], candidates: Sequence[str]) -> float | None:
    normalized = {key.strip().lower(): value for key, value in row.items() if key}
    for candidate in candidates:
        value = normalized.get(candidate.lower())
        number = _float(value)
        if number is not None:
            return number
    return None


def _median_step_seconds(times: list[float]) -> float:
    deltas = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
    return statistics.median(deltas) if deltas else 0.0


def analyze_geometry(
    ownship_path: Path | None,
    target_path: Path | None,
    wez_min_range_m: float,
    wez_max_range_m: float,
    wez_angle_deg: float,
) -> FlightGeometry:
    if ownship_path is None:
        return FlightGeometry()

    own_rows = _read_flight_csv(ownship_path)
    if not own_rows:
        return FlightGeometry()

    times = [
        value
        for row in own_rows
        if (value := _column(row, ["Time", "time_s", "timestamp"])) is not None
    ]
    altitudes = [
        value
        for row in own_rows
        if (value := _column(row, ["Altitude", "altitude_m", "alt"])) is not None
    ]
    rolls = [
        abs(value)
        for row in own_rows
        if (value := _column(row, ["Roll (deg)", "roll_deg", "roll"])) is not None
    ]
    pitches = [
        abs(value)
        for row in own_rows
        if (value := _column(row, ["Pitch (deg)", "pitch_deg", "pitch"])) is not None
    ]
    geometry = FlightGeometry(
        duration_s=(max(times) - min(times)) if len(times) >= 2 else None,
        min_altitude_m=min(altitudes) if altitudes else None,
        max_abs_roll_deg=max(rolls) if rolls else None,
        max_abs_pitch_deg=max(pitches) if pitches else None,
    )
    if target_path is None:
        return geometry

    target_rows = _read_flight_csv(target_path)
    count = min(len(own_rows), len(target_rows))
    if count == 0:
        return geometry

    ranges: list[float] = []
    atas: list[float] = []
    wez_flags: list[bool] = []
    paired_times: list[float] = []

    for own, target in zip(own_rows[:count], target_rows[:count]):
        own_lat = _column(own, ["Latitude", "latitude_deg", "lat"])
        own_lon = _column(own, ["Longitude", "longitude_deg", "lon"])
        own_alt = _column(own, ["Altitude", "altitude_m", "alt"])
        own_yaw = _column(own, ["Yaw (deg)", "yaw_deg", "yaw"])
        own_pitch = _column(own, ["Pitch (deg)", "pitch_deg", "pitch"])
        target_lat = _column(target, ["Latitude", "latitude_deg", "lat"])
        target_lon = _column(target, ["Longitude", "longitude_deg", "lon"])
        target_alt = _column(target, ["Altitude", "altitude_m", "alt"])
        if None in {
            own_lat,
            own_lon,
            own_alt,
            own_yaw,
            own_pitch,
            target_lat,
            target_lon,
            target_alt,
        }:
            continue

        mean_lat_rad = math.radians((own_lat + target_lat) / 2.0)
        north_m = (target_lat - own_lat) * EARTH_METERS_PER_DEGREE
        east_m = (
            (target_lon - own_lon)
            * EARTH_METERS_PER_DEGREE
            * math.cos(mean_lat_rad)
        )
        up_m = target_alt - own_alt
        distance_m = math.sqrt(north_m**2 + east_m**2 + up_m**2)
        if distance_m <= 0.0:
            continue

        yaw_rad = math.radians(own_yaw)
        pitch_rad = math.radians(own_pitch)
        forward_north = math.cos(pitch_rad) * math.cos(yaw_rad)
        forward_east = math.cos(pitch_rad) * math.sin(yaw_rad)
        forward_up = math.sin(pitch_rad)
        cosine = (
            forward_north * north_m + forward_east * east_m + forward_up * up_m
        ) / distance_m
        ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

        ranges.append(distance_m)
        atas.append(ata_deg)
        wez_flags.append(
            wez_min_range_m <= distance_m <= wez_max_range_m
            and ata_deg <= wez_angle_deg
        )
        time_value = _column(own, ["Time", "time_s", "timestamp"])
        if time_value is not None:
            paired_times.append(time_value)

    if not ranges:
        return geometry

    step_seconds = _median_step_seconds(paired_times)
    longest_run = 0
    current_run = 0
    for in_wez in wez_flags:
        current_run = current_run + 1 if in_wez else 0
        longest_run = max(longest_run, current_run)

    min_index = min(range(len(ranges)), key=ranges.__getitem__)
    geometry.calculated_min_distance_m = min(ranges)
    geometry.calculated_mean_distance_m = statistics.fmean(ranges)
    geometry.calculated_min_ata_deg = min(atas)
    geometry.calculated_ata_at_min_distance_deg = atas[min_index]
    geometry.calculated_wez_steps = sum(wez_flags)
    geometry.calculated_wez_total_s = sum(wez_flags) * step_seconds
    geometry.calculated_wez_max_continuous_s = longest_run * step_seconds
    return geometry


def analyze_summary(
    label: str,
    summary_path: Path,
    wez_min_range_m: float,
    wez_max_range_m: float,
    wez_angle_deg: float,
) -> EpisodeResult:
    with summary_path.open("r", encoding="utf-8-sig") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError("summary JSON root must be an object")

    ownship_log, target_log = find_flight_logs(summary_path)
    geometry = analyze_geometry(
        ownship_log,
        target_log,
        wez_min_range_m,
        wez_max_range_m,
        wez_angle_deg,
    )
    stage, iteration, episode = parse_location(summary_path)

    ownship_health = _float(summary.get("ownship_health"))
    target_health = _float(summary.get("target_health"))
    damage_dealt = _float(summary.get("damage_dealt_percent"))
    damage_received = _float(summary.get("damage_received_percent"))
    if damage_dealt is None and target_health is not None:
        damage_dealt = max(0.0, (1.0 - target_health) * 100.0)
    if damage_received is None and ownship_health is not None:
        damage_received = max(0.0, (1.0 - ownship_health) * 100.0)

    official_min_distance = _float(
        _first_value(summary, ["ep_min_distance_m", "min_distance_m", "min_distance"])
    )
    official_mean_distance = _float(
        _first_value(summary, ["ep_mean_distance_m", "mean_distance_m", "mean_distance"])
    )

    return EpisodeResult(
        label=label,
        stage=stage,
        iteration=iteration,
        episode=episode,
        result_class=classify_result(summary),
        outcome=str(summary.get("outcome", "")),
        end_condition=str(summary.get("end_condition", "")),
        ownship_health=ownship_health,
        target_health=target_health,
        damage_dealt_percent=damage_dealt,
        damage_received_percent=damage_received,
        episode_steps=_int(_first_value(summary, ["episode_steps", "steps"])),
        duration_s=geometry.duration_s,
        min_distance_m=official_min_distance,
        mean_distance_m=official_mean_distance,
        final_ata_deg=_float(summary.get("final_ata_deg")),
        final_aa_deg=_float(summary.get("final_aa_deg")),
        official_wez_steps=_int(summary.get("ep_wez_steps")),
        official_wez_entered=_bool(summary.get("wez_entered")),
        calculated_min_distance_m=geometry.calculated_min_distance_m,
        calculated_mean_distance_m=geometry.calculated_mean_distance_m,
        calculated_min_ata_deg=geometry.calculated_min_ata_deg,
        calculated_ata_at_min_distance_deg=geometry.calculated_ata_at_min_distance_deg,
        calculated_wez_steps=geometry.calculated_wez_steps,
        calculated_wez_total_s=geometry.calculated_wez_total_s,
        calculated_wez_max_continuous_s=geometry.calculated_wez_max_continuous_s,
        min_altitude_m=geometry.min_altitude_m,
        max_abs_roll_deg=geometry.max_abs_roll_deg,
        max_abs_pitch_deg=geometry.max_abs_pitch_deg,
        summary_path=str(summary_path),
        ownship_log=str(ownship_log) if ownship_log else "",
        target_log=str(target_log) if target_log else "",
    )


def discover_summaries(root: Path) -> list[Path]:
    if root.is_file() and root.name.endswith("_summary.json"):
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"input path does not exist: {root}")
    return sorted(root.rglob("*_summary.json"))


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty label in input: {spec}")
        return label, Path(raw_path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return path.name or "experiment", path


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _maximum(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _minimum(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return min(clean) if clean else None


def summarize_label(results: Sequence[EpisodeResult]) -> dict[str, object]:
    counts = Counter(result.result_class for result in results)
    episodes = len(results)
    official_wez_entries = sum(result.official_wez_entered is True for result in results)
    calculated_wez_entries = sum(
        (result.calculated_wez_steps or 0) > 0 for result in results
    )
    official_wez_observations = sum(
        result.official_wez_entered is not None or result.official_wez_steps is not None
        for result in results
    )
    return {
        "episodes": episodes,
        "result_counts": dict(sorted(counts.items())),
        "self_shot_kills": counts["self_shot_kill"],
        "target_crashes": counts["target_crash"],
        "ownship_crashes": counts["ownship_crash"],
        "timeouts": counts["timeout"],
        "self_shot_kill_rate": counts["self_shot_kill"] / episodes if episodes else 0.0,
        "ownship_crash_rate": counts["ownship_crash"] / episodes if episodes else 0.0,
        "average_damage_dealt_percent": _mean(
            result.damage_dealt_percent for result in results
        ),
        "average_damage_received_percent": _mean(
            result.damage_received_percent for result in results
        ),
        "average_min_distance_m": _mean(
            result.min_distance_m
            if result.min_distance_m is not None
            else result.calculated_min_distance_m
            for result in results
        ),
        "minimum_altitude_m": _minimum(result.min_altitude_m for result in results),
        "maximum_abs_roll_deg": _maximum(result.max_abs_roll_deg for result in results),
        "maximum_abs_pitch_deg": _maximum(result.max_abs_pitch_deg for result in results),
        "official_wez_entries": official_wez_entries,
        "official_wez_observations": official_wez_observations,
        "calculated_wez_entries": calculated_wez_entries,
        "maximum_calculated_wez_hold_s": _maximum(
            result.calculated_wez_max_continuous_s for result in results
        ),
    }


def build_summary(results: Sequence[EpisodeResult]) -> dict[str, object]:
    labels = sorted({result.label for result in results})
    label_summaries: dict[str, object] = {}
    for label in labels:
        label_results = [result for result in results if result.label == label]
        label_summary = summarize_label(label_results)
        stages = sorted(
            {result.stage for result in label_results if result.stage is not None}
        )
        label_summary["stages"] = {
            str(stage): summarize_label(
                [result for result in label_results if result.stage == stage]
            )
            for stage in stages
        }
        label_summaries[label] = label_summary
    return {
        "labels": label_summaries,
        "total_episodes": len(results),
    }


def _fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# 교전 로그 자동 분석 결과",
        "",
        "## 모델 비교",
        "",
        "| 모델 | 경기 | 자체 격추 | 상대 추락 | 아군 추락 | Timeout | 평균 가한 피해 | 평균 받은 피해 | 평균 최소 거리 | WEZ 진입 | 최대 WEZ 유지 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = summary.get("labels", {})
    assert isinstance(labels, dict)
    for label, raw_metrics in labels.items():
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        wez_entries = (
            metrics.get("official_wez_entries")
            if metrics.get("official_wez_observations")
            else metrics.get("calculated_wez_entries")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(label),
                    _fmt(metrics.get("episodes"), 0),
                    _fmt(metrics.get("self_shot_kills"), 0),
                    _fmt(metrics.get("target_crashes"), 0),
                    _fmt(metrics.get("ownship_crashes"), 0),
                    _fmt(metrics.get("timeouts"), 0),
                    f"{_fmt(metrics.get('average_damage_dealt_percent'))}%",
                    f"{_fmt(metrics.get('average_damage_received_percent'))}%",
                    f"{_fmt(metrics.get('average_min_distance_m'))}m",
                    _fmt(wez_entries, 0),
                    f"{_fmt(metrics.get('maximum_calculated_wez_hold_s'), 2)}초",
                ]
            )
            + " |"
        )

    stage_rows: list[str] = []
    for label, raw_metrics in labels.items():
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        stages = metrics.get("stages", {})
        if not isinstance(stages, dict):
            continue
        for stage, raw_stage_metrics in stages.items():
            stage_metrics = raw_stage_metrics if isinstance(raw_stage_metrics, dict) else {}
            wez_entries = (
                stage_metrics.get("official_wez_entries")
                if stage_metrics.get("official_wez_observations")
                else stage_metrics.get("calculated_wez_entries")
            )
            stage_rows.append(
                "| "
                + " | ".join(
                    [
                        str(label),
                        str(stage),
                        _fmt(stage_metrics.get("episodes"), 0),
                        _fmt(stage_metrics.get("self_shot_kills"), 0),
                        _fmt(stage_metrics.get("target_crashes"), 0),
                        _fmt(stage_metrics.get("ownship_crashes"), 0),
                        _fmt(stage_metrics.get("timeouts"), 0),
                        f"{_fmt(stage_metrics.get('average_damage_dealt_percent'))}%",
                        f"{_fmt(stage_metrics.get('average_min_distance_m'))}m",
                        _fmt(wez_entries, 0),
                    ]
                )
                + " |"
            )

    if stage_rows:
        lines.extend(
            [
                "",
                "## Stage별 결과",
                "",
                "| 모델 | Stage | 경기 | 자체 격추 | 상대 추락 | 아군 추락 | Timeout | 평균 가한 피해 | 평균 최소 거리 | WEZ 진입 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *stage_rows,
            ]
        )

    lines.extend(["", "## 판정 기준", ""])
    lines.extend(
        [
            "- `self_shot_kill`: `target destroyed`로 종료된 실제 격추",
            "- `target_crash`: 상대가 최저 고도 아래로 내려가 종료",
            "- `ownship_crash`: 아군이 최저 고도 아래로 내려가 종료",
            "- `timeout`: 제한 시간 종료",
            "- 공식 WEZ 값이 없을 때만 비행 CSV의 거리와 ATA로 근사 계산",
            "",
            "## 해석 시 주의점",
            "",
            "- 동일 seed와 초기 조건으로 평가한 결과인지 별도로 확인해야 한다.",
            "- 상대 추락은 자체 사격 격추에 포함하지 않는다.",
            "- CSV 기반 ATA와 WEZ는 로그 좌표로 다시 계산한 근사값이다.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    results: Sequence[EpisodeResult], summary: dict[str, object], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_results.csv"
    field_names = [field.name for field in fields(EpisodeResult)]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    (output_dir / "evaluation_summary.md").write_text(
        build_markdown(summary), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze AI Pilot dogfight summary JSON and flight CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="[LABEL=]PATH",
        help="One or more result folders. Prefix with LABEL= for comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output"),
        help="Output directory (default: ./analysis_output)",
    )
    parser.add_argument(
        "--wez-min-range-m", type=float, default=DEFAULT_WEZ_MIN_RANGE_M
    )
    parser.add_argument(
        "--wez-max-range-m", type=float, default=DEFAULT_WEZ_MAX_RANGE_M
    )
    parser.add_argument("--wez-angle-deg", type=float, default=DEFAULT_WEZ_ANGLE_DEG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wez_min_range_m < 0 or args.wez_max_range_m <= args.wez_min_range_m:
        print("error: invalid WEZ range", file=sys.stderr)
        return 2
    if not 0 < args.wez_angle_deg <= 180:
        print("error: WEZ angle must be between 0 and 180 degrees", file=sys.stderr)
        return 2

    results: list[EpisodeResult] = []
    labels_seen: set[str] = set()
    try:
        for spec in args.inputs:
            label, root = parse_input_spec(spec)
            if label in labels_seen:
                raise ValueError(f"duplicate label: {label}")
            labels_seen.add(label)
            summaries = discover_summaries(root)
            if not summaries:
                raise ValueError(f"no *_summary.json files found under: {root}")
            for summary_path in summaries:
                results.append(
                    analyze_summary(
                        label,
                        summary_path,
                        args.wez_min_range_m,
                        args.wez_max_range_m,
                        args.wez_angle_deg,
                    )
                )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = build_summary(results)
    write_outputs(results, summary, args.output.resolve())
    print(f"Analyzed {len(results)} episodes.")
    print(f"Results: {args.output.resolve() / 'evaluation_results.csv'}")
    print(f"Summary: {args.output.resolve() / 'evaluation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
