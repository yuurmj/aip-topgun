from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
from pathlib import Path
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "TopGun_예선제출_팀명미정"
MAX_FILE_BYTES = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill team identity fields and build a verified submission ZIP."
    )
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--team-number", required=True)
    parser.add_argument("--university", required=True)
    parser.add_argument("--members", required=True, help="Names and roles on one line")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--date-tag", default=date.today().strftime("%Y%m%d"))
    return parser.parse_args()


def safe_filename_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("team name cannot form a safe Windows filename")
    return cleaned


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def replace_identity_text(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8-sig")
    for before, after in replacements.items():
        text = text.replace(before, after)
    path.write_text(text, encoding="utf-8")


def build_manifest(stage: Path) -> list[tuple[str, int, str]]:
    manifest = stage / "MANIFEST_SHA256.csv"
    rows: list[tuple[str, int, str]] = []
    for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == manifest:
            continue
        size = path.stat().st_size
        if size >= MAX_FILE_BYTES:
            raise ValueError(f"file is not below 1 GiB: {path}")
        rows.append((path.relative_to(stage).as_posix(), size, sha256(path)))
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["RelativePath", "Bytes", "SHA256"])
        writer.writerows(rows)
    return rows


def build_zip(stage: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())
    if zip_path.stat().st_size >= MAX_FILE_BYTES:
        raise ValueError(f"ZIP is not below 1 GiB: {zip_path}")


def main() -> int:
    args = parse_args()
    team_name = args.team_name.strip()
    if not team_name:
        raise ValueError("team name must not be empty")
    if any(ord(character) < 32 for character in team_name):
        raise ValueError("team name must be a single line without control characters")
    encoded_name = team_name.encode("utf-8")
    if len(encoded_name) > 29:
        raise ValueError(
            f"team name is {len(encoded_name)} UTF-8 bytes; server maximum is 29"
        )

    identity_fields = {
        "team number": args.team_number.strip(),
        "university": args.university.strip(),
        "members": args.members.strip(),
    }
    empty_fields = [label for label, value in identity_fields.items() if not value]
    if empty_fields:
        raise ValueError(
            "identity fields must not be empty: " + ", ".join(empty_fields)
        )
    if not re.fullmatch(r"\d{8}", args.date_tag):
        raise ValueError("date tag must contain exactly 8 digits (YYYYMMDD)")

    source = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename_component(team_name)
    stage = output_root / f"TopGun_예선제출_{safe_name}"
    zip_path = output_root / f"TopGun_예선제출_{safe_name}_{args.date_tag}.zip"
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256.txt")
    for target in (stage, zip_path, sidecar):
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output: {target}"
            )

    shutil.copytree(
        source,
        stage,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    replacements = {
        "TEAM_NAME_TO_BE_FILLED": team_name,
        "TEAM_NUMBER_TO_BE_FILLED": identity_fields["team number"],
        "UNIVERSITY_TO_BE_FILLED": identity_fields["university"],
        "MEMBERS_TO_BE_FILLED": identity_fields["members"],
        "팀명미정": safe_name,
    }
    for path in stage.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".bat", ".md", ".txt"}:
            replace_identity_text(path, replacements)

    (stage / "team_name.txt").write_text(team_name + "\n", encoding="utf-8")
    renames = {
        stage / "추가필요라이브러리_팀명미정.txt":
            stage / f"추가필요라이브러리_{safe_name}.txt",
        stage / "팀_소개서_팀명미정.txt":
            stage / f"팀_소개서_{safe_name}.txt",
    }
    for old, new in renames.items():
        if old.exists():
            old.rename(new)

    forbidden = {
        "TEAM_NAME_TO_BE_FILLED",
        "TEAM_NUMBER_TO_BE_FILLED",
        "UNIVERSITY_TO_BE_FILLED",
        "MEMBERS_TO_BE_FILLED",
    }
    for path in stage.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".bat", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8-sig")
            remaining = sorted(item for item in forbidden if item in text)
            if remaining:
                raise ValueError(f"unfilled placeholders in {path}: {remaining}")

    rows = build_manifest(stage)
    build_zip(stage, zip_path)
    archive_hash = sha256(zip_path)
    sidecar.write_text(
        f"{archive_hash} *{zip_path.name}\n", encoding="utf-8"
    )
    print(f"stage={stage}")
    print(f"manifest_files={len(rows)}")
    print(f"zip={zip_path}")
    print(f"zip_bytes={zip_path.stat().st_size}")
    print(f"zip_sha256={archive_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
