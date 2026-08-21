from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST_SHA256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    failures: list[str] = []
    checked = 0
    expected: set[str] = set()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = row["RelativePath"]
            normalized = Path(relative).as_posix()
            if normalized in expected:
                failures.append(f"duplicate manifest entry: {relative}")
                continue
            expected.add(normalized)
            path = (ROOT / relative).resolve()
            try:
                path.relative_to(ROOT)
            except ValueError:
                failures.append(f"outside package: {relative}")
                continue
            if not path.is_file():
                failures.append(f"missing: {relative}")
                continue
            checked += 1
            if path.stat().st_size != int(row["Bytes"]):
                failures.append(f"size mismatch: {relative}")
            if sha256(path) != row["SHA256"].upper():
                failures.append(f"SHA-256 mismatch: {relative}")

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    for relative in sorted(actual - expected):
        failures.append(f"unexpected file: {relative}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"[OK] Manifest verified: {checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
