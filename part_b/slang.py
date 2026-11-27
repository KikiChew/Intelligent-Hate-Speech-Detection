"""Dataset-driven slang normalization for Sentinel AI.

Source: https://www.kaggle.com/datasets/gowrishankarp/chat-slang-abbreviations-acronyms

No slang dictionary is hard-coded. Keep Kaggle dataset files inside
part_b/data/slang/ (JSON preferred, CSV also accepted).

Public backward-compatible API: normalize_slang(text: str) -> str
"""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parent / "data" / "slang"
DATASET_SOURCE = "https://www.kaggle.com/datasets/gowrishankarp/chat-slang-abbreviations-acronyms"
# Do not transform URLs, email addresses, or @mentions.
PROTECTED_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|@[A-Za-z0-9_]+", re.I)


def _valid_pair(abbrev: Any, expansion: Any) -> tuple[str, str] | None:
    if not isinstance(abbrev, str) or not isinstance(expansion, str):
        return None
    short = " ".join(abbrev.strip().split()).lower()
    full = " ".join(expansion.strip().split())
    if (not short or not full or len(short) > 50 or len(full) > 250
            or short == full.lower() or not re.search(r"[a-z0-9]", short)
            or "\n" in short or "\n" in full):
        return None
    return short, full


def _pairs_from_json(data: Any) -> list[tuple[str, str]]:
    # Kaggle-derived copies have appeared as {"2day":"today"} and
    # occasionally {"root": {"2day":"today"}}.
    if isinstance(data, dict):
        if "root" in data and isinstance(data["root"], (dict, list)):
            return _pairs_from_json(data["root"])
        if "data" in data and isinstance(data["data"], (dict, list)):
            return _pairs_from_json(data["data"])
        result = []
        for key, value in data.items():
            if isinstance(value, str):
                pair = _valid_pair(key, value)
                if pair:
                    result.append(pair)
            elif isinstance(value, dict):
                meaning = next((value[k] for k in ("meaning", "full_form", "expansion", "full", "original") if isinstance(value.get(k), str)), None)
                pair = _valid_pair(key, meaning)
                if pair:
                    result.append(pair)
        return result
    if isinstance(data, list):
        result = []
        for row in data:
            if not isinstance(row, dict):
                continue
            lowered = {str(k).strip().lower(): v for k, v in row.items()}
            a = next((lowered[k] for k in ("slang", "abbreviation", "acronym", "short_form", "shortform", "short", "term") if k in lowered), None)
            b = next((lowered[k] for k in ("meaning", "full_form", "expansion", "original", "full", "definition") if k in lowered), None)
            pair = _valid_pair(a, b)
            if pair:
                result.append(pair)
        return result
    return []


def _pairs_from_csv(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        result = []
        for row in reader:
            lowered = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            a = next((lowered[k] for k in ("slang", "abbreviation", "acronym", "short_form", "shortform", "short", "term") if k in lowered), None)
            b = next((lowered[k] for k in ("meaning", "full_form", "expansion", "original", "full", "definition") if k in lowered), None)
            pair = _valid_pair(a, b)
            if pair:
                result.append(pair)
        return result


@lru_cache(maxsize=1)
def load_slang_mapping() -> dict[str, str]:
    """Load abbreviation mappings from Kaggle dataset files; never a hard-coded list."""
    files = sorted(DATASET_DIR.glob("*.json")) + sorted(DATASET_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Slang dataset not found at {DATASET_DIR}. Download the Kaggle "
            f"dataset ({DATASET_SOURCE}) and extract its .json/.csv file here."
        )
    mapping: dict[str, str] = {}
    for path in files:
        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8-sig") as f:
                    pairs = _pairs_from_json(json.load(f))
            else:
                pairs = _pairs_from_csv(path)
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
            raise ValueError(f"Cannot read slang dataset {path.name}: {exc}") from exc
        for short, full in pairs:
            mapping.setdefault(short, full)
    if not mapping:
        raise ValueError(f"No valid slang mappings found in {DATASET_DIR}. Check the dataset file format.")
    return mapping


@lru_cache(maxsize=1)
def _substitution_pattern() -> re.Pattern[str]:
    terms = sorted(load_slang_mapping(), key=lambda x: (-len(x), x))
    # Whole-term matches, including phrases and alphanumeric chat terms.
    escaped = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in terms)
    return re.compile(r"(?<!\w)(?:" + escaped + r")(?!\w)", re.I)


def normalize_slang(text: str) -> str:
    """Expand matching abbreviations; preserve unrecognized text and punctuation."""
    if not isinstance(text, str):
        return text
    if not text.strip():
        return text
    mapping = load_slang_mapping()
    pattern = _substitution_pattern()
    protected = list(PROTECTED_PATTERN.finditer(text))
    spans = [(m.start(), m.end()) for m in protected]

    def replace(match: re.Match[str]) -> str:
        a, b = match.span()
        if any(a < end and b > start for start, end in spans):
            return match.group(0)
        key = " ".join(match.group(0).lower().split())
        return mapping.get(key, match.group(0))

    return pattern.sub(replace, text)


def get_slang_stats() -> dict[str, Any]:
    """Useful for displaying dataset provenance/coverage in the Streamlit UI."""
    mapping = load_slang_mapping()
    return {"entries": len(mapping), "source": DATASET_SOURCE,
            "files": [p.name for p in sorted(DATASET_DIR.glob("*.json")) + sorted(DATASET_DIR.glob("*.csv"))]}


if __name__ == "__main__":
    print("Dataset:", get_slang_stats())
    for sample in ("idk why u said that lol", "stfu bro", "Visit https://example.com/idk"):
        print(sample, "=>", normalize_slang(sample))
