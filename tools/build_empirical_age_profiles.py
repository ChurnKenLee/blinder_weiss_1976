#!/usr/bin/env python3
"""Build descriptive ACS/ATUS age profiles; retain survey units and universes.

These profiles are inputs to measurement design, not automatically accepted
model calibration targets. No individual survey records are written or printed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ipums_download import atomic_json, now  # noqa: E402

AGE_BINS = ((18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 79), (80, 84))


def dictionary(directory: Path) -> tuple[dict, dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    columns = {}
    for node in ET.parse(next(directory.glob("*.xml"))).getroot().findall(".//{*}var"):
        location = node.find("{*}location")
        columns[node.attrib["name"]] = (
            int(location.attrib["StartPos"]) - 1,
            int(location.attrib["EndPos"]),
            int(node.attrib.get("dcml", "0")),
        )
    return columns, manifest


def records(directory: Path, names: list[str]):
    columns, manifest = dictionary(directory)
    source = next(directory.glob("*.dat.gz"))
    expected = next(f for f in manifest["files"] if f["file"] == source.name)
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
            raise ValueError("Source data checksum no longer matches the validated manifest")
    width = max(c[1] for c in columns.values())
    with gzip.open(source, "rb") as stream:
        for raw in stream:
            raw = raw.rstrip(b"\r\n")
            if len(raw) != width:
                raise ValueError("Unexpected fixed-width record length")
            result = {}
            for name in names:
                start, end, decimals = columns[name]
                token = raw[start:end]
                unscaled = float(token) if b"." in token else int(token)
                result[name] = unscaled / 10**decimals if decimals else unscaled
            yield result


def accumulate(groups, age: int, sex: int, weight: float, observables: dict):
    if weight < 0:
        raise ValueError("Negative survey weight")
    band = next((f"{lo}-{hi}" for lo, hi in AGE_BINS if lo <= age <= hi), None)
    if band is None or sex not in (1, 2):
        return
    for label in ("all", "men" if sex == 1 else "women"):
        key = (label, band)
        row = groups[key]
        row["record_count"] += 1
        row["weight"] += weight
        row["squared_weight"] += weight * weight
        row["weighted_age"] += weight * age
        for name, value in observables.items():
            row[name + "_numerator"] += 0.0
            row[name + "_denominator"] += 0.0
            if value is not None:
                row[name + "_numerator"] += weight * value
                row[name + "_denominator"] += weight


def finish(groups):
    rows = []
    for (sex, band), group in sorted(groups.items()):
        result = {
            "sex": sex,
            "age_band": band,
            "records": int(group["record_count"]),
            "survey_weight_sum": group["weight"],
            "effective_sample_size_weights_only": group["weight"] ** 2 / group["squared_weight"],
            "weighted_mean_recorded_age": group["weighted_age"] / group["weight"],
        }
        for name in group:
            if name.endswith("_numerator"):
                field = name.removesuffix("_numerator")
                denominator = group[field + "_denominator"]
                result[field] = group[name] / denominator if denominator else None
                result[field + "_coverage"] = denominator / group["weight"]
        rows.append(result)
    return rows


def acs_profiles(directory: Path):
    groups = defaultdict(lambda: defaultdict(float))
    count = excluded_institutions = 0
    names = ["YEAR", "AGE", "SEX", "PERWT", "GQ", "EMPSTAT", "UHRSWORK"]
    for row in records(directory, names):
        count += 1
        if row["YEAR"] != 2024:
            raise ValueError("This measurement specification requires 2024")
        if row["GQ"] == 3:
            excluded_institutions += 1
            continue
        status, hours = int(row["EMPSTAT"]), row["UHRSWORK"]
        accumulate(
            groups,
            int(row["AGE"]),
            int(row["SEX"]),
            row["PERWT"],
            {
                "employment_rate": float(status == 1) if status in (1, 2, 3) else None,
                "usual_weekly_hours_among_reporters": hours if 1 <= hours <= 99 else None,
                "hours_topcoded_share_among_reporters": float(hours == 99)
                if 1 <= hours <= 99
                else None,
            },
        )
    return {
        "source_records": count,
        "excluded_institutional_records_all_ages": excluded_institutions,
        "profiles": finish(groups),
    }


def atus_profiles(directory: Path):
    respondents = {}
    names = ["YEAR", "CASEID", "AGE", "SEX", "WT06", "EMPSTAT", "ACTIVITY", "DURATION"]
    activity_count = 0
    for row in records(directory, names):
        activity_count += 1
        if row["YEAR"] != 2024:
            raise ValueError("This measurement specification requires 2024")
        key = (int(row["YEAR"]), int(row["CASEID"]))
        attributes = (int(row["AGE"]), int(row["SEX"]), row["WT06"], int(row["EMPSTAT"]))
        person = respondents.setdefault(
            key, {"attributes": attributes, "total": 0, "working": 0, "education": 0}
        )
        if person["attributes"] != attributes:
            raise ValueError("Respondent attributes differ across activities")
        duration, activity = int(row["DURATION"]), int(row["ACTIVITY"])
        if not 0 <= duration <= 1440:
            raise ValueError("Invalid diary duration")
        person["total"] += duration
        person["working"] += duration * (activity // 100 == 501)
        person["education"] += duration * (activity // 10000 == 6)
    groups = defaultdict(lambda: defaultdict(float))
    for person in respondents.values():
        if person["total"] != 1440:
            raise ValueError("Diary must total 1440 minutes")
        age, sex, weight, status = person["attributes"]
        accumulate(
            groups,
            age,
            sex,
            weight,
            {
                "employment_rate": float(status in (1, 2)) if status in (1, 2, 3, 4, 5) else None,
                "working_minutes_per_day": person["working"],
                "education_minutes_per_day": person["education"],
            },
        )
    return {
        "source_activity_records": activity_count,
        "source_respondents": len(respondents),
        "all_diaries_1440_minutes": True,
        "profiles": finish(groups),
    }


def build(root: Path, output: Path):
    report = {
        "created_at": now(),
        "status": "descriptive_profiles_pending_model_measurement_mapping",
        "year": 2024,
        "age_bins": AGE_BINS,
        "measurement": {
            "sex_profiles": "All, men, and women separately; no weighting across profile rows",
            "age_topcoding": (
                "ATUS code80 pools ages80-84; shown as a band, never interpreted "
                "as exact age80"
            ),
            "acs_universe": "All respondents age18-84 excluding institutional group quarters GQ=3",
            "atus_universe": "ATUS respondent-day population age18-84; WT06 applied once per diary",
            "employment": "ACS EMPSTAT=1; ATUS EMPSTAT in {1,2}; nonemployment kept in denominator",
            "acs_hours": (
                "UHRSWORK 1..99 only;0 N/A excluded with coverage reported;99 "
                "retained as topcoded"
            ),
            "acs_reference_period": (
                "Employment is reference-week status; usual hours describe weeks "
                "worked in preceding12months"
            ),
            "atus_working": (
                "ACTIVITY0501xx including waiting/security; excludes work travel "
                "and other work-related categories"
            ),
            "atus_education": (
                "ACTIVITY06xxxx; broader than productive training and incomplete "
                "for learning on the job"
            ),
            "standard_errors": (
                "Not estimated; weights-only effective sample size is not a "
                "survey variance estimate"
            ),
            "unobserved_states": (
                "Neither extract supplies model assets, consumption, or latent "
                "human capital"
            ),
            "model_mapping": (
                "No hours-endowment, dollar normalization, initial asset law, or "
                "calibration loss weights imposed"
            ),
        },
        "sources": {
            "acs": "https://usa.ipums.org/usa-action/variables/EMPSTAT",
            "hours": "https://usa.ipums.org/usa-action/variables/UHRSWORK",
            "atus": "https://www.atusdata.org/atus-action/variables/ACTIVITY",
        },
        "acs": acs_profiles(root / "raw/usa_00041"),
        "atus": atus_profiles(root / "raw/atus_00004"),
        "source_manifests": {
            name: json.loads((root / "raw" / name / "manifest.json").read_text())["files"]
            for name in ("usa_00041", "atus_00004")
        },
    }
    atomic_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "acs_records": report["acs"]["source_records"],
                "atus_respondents": report["atus"]["source_respondents"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ipums"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/calibration/empirical_age_profiles_2024.json")
    )
    args = parser.parse_args()
    build(args.root, args.output)
