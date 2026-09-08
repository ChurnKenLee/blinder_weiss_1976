from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "empirical_age_profiles", Path(__file__).parents[1] / "tools/build_empirical_age_profiles.py"
)
assert _spec is not None and _spec.loader is not None
profiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profiles)


def test_atus_weights_applied_once_and_zero_work_kept(monkeypatch):
    # A high-weight employed respondent reports no work in a complete diary;
    # a second respondent has two activities. Activity counts must not change
    # person weights or turn the first respondent into a nonparticipant.
    rows = [
        dict(YEAR=2024, CASEID=1, AGE=30, SEX=1, WT06=3, EMPSTAT=2,
             ACTIVITY=10101, DURATION=1440),
        dict(YEAR=2024, CASEID=2, AGE=30, SEX=2, WT06=1, EMPSTAT=1,
             ACTIVITY=50101, DURATION=480),
        dict(YEAR=2024, CASEID=2, AGE=30, SEX=2, WT06=1, EMPSTAT=1,
             ACTIVITY=10101, DURATION=960),
    ]
    monkeypatch.setattr(profiles, "records", lambda *_: iter(rows))
    report = profiles.atus_profiles(Path("unused"))
    combined = next(row for row in report["profiles"] if row["sex"] == "all")
    assert report["source_respondents"] == 2
    assert combined["survey_weight_sum"] == 4
    assert combined["employment_rate"] == 1
    assert combined["working_minutes_per_day"] == 120
    assert combined["education_minutes_per_day"] == 0
    rows[-1]["DURATION"] = 900
    with pytest.raises(ValueError, match="1440"):
        profiles.atus_profiles(Path("unused"))


def test_acs_denominators_and_hours_special_values(monkeypatch):
    rows = [
        dict(YEAR=2024, AGE=40, SEX=1, PERWT=3, GQ=1, EMPSTAT=3, UHRSWORK=0),
        dict(YEAR=2024, AGE=40, SEX=2, PERWT=1, GQ=1, EMPSTAT=1, UHRSWORK=40),
        dict(YEAR=2024, AGE=40, SEX=2, PERWT=100, GQ=3, EMPSTAT=1, UHRSWORK=99),
    ]
    monkeypatch.setattr(profiles, "records", lambda *_: iter(rows))
    report = profiles.acs_profiles(Path("unused"))
    combined = next(row for row in report["profiles"] if row["sex"] == "all")
    assert report["excluded_institutional_records_all_ages"] == 1
    assert combined["employment_rate"] == 0.25
    assert combined["usual_weekly_hours_among_reporters"] == 40
    assert combined["usual_weekly_hours_among_reporters_coverage"] == 0.25
    men = next(row for row in report["profiles"] if row["sex"] == "men")
    assert men["usual_weekly_hours_among_reporters"] is None
    assert men["usual_weekly_hours_among_reporters_coverage"] == 0
    np.testing.assert_allclose(combined["effective_sample_size_weights_only"], 1.6)
