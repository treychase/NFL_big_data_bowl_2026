"""Reading the competition files, and the judgement calls made while reading."""

import pandas as pd
import pytest

from nfl_scouting import data


class TestHeightParsing:
    @pytest.mark.parametrize("height, inches", [
        ("6-1", 73.0), ("5-11", 71.0), ("6-0", 72.0), ("7-0", 84.0),
    ])
    def test_feet_dash_inches(self, height, inches):
        assert data._parse_height(height) == inches

    @pytest.mark.parametrize("bad", ["", "six-one", "72", None, float("nan"), 71])
    def test_unparseable_heights_become_nan_not_a_guess(self, bad):
        assert pd.isna(data._parse_height(bad))

    def test_vectorised_parse_keeps_the_index(self):
        heights = pd.Series(["6-2", "bad"], index=[7, 9])
        parsed = data.parse_heights(heights)
        assert parsed.loc[7] == 74.0
        assert pd.isna(parsed.loc[9])
        assert parsed.dtype == "float64"


def _supplementary(rows):
    base = {
        "play_description": "", "pass_result": "C", "play_nullified_by_penalty": "N",
        "yards_gained": 10, "team_coverage_man_zone": "ZONE_COVERAGE",
        "team_coverage_type": "COVER_3_ZONE", "route_of_targeted_receiver": "HITCH",
        "offense_formation": "SHOTGUN", "receiver_alignment": "3x1",
        "dropback_type": "TRADITIONAL", "pass_location_type": "INSIDE_BOX",
        "play_action": "FALSE", "game_id": 1, "play_id": 1,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


class TestTouchdownExtraction:
    """The file only says TOUCHDOWN inside the play description text."""

    def test_a_completed_touchdown_counts(self, tmp_path, monkeypatch):
        frame = _supplementary([
            {"pass_result": "C", "play_description": "J.Goff pass short right. TOUCHDOWN."},
        ])
        assert self._flag(frame, tmp_path, monkeypatch).iloc[0]

    def test_a_pick_six_is_not_a_passing_touchdown(self, tmp_path, monkeypatch):
        """The defence scored. Charging it to the passer's touchdown column
        would put an interception and a touchdown on the same throw."""
        frame = _supplementary([
            {"pass_result": "IN", "play_description": "INTERCEPTED. Returned. TOUCHDOWN."},
        ])
        assert not self._flag(frame, tmp_path, monkeypatch).iloc[0]

    def test_an_incompletion_mentioning_a_touchdown_does_not_count(self, tmp_path, monkeypatch):
        frame = _supplementary([
            {"pass_result": "I", "play_description": "Pass incomplete in the end zone. "
                                                     "Penalty enforced from the TOUCHDOWN spot."},
        ])
        assert not self._flag(frame, tmp_path, monkeypatch).iloc[0]

    @staticmethod
    def _flag(frame, tmp_path, monkeypatch):
        path = tmp_path / "supplementary_data.csv"
        frame.to_csv(path, index=False)
        monkeypatch.setenv("BDB2026_DATA_DIR", str(tmp_path))
        return data.load_supplementary(tmp_path)["is_touchdown"]


class TestAnalysisPlays:
    def test_a_nullified_play_is_dropped(self, tmp_path):
        """A flag wiped the snap out, so nobody should be charged for it."""
        frame = _supplementary([
            {"play_id": 1, "play_nullified_by_penalty": "N"},
            {"play_id": 2, "play_nullified_by_penalty": "Y"},
        ])
        path = tmp_path / "supplementary_data.csv"
        frame.to_csv(path, index=False)
        plays = data.analysis_plays(data.load_supplementary(tmp_path))
        assert plays["play_id"].tolist() == [1]

    def test_non_pass_results_are_dropped(self, tmp_path):
        frame = _supplementary([
            {"play_id": 1, "pass_result": "C"},
            {"play_id": 2, "pass_result": "S"},     # sack: no throw happened
            {"play_id": 3, "pass_result": "IN"},
        ])
        (tmp_path / "supplementary_data.csv").write_text(frame.to_csv(index=False))
        plays = data.analysis_plays(data.load_supplementary(tmp_path))
        assert sorted(plays["play_id"]) == [1, 3]

    def test_result_flags_are_mutually_exclusive(self, tmp_path):
        frame = _supplementary([
            {"play_id": 1, "pass_result": "C"},
            {"play_id": 2, "pass_result": "I"},
            {"play_id": 3, "pass_result": "IN"},
        ])
        (tmp_path / "supplementary_data.csv").write_text(frame.to_csv(index=False))
        plays = data.load_supplementary(tmp_path)
        assert not (plays["is_complete"] & plays["is_interception"]).any()
        assert plays["is_complete"].sum() == 1
        assert plays["is_interception"].sum() == 1


def test_missing_categoricals_become_a_visible_unknown(tmp_path):
    """A blank route must not silently join a real route's bucket."""
    frame = _supplementary([{"route_of_targeted_receiver": None}])
    (tmp_path / "supplementary_data.csv").write_text(frame.to_csv(index=False))
    loaded = data.load_supplementary(tmp_path)
    assert loaded["route_of_targeted_receiver"].iloc[0] == "UNKNOWN"
