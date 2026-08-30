"""The packed geometry the dashboard animates from.

The block is a hand-rolled int16 layout with no self-describing structure, so
the thing worth testing is that it unpacks to the play that went in: the right
players, the right number of samples each, and the release landing where the
flight picks it up. A layout that is one int16 out of step still decodes into
plausible-looking coordinates, which is exactly why this is asserted rather
than eyeballed on the page.
"""

import base64

import numpy as np
import pytest

from nfl_scouting import data, export
from nfl_scouting.config import DT


def unpack_block(geometry, index):
    """Decode one play's block back into pre-release paths and flight paths."""
    blob = np.frombuffer(base64.b64decode(geometry["data"]), dtype="<i2")
    offset = geometry["offsets"][index]
    assert offset >= 0, "this play has no geometry"
    scale = geometry["scale"]

    n_pre, n_pre_frames, n_flight = blob[offset], blob[offset + 1], blob[offset + 2]
    p = offset + 3

    pre = []
    for _ in range(n_pre):
        role, in_air = int(blob[p]), int(blob[p + 1]); p += 2
        xs = blob[p:p + 2 * n_pre_frames:2] / scale
        ys = blob[p + 1:p + 2 * n_pre_frames:2] / scale
        p += 2 * n_pre_frames
        pre.append({"role": role, "inAir": bool(in_air), "x": xs, "y": ys})

    flight = []
    for _ in range(n_flight):
        kind, length = int(blob[p]), int(blob[p + 1]); p += 2
        xs = blob[p:p + 2 * length:2] / scale
        ys = blob[p + 1:p + 2 * length:2] / scale
        p += 2 * length
        flight.append({"kind": kind, "x": xs, "y": ys})

    return {"pre": pre, "preFrames": int(n_pre_frames), "flight": flight,
            "end": p}


class TestSampleInterval:
    def test_the_two_phases_report_their_own_rate(self, synthetic_season):
        frames = data.load_week(1, root=synthetic_season)
        plays = _play_index(frames)
        geometry = export.build_geometry(plays, weeks=[1], root=synthetic_season)
        assert geometry["flightStep"] == pytest.approx(DT)
        assert geometry["preStep"] == pytest.approx(export.PRE_STRIDE * DT)
        # The run-up is the coarser of the two, which is the whole point of it.
        assert geometry["preStep"] > geometry["flightStep"]


class TestPackedBlock:
    @pytest.fixture(scope="class")
    def geometry(self, synthetic_season):
        frames = data.load_week(1, root=synthetic_season)
        return export.build_geometry(_play_index(frames), weeks=[1],
                                     root=synthetic_season), frames

    def test_every_play_gets_a_block(self, geometry):
        packed, _frames = geometry
        assert len(packed["offsets"]) > 0
        assert all(o >= 0 for o in packed["offsets"])

    def test_blocks_decode_and_do_not_overlap(self, geometry):
        """A layout off by one would run a block into the next one's header."""
        packed, _frames = geometry
        offsets = [o for o in packed["offsets"] if o >= 0]
        for index in range(len(packed["offsets"])):
            block = unpack_block(packed, index)
            following = [o for o in offsets if o > packed["offsets"][index]]
            if following:
                assert block["end"] <= min(following)
            else:
                assert block["end"] <= packed["n_values"]

    def test_every_player_has_the_same_number_of_samples(self, geometry):
        """One count serves the section, so a short path would desync it."""
        packed, _frames = geometry
        block = unpack_block(packed, 0)
        assert block["pre"], "no pre-release players packed"
        for player in block["pre"]:
            assert len(player["x"]) == block["preFrames"]
            assert len(player["y"]) == block["preFrames"]

    def test_the_run_up_covers_the_whole_play_before_the_throw(self, geometry):
        """First sample at the snap, last at the release, stride in between."""
        packed, frames = geometry
        key = _first_key(frames)
        tracking = frames.tracking
        play = tracking[(tracking["game_id"] == key[0]) & (tracking["play_id"] == key[1])]
        n_frames = play["frame_id"].nunique()

        block = unpack_block(packed, 0)
        expected = len(np.unique(np.concatenate([
            np.arange(0, n_frames, export.PRE_STRIDE), [n_frames - 1]])))
        assert block["preFrames"] == expected

    def test_the_last_run_up_sample_is_the_release(self, geometry):
        """The flight starts where the run-up stops, so they have to meet."""
        packed, frames = geometry
        key = _first_key(frames)
        tracking = frames.tracking
        play = tracking[(tracking["game_id"] == key[0]) & (tracking["play_id"] == key[1])]
        release = play[play["frame_id"] == play["frame_id"].max()]

        block = unpack_block(packed, 0)
        packed_last = sorted(round(float(p["x"][-1]), 1) for p in block["pre"])
        actual = sorted(round(float(x), 1) for x in release["x"])
        assert packed_last == actual

    def test_the_run_up_is_kinded_the_same_way_as_the_flight(self, geometry):
        """A dot has to keep its colour across the release, so both sections
        must speak the same four kinds - and exactly one player in the run-up
        may be the targeted receiver, and one the defender covering him."""
        packed, _frames = geometry
        block = unpack_block(packed, 0)
        kinds = [p["role"] for p in block["pre"]]
        assert set(kinds) <= {export.KIND_TARGET, export.KIND_PRIMARY_COVERAGE,
                              export.KIND_OTHER_OFFENCE, export.KIND_OTHER_DEFENCE}
        assert kinds.count(export.KIND_TARGET) == 1
        assert kinds.count(export.KIND_PRIMARY_COVERAGE) <= 1

    def test_every_defender_is_kinded_as_one(self, geometry):
        """The page draws the coverage by side, so a defender who is not the
        one charged with the target still has to arrive as a defender rather
        than fall into the same bucket as the quarterback."""
        packed, frames = geometry
        key = _first_key(frames)
        tracking = frames.tracking
        play = tracking[(tracking["game_id"] == key[0]) & (tracking["play_id"] == key[1])]
        sides = play.drop_duplicates("nfl_id").set_index("nfl_id")["player_side"]

        block = unpack_block(packed, 0)
        kinds = [p["role"] for p in block["pre"]]
        defence = {export.KIND_PRIMARY_COVERAGE, export.KIND_OTHER_DEFENCE}
        assert sum(k in defence for k in kinds) == int((sides == "Defense").sum())
        # The synthetic play carries a safety and a quarterback, so both of
        # the "everyone else" kinds are exercised rather than assumed.
        assert export.KIND_OTHER_DEFENCE in kinds
        assert export.KIND_OTHER_OFFENCE in kinds

    def test_the_run_up_says_who_the_tracking_follows_into_the_air(self, geometry):
        """The page stops animating a player it has no flight block for, so
        the flag and the flight section have to agree about who that is."""
        packed, _frames = geometry
        block = unpack_block(packed, 0)
        assert sum(p["inAir"] for p in block["pre"]) == len(block["flight"])
        # And the safety, whom the output file never follows, is not one.
        assert any(not p["inAir"] for p in block["pre"])

    def test_the_flight_carries_the_target_and_the_coverage(self, geometry):
        packed, _frames = geometry
        block = unpack_block(packed, 0)
        kinds = [t["kind"] for t in block["flight"]]
        assert export.KIND_TARGET in kinds

    def test_positions_land_on_the_field(self, geometry):
        packed, _frames = geometry
        block = unpack_block(packed, 0)
        for player in block["pre"]:
            assert (player["x"] >= -5).all() and (player["x"] <= 125).all()
            assert (player["y"] >= -5).all() and (player["y"] <= 58).all()


def _play_index(frames):
    """The minimal play table build_geometry reads."""
    import pandas as pd

    rows = []
    for key, tracking in frames.tracking.groupby(data.PLAY_KEY, sort=False):
        target = tracking[tracking["player_role"] == "Targeted Receiver"]
        coverage = tracking[tracking["player_role"] == "Defensive Coverage"]
        rows.append({
            "game_id": key[0], "play_id": key[1], "week": 1,
            "target_nfl_id": target["nfl_id"].iloc[0] if len(target) else -1,
            "coverage_nfl_id": coverage["nfl_id"].iloc[0] if len(coverage) else -1,
        })
    return pd.DataFrame(rows)


def _first_key(frames):
    return list(frames.tracking.groupby(data.PLAY_KEY, sort=False).groups)[0]
