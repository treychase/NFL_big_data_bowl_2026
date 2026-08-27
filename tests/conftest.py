"""Synthetic plays built to the competition's schema.

The real files are 800 MB and are not in the test path, so the fixtures here
build small ones with the same columns, dtypes and conventions: 10 Hz
frames, `dir` in degrees clockwise from +y, an input file that stops at the
release and an output file that starts a tenth of a second later.

Building plays from an explicit path per player means a test can state the
answer it expects - a receiver who runs a straight line at 6 yards a second
has a known speed, a known heading and zero change of direction - instead of
asserting whatever the code happened to produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_scouting.config import DT, ROLE_COVERAGE, ROLE_OTHER_ROUTE, ROLE_PASSER, ROLE_TARGET

GAME_ID = 2023090700
PLAY_ID = 101


def straight_path(x0, y0, heading_deg, speed, n_frames, dt=DT):
    """A constant-velocity path, in the tracking file's angle convention."""
    t = np.arange(n_frames) * dt
    radians = np.radians(heading_deg)
    return x0 + speed * t * np.sin(radians), y0 + speed * t * np.cos(radians)


def make_player(nfl_id, name, position, side, role, x, y, *, predict=False,
                speed=None, direction=90.0, orientation=None, weight=200,
                height="6-1"):
    """One player's rows, as the input file would carry them."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if speed is None:
        speed = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y)) / DT])
    return pd.DataFrame({
        "nfl_id": nfl_id,
        "frame_id": np.arange(1, n + 1),
        "player_to_predict": predict,
        "player_name": name,
        "player_height": height,
        "player_weight": weight,
        "player_birth_date": "1997-01-01",
        "player_position": position,
        "player_side": side,
        "player_role": role,
        "x": x,
        "y": y,
        "s": speed,
        "a": 0.0,
        "dir": direction,
        "o": orientation if orientation is not None else direction,
    })


def assemble_play(players, *, ball_land, air_paths, play_direction="right",
                  los_x=40.0, game_id=GAME_ID, play_id=PLAY_ID):
    """Stitch player frames into an input/output pair for one play.

    `air_paths` maps nfl_id to the (x, y) the player follows while the ball
    is in the air, which is what the output file holds.
    """
    tracking = pd.concat(players, ignore_index=True)
    n_air = max(len(px) for px, _ in air_paths.values())
    tracking = tracking.assign(
        game_id=game_id, play_id=play_id, play_direction=play_direction,
        absolute_yardline_number=los_x,
        num_frames_output=n_air,
        ball_land_x=ball_land[0], ball_land_y=ball_land[1],
    )
    rows = []
    for nfl_id, (px, py) in air_paths.items():
        rows.append(pd.DataFrame({
            "game_id": game_id, "play_id": play_id, "nfl_id": nfl_id,
            "frame_id": np.arange(1, len(px) + 1),
            "x": np.asarray(px, float), "y": np.asarray(py, float),
        }))
    ball_air = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["game_id", "play_id", "nfl_id", "frame_id", "x", "y"])
    return tracking, ball_air


@pytest.fixture
def simple_play():
    """A receiver breaking open, a corner trailing, and a catchable ball.

    The receiver runs a straight go route at 6 yd/s and keeps running to the
    ball. The corner starts 2 yards behind him and runs the same line a
    little slower, so separation grows during the flight - a fact the tests
    assert against rather than infer.
    """
    n_pre, n_air = 20, 10
    rx, ry = straight_path(40.0, 20.0, 90.0, 6.0, n_pre)
    cx, cy = straight_path(38.0, 20.0, 90.0, 6.0, n_pre)
    px, py = straight_path(33.0, 26.5, 90.0, 0.0, n_pre)
    ox, oy = straight_path(40.0, 45.0, 90.0, 5.0, n_pre)

    # Flight: receiver keeps 6 yd/s, corner drops to 5 yd/s and falls behind.
    arx, ary = straight_path(rx[-1] + 6.0 * DT, ry[-1], 90.0, 6.0, n_air)
    acx, acy = straight_path(cx[-1] + 5.0 * DT, cy[-1], 90.0, 5.0, n_air)
    ball_land = (float(arx[-1]), float(ary[-1]))

    players = [
        make_player(1001, "Wide Out", "WR", "Offense", ROLE_TARGET, rx, ry,
                    predict=True, direction=90.0),
        make_player(2001, "Corner Back", "CB", "Defense", ROLE_COVERAGE, cx, cy,
                    predict=True, direction=90.0),
        make_player(3001, "Quarter Back", "QB", "Offense", ROLE_PASSER, px, py,
                    direction=90.0),
        make_player(1002, "Slot Man", "WR", "Offense", ROLE_OTHER_ROUTE, ox, oy,
                    direction=90.0),
        make_player(2002, "Safety Man", "FS", "Defense", ROLE_COVERAGE,
                    *straight_path(60.0, 26.0, 270.0, 3.0, n_pre), direction=270.0),
    ]
    return assemble_play(players, ball_land=ball_land,
                         air_paths={1001: (arx, ary), 2001: (acx, acy)})


@pytest.fixture
def play_table():
    """A small scored play table, shaped like what the pipeline produces."""
    rng = np.random.default_rng(26)
    n = 240
    separation = rng.gamma(3.0, 1.4, n)
    air_yards = rng.normal(9.0, 8.0, n)
    # Completion falls off with air yards and rises with separation, so a
    # model has something real to find and a rate table has real variation.
    logit = 1.1 + 0.28 * separation - 0.055 * air_yards
    complete = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame({
        "game_id": rng.integers(1, 13, n),
        "play_id": np.arange(n),
        "week": rng.integers(1, 5, n),
        "target_nfl_id": rng.integers(1, 7, n),
        "target_name": "Receiver",
        "target_position": "WR",
        "coverage_nfl_id": rng.integers(50, 56, n).astype(float),
        "coverage_name": "Defender",
        "coverage_position": "CB",
        "is_complete": complete,
        "is_touchdown": complete & (rng.random(n) < 0.07),
        "is_interception": (~complete) & (rng.random(n) < 0.09),
        "yards_gained": np.where(complete, np.maximum(air_yards, 0) + rng.gamma(2, 2, n), 0),
        "separation_at_throw_yd": separation,
        "separation_at_arrival_yd": np.maximum(separation - rng.normal(0.9, 1.2, n), 0.05),
        "air_yards_yd": air_yards,
        "route_of_targeted_receiver": rng.choice(["HITCH", "OUT", "GO", "SLANT"], n),
        "team_coverage_man_zone": rng.choice(["MAN_COVERAGE", "ZONE_COVERAGE"], n),
        "rec_cod_total_deg": rng.gamma(3, 10, n),
        "rec_accel_burst_yps2": rng.normal(2.5, 1.0, n),
        "cov_cod_total_deg": rng.gamma(4, 10, n),
        "cov_pursuit_efficiency": rng.uniform(-0.5, 1.0, n),
        "coverage_tracked_in_air": True,
    }).assign(
        separation_change_yd=lambda d: d.separation_at_arrival_yd - d.separation_at_throw_yd)


def write_synthetic_season(root, n_games=8, plays_per_game=14, seed=26):
    """A miniature season on disk, in the competition's own file layout.

    Enough games for a grouped split to have something to group, and enough
    variation in separation and air yards that the model has a real signal to
    find rather than noise to memorise.
    """
    rng = np.random.default_rng(seed)
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)

    routes = ["HITCH", "OUT", "GO", "SLANT", "CROSS"]
    play_rows = []
    by_week: dict[int, list] = {}

    for week in (1, 2):
        by_week[week] = ([], [])
        for game in range(n_games):
            game_id = 2023090700 + week * 100 + game
            for play in range(plays_per_game):
                play_id = 100 + play
                n_pre = int(rng.integers(14, 30))
                n_air = int(rng.integers(5, 16))

                los = float(rng.uniform(25, 75))
                lane = float(rng.uniform(6, 46))
                speed = float(rng.uniform(4.5, 8.0))
                cushion = float(rng.uniform(0.6, 7.0))

                rx, ry = straight_path(los, lane, 90.0, speed, n_pre)
                cx, cy = straight_path(los - cushion, lane, 90.0, speed * 0.95, n_pre)
                qx, qy = straight_path(los - 7, 26.5, 90.0, 0.0, n_pre)

                arx, ary = straight_path(rx[-1] + speed * DT, ry[-1], 90.0, speed, n_air)
                acx, acy = straight_path(
                    cx[-1] + speed * DT, cy[-1], 90.0, speed * 0.95, n_air)
                land = (float(arx[-1]), float(ary[-1]))

                players = [
                    make_player(1000 + play, f"Receiver {play}", "WR", "Offense",
                                ROLE_TARGET, rx, ry, predict=True),
                    make_player(2000 + play, f"Corner {play}", "CB", "Defense",
                                ROLE_COVERAGE, cx, cy, predict=True),
                    make_player(3000, "Passer", "QB", "Offense", ROLE_PASSER, qx, qy),
                    make_player(2500, "Safety", "FS", "Defense", ROLE_COVERAGE,
                                *straight_path(los + 15, 26.5, 270.0, 3.0, n_pre)),
                ]
                tracking, air = assemble_play(
                    players, ball_land=land, air_paths={1000 + play: (arx, ary),
                                                        2000 + play: (acx, acy)},
                    los_x=los, game_id=game_id, play_id=play_id)
                by_week[week][0].append(tracking)
                by_week[week][1].append(air)

                air_yards = land[0] - los
                complete = rng.random() < 1 / (1 + np.exp(-(0.4 * cushion - 0.05 * air_yards)))
                intercepted = (not complete) and rng.random() < 0.08
                touchdown = complete and rng.random() < 0.08
                play_rows.append({
                    "game_id": game_id, "play_id": play_id, "season": 2023, "week": week,
                    "game_date": "09/07/2023", "quarter": int(rng.integers(1, 5)),
                    "game_clock": "10:00", "down": int(rng.integers(1, 5)),
                    "yards_to_go": int(rng.integers(1, 16)),
                    "possession_team": "DET", "defensive_team": "KC",
                    "play_description": ("Pass complete. TOUCHDOWN." if touchdown
                                         else "INTERCEPTED." if intercepted else "Pass."),
                    "pass_result": "C" if complete else ("IN" if intercepted else "I"),
                    "yards_gained": float(max(air_yards, 0)) if complete else 0.0,
                    "play_nullified_by_penalty": "N",
                    "route_of_targeted_receiver": rng.choice(routes),
                    "team_coverage_man_zone": rng.choice(["MAN_COVERAGE", "ZONE_COVERAGE"]),
                    "team_coverage_type": rng.choice(["COVER_1_MAN", "COVER_3_ZONE"]),
                    "offense_formation": "SHOTGUN", "receiver_alignment": "3x1",
                    "dropback_type": "TRADITIONAL", "pass_location_type": "INSIDE_BOX",
                    "play_action": bool(rng.random() < 0.25),
                    "defenders_in_the_box": int(rng.integers(4, 8)),
                    "expected_points_added": float(rng.normal(0, 1)),
                    "absolute_yardline_number": los,
                })

        tracking_frames, air_frames = by_week[week]
        pd.concat(tracking_frames, ignore_index=True).to_csv(
            train / f"input_2023_w{week:02d}.csv", index=False)
        pd.concat(air_frames, ignore_index=True).to_csv(
            train / f"output_2023_w{week:02d}.csv", index=False)

    pd.DataFrame(play_rows).to_csv(root / "supplementary_data.csv", index=False)
    return root


@pytest.fixture(scope="session")
def synthetic_season(tmp_path_factory):
    return write_synthetic_season(tmp_path_factory.mktemp("season"))
