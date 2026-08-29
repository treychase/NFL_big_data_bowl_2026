# NFL Big Data Bowl 2026 — receiver and coverage scouting

Separation, coverage quality and catch probability from the Big Data Bowl
2026 tracking release: every targeted pass of the 2023 NFL season, measured
at the release and again through the flight.

The competition's data is unusual in one specific way, and this project is
built around it. Alongside the pre-pass tracking it ships `ball_land_x` and
`ball_land_y` — where the ball came down — and the positions of the players
near it while the ball was in the air. That makes the flight the only window
in this kind of data where a receiver and a defender can be watched reacting
to the same known destination, and compared on what each did about it.

## What it produces

**Separation.** Yards to the nearest defender when the ball is released, and
again when it arrives. Reported as a mean and as an open rate — the share of
targets at three or more yards, which is where Next Gen Stats draws the line,
so the numbers are comparable to the published ones.

**Passer rating allowed.** The standard NFL formula applied to the throws
each defender was charged with covering. Coverage is charged to the defender
nearest the receiver at the release.

**Route by coverage.** Every route type against man and zone, and against
each coverage shell: catch rate, separation, passer rating, and what the
receiver and defender did in the air.

**Ball-in-air kinematics.** Acceleration, top speed, best half-second burst,
and total change of direction, for the receiver and for the covering
defender — plus, for the defender, how much of his angle to the ball he took
out during the flight, how efficiently he closed on the landing spot, and how
fast he shut the gap.

**Catch probability.** A gradient boosted model over release-time geometry.
Out of fold, grouped by game: **AUC 0.802, Brier 0.157**, against 0.484 and
0.214 for the league base rate and 0.782 / 0.167 for logistic regression.
Calibration is close to the diagonal in every decile, which is what makes
catch rate over expected worth reading.

**A dashboard.** Every play in the season, filterable and drawn on the field
with the flight animated, plus the scouting tables and the model's own
diagnostics. It is one self-contained HTML file that needs no server and no
network; the template, the script and the build that assembles them all live
in `dashboard/`, and it is published to
[treychase.github.io](https://treychase.github.io/projects/nfl-scouting.html)
by copying the built file across.

## Running it

```bash
pip install pandas numpy scipy scikit-learn pyarrow pytest

python -m nfl_scouting build            # extract, fit, score, write artifacts
python -m nfl_scouting report           # the headline numbers
python -m nfl_scouting export --out dashboard_data.json
python -m nfl_scouting page              # render dashboard/nfl-scouting.html
```

`build` caches the extracted play table to `artifacts/play_features.parquet`;
pass `--refresh` to re-extract. `--weeks 1 2 3` runs a slice.

Artifacts land in `artifacts/`: the per-play feature table, the receiver and
defender scouting tables, the route-by-coverage splits, the model's fold
scores, calibration and feature importance, and a `summary.json`.

## Tests

```bash
python -m pytest
```

153 tests. The kinematics are checked against motion with a closed-form
answer — a track at constant speed must return zero acceleration at every
frame including the first and last, a 90-degree cut must measure 90 degrees
at any realistic break sharpness — and the passer rating implementation is
checked against real published season lines.

## Layout

```
nfl_scouting/
  config.py      paths, field geometry, and every threshold that is a judgement call
  data.py        reading the three competition files
  geometry.py    angles and distances in the tracking file's frame
  kinematics.py  velocity, acceleration and change of direction from positions
  features.py    one row per pass attempt: the release, the flight, the arrival
  metrics.py     the scouting tables and the passer rating formula
  model.py       catch probability, its validation, and the leakage guard
  pipeline.py    build everything, write everything
  export.py      pack the artifacts for the dashboard
  page.py        assemble the dashboard page from the payload and dashboard/
  cli.py         python -m nfl_scouting
```

## Decisions worth knowing about

**The model is not allowed to see the flight.** It would be trivial to reach
0.95 AUC by feeding it the receiver's distance to the ball at the moment the
ball arrives, and the result would say nothing. The cut is the release: where
everyone was and how they were moving when the ball left the hand, plus where
it was going and how long it would be in the air. `model.assert_no_leakage`
enumerates the post-release columns and the test suite fails the build if one
reaches the feature list, because this is a mistake that does not show up in
the metrics — it only makes them better.

Ball destination and flight time are known at release to the analyst but not
to the players, which makes this completion probability conditional on the
throw — the same framing the league's own model uses — rather than a model of
the quarterback's decision.

**Validation is grouped by game.** Plays from one game share a quarterback, a
secondary, a stadium and a weather. A random split lets the model recognise
the game rather than the throw.

**Coverage assignment is a proxy.** The data does not say who was covering
whom, so the defender nearest the receiver at the release is charged with the
target — the same proxy Next Gen Stats uses for separation. On a genuine
bracket it names one of the two defenders; on a busted zone exchange it can
name the man who was closest rather than the man who was beaten. The column
is `targets_covered`, not `targets`, to keep that visible.

**Separation change is measured over one set of players.** Only some
defenders are tracked through the flight — about eleven plays in a hundred do
not include the man who was nearest at the release. Subtracting arrival
separation, which can only be measured against tracked defenders, from the
headline release separation, which is measured against all eleven, produced
receivers who gained thirty yards of separation in a second. The change is
now taken between two measurements over the same tracked set;
`separation_at_throw_yd` remains the all-defenders number for comparability,
and `separation_at_throw_tracked_yd` is the basis for the difference.

**Separation is not comparable across positions.** A back released into the
flat is open by ten yards because nobody is covering him there. The
separation leaderboard is running backs from top to bottom, and it should be
read one position at a time. Catch rate over expected is adjusted for the
throw and does compare.

**A few landing spots are dropped.** On a handful of plays the recorded
landing spot is further from the targeted receiver than any human could
cover in the ball's flight, which means it is not describing that target.
Those are dropped rather than scored as impossible throws that were somehow
completed. Badly overthrown balls, where the receiver genuinely could not get
there, are kept — those are real football and belong in the completion rate.

**Derivatives are fitted, not differenced.** The flight frames carry position
only, so speed, acceleration and heading are differentiated from x and y. A
local quadratic fit is used rather than smoothing-then-differencing, which
drags the endpoints inward and invents acceleration at exactly the two frames
that matter, the throw and the catch. The fit is quadratic rather than cubic
because a cubic rings at a sharp break — it inflates a 90-degree cut to 101 —
while measuring acceleration no better.

## Data

The Big Data Bowl 2026 release, under
`data/nfl-big-data-bowl-2026/`: per-week `input_` files of pre-pass tracking,
`output_` files of ball-in-air positions for the players near the play, and
`supplementary_data.csv` with the route, the coverage and the outcome.
14,107 targeted passes across 272 games, weeks 1 to 18 of the 2023 season.
