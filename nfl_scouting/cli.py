"""Command line entry point: build the artifacts, or the dashboard payload.

    python -m nfl_scouting build                 # every artifact, cached
    python -m nfl_scouting build --refresh       # re-extract the tracking
    python -m nfl_scouting build --weeks 1 2 3   # a slice, for a quick look
    python -m nfl_scouting export --out data.json
    python -m nfl_scouting page                  # render the dashboard page
    python -m nfl_scouting report                # what the last build found
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import export, page, pipeline
from .config import N_CV_FOLDS, RANDOM_SEED, SEASON, artifact_dir


def _add_common(parser: argparse.ArgumentParser) -> None:
    # Accepted after the subcommand as well as before it, because
    # `build -v` is what anyone actually types.
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="competition data directory (default: the repo's)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where artifacts are written (default: ./artifacts)")
    parser.add_argument("--weeks", type=int, nargs="+", default=None,
                        help="weeks to include (default: every week on disk)")
    parser.add_argument("--season", type=int, default=SEASON)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nfl_scouting", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="extract features, fit the model, write artifacts")
    _add_common(build)
    build.add_argument("--refresh", action="store_true",
                       help="re-extract the tracking instead of reusing the cache")
    build.add_argument("--folds", type=int, default=N_CV_FOLDS)
    build.add_argument("--seed", type=int, default=RANDOM_SEED)

    exporter = sub.add_parser("export", help="write the dashboard payload")
    _add_common(exporter)
    exporter.add_argument("--out", type=Path, default=None,
                          help="payload path (default: <out-dir>/dashboard_data.json)")
    exporter.add_argument("--no-geometry", action="store_true",
                          help="skip per-play tracking, for a much smaller file")

    pager = sub.add_parser(
        "page", help="render the self-contained dashboard page")
    _add_common(pager)
    pager.add_argument("--data", type=Path, default=None,
                       help="payload to render (default: export one from the "
                            "last build)")
    pager.add_argument("--out", type=Path, default=None,
                       help=f"page path (default: {page.DEFAULT_OUT})")
    pager.add_argument("--no-geometry", action="store_true",
                       help="skip per-play tracking, for a much smaller page")

    report = sub.add_parser("report", help="print the last build's headline numbers")
    _add_common(report)
    return parser


def _report(out_dir: Path) -> int:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        print(f"no build found at {out_dir}; run `build` first", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text())
    model = summary["model"]

    # The competition is named for 2026; the tracking in it is the 2023 season.
    print(f"NFL Big Data Bowl 2026 - {summary['season']} season - scouting build")
    print(f"  {summary['n_plays']:,} targets over {summary['n_games']} games, "
          f"weeks {min(summary['weeks'])}-{max(summary['weeks'])}")
    print(f"  {summary['n_receivers']} receivers targeted, "
          f"{summary['n_defenders']} defenders charged with coverage")
    print(f"  completion rate {summary['completion_rate']:.3f}")
    print(f"  separation {summary['mean_separation_at_throw_yd']:.2f} yd at the throw, "
          f"{summary['mean_separation_at_arrival_yd']:.2f} yd on arrival")
    print(f"\n  catch probability, out of fold, grouped by {model['grouped_by']}:")
    print(f"    AUC {model['scores']['roc_auc']:.4f}   "
          f"Brier {model['scores']['brier']:.4f}   "
          f"log loss {model['scores']['log_loss']:.4f}")
    for name, scores in model["baselines"].items():
        print(f"    vs {name:<10} AUC {scores['roc_auc']:.4f}   "
              f"Brier {scores['brier']:.4f}   log loss {scores['log_loss']:.4f}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    out_dir = Path(args.out_dir or artifact_dir())

    if args.command == "report":
        return _report(out_dir)

    if args.command == "build":
        artifacts = pipeline.run(
            out_dir=out_dir, weeks=args.weeks, season=args.season,
            root=args.data_dir, refresh=args.refresh,
            n_folds=args.folds, random_state=args.seed)
        print(f"wrote artifacts to {out_dir}")
        return _report(out_dir)

    if args.command == "page":
        # A payload on disk is the common case when only the front end
        # changed; without one, export straight from the last build rather
        # than making the caller write a temporary file just to read it back.
        if args.data:
            payload = json.loads(Path(args.data).read_text())
        else:
            payload = export.build_payload(
                _load_artifacts(out_dir), weeks=args.weeks, season=args.season,
                root=args.data_dir, with_geometry=not args.no_geometry)
        out = Path(args.out or page.DEFAULT_OUT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page.render(payload))
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), "
              f"{payload['meta']['nPlays']:,} targets, "
              f"{payload['meta']['nGames']} games, "
              f"{payload['meta']['season']} weeks "
              f"{payload['meta']['weeks'][0]}-{payload['meta']['weeks'][-1]}")
        return 0

    if args.command == "export":
        artifacts = _load_artifacts(out_dir)
        payload = export.build_payload(
            artifacts, weeks=args.weeks, season=args.season, root=args.data_dir,
            with_geometry=not args.no_geometry)
        path = export.write_payload(payload, args.out or out_dir / "dashboard_data.json")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
        return 0

    return 1


def _load_artifacts(out_dir: Path) -> pipeline.Artifacts:
    """Rebuild the Artifacts bundle from a finished build, without refitting."""
    required = out_dir / "summary.json"
    if not required.exists():
        raise SystemExit(f"no build found at {out_dir}; run `build` first")
    read = lambda name: pd.read_csv(out_dir / f"{name}.csv")
    return pipeline.Artifacts(
        plays=pd.read_parquet(out_dir / pipeline.PLAY_TABLE_NAME),
        receivers=read("receiver_scouting"),
        defenders=read("defender_coverage"),
        route_coverage=read("route_by_coverage"),
        coverage_type=read("route_by_coverage_type"),
        receiver_routes=read("receiver_route_splits"),
        importance=read("model_feature_importance"),
        cv=None,
        summary=json.loads(required.read_text()),
    )


if __name__ == "__main__":
    raise SystemExit(main())
