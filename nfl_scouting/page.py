"""Render the dashboard page: payload, template and script into one file.

The page is self-contained by design - a season of tracking, the tables, the
model's diagnostics and every line of its behaviour in a single HTML file
that needs no server and no network. That only works if something assembles
it, and this is that something.

The three parts live next to each other in ``dashboard/``:

======================  ==================================================
``template.html``       the page's markup and styling
``app.js``              everything it does: filters, tables, the field
``nfl-scouting.html``   the built result, which is what gets published
======================  ==================================================

Field geometry is injected rather than written into the script, because the
drawing and the analysis have to agree about where the field ends. The end
zone the page shades and the end zone :mod:`nfl_scouting.config` measures
against are the same ten yards, so changing one changes both.

The payload itself is not unpacked here. It arrives from
:func:`nfl_scouting.export.build_payload` with its numeric columns already
packed as base64 typed arrays and its repeated text interned, and this only
checks that the shape is the one the page reads - so a truncated or stale
export fails at the build rather than as a blank panel in somebody's browser.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import END_ZONE_DEPTH, FIELD_LENGTH, FIELD_WIDTH, REPO_ROOT

__all__ = [
    "DASHBOARD_DIR",
    "REQUIRED_KEYS",
    "field_geometry",
    "render",
    "validate",
]

DASHBOARD_DIR = REPO_ROOT / "dashboard"
TEMPLATE = DASHBOARD_DIR / "template.html"
SCRIPT = DASHBOARD_DIR / "app.js"
DEFAULT_OUT = DASHBOARD_DIR / "nfl-scouting.html"

DATA_TOKEN = "/*__DATA__*/"
SCRIPT_TOKEN = "/*__SCRIPT__*/"
FIELD_TOKEN = "/*__FIELD__*/"

# Every top-level key the page reads. Listed rather than inferred so that a
# payload built by an older version of the exporter fails loudly.
REQUIRED_KEYS = (
    "meta", "plays", "receivers", "defenders",
    "routeCoverage", "coverageType", "model", "strings",
)
REQUIRED_META = (
    "season", "weeks", "nPlays", "nGames", "nReceivers", "nDefenders",
    "completionRate", "meanSeparationThrow", "meanSeparationArrival",
    "openThreshold",
)


def field_geometry() -> dict:
    """The field the page draws, in the units the tracking uses.

    ``endZone`` is the depth outside each goal line, so the goal lines
    themselves sit at ``endZone`` and ``length - endZone``.
    """
    return {
        "length": FIELD_LENGTH,
        "width": FIELD_WIDTH,
        "endZone": END_ZONE_DEPTH,
    }


def validate(payload: dict) -> None:
    """Fail on a payload the page cannot draw, naming what is missing."""
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise SystemExit(f"payload is missing top-level keys: {missing}")

    missing_meta = [k for k in REQUIRED_META if k not in payload["meta"]]
    if missing_meta:
        raise SystemExit(f"payload meta is missing: {missing_meta}")

    plays = payload["plays"]
    for key in ("n", "int16", "f32"):
        if key not in plays:
            raise SystemExit(f"payload plays block is missing '{key}'")
    if plays["n"] <= 0:
        raise SystemExit("payload contains no plays")

    # The play explorer reads these two per play: the down and distance label
    # on the line to gain, and the line itself.
    for column in ("down", "toGo"):
        if column not in plays["int16"]:
            raise SystemExit(f"payload plays block is missing int16 '{column}'")
    if "losX" not in plays["f32"]:
        raise SystemExit("payload plays block is missing f32 'losX'")

    for key in ("scores", "calibration", "importance", "baselines"):
        if key not in payload["model"]:
            raise SystemExit(f"payload model block is missing '{key}'")

    geometry = payload.get("geometry")
    if geometry is not None:
        if len(geometry["offsets"]) != plays["n"]:
            raise SystemExit(
                f"geometry has {len(geometry['offsets'])} offsets for "
                f"{plays['n']} plays")
        drawable = sum(1 for o in geometry["offsets"] if o >= 0)
        if not drawable:
            raise SystemExit("geometry is present but no play can be drawn from it")


def render(payload: dict,
           template: Path | None = None,
           script: Path | None = None) -> str:
    """The finished page, as one string."""
    validate(payload)

    template_path = template or TEMPLATE
    script_path = script or SCRIPT
    for path, name in ((template_path, "template"), (script_path, "script")):
        if not path.exists():
            raise SystemExit(f"no dashboard {name} at {path}")

    markup = template_path.read_text()
    for token, name in ((DATA_TOKEN, "data"), (SCRIPT_TOKEN, "script"),
                        (FIELD_TOKEN, "field geometry")):
        if token not in markup:
            raise SystemExit(f"template has no {name} placeholder ({token})")

    # The payload goes into a JSON script tag, so the only thing that can
    # break out of it is the closing tag itself.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    field = json.dumps(field_geometry(), separators=(",", ":"))

    return (markup
            .replace(DATA_TOKEN, blob)
            .replace(FIELD_TOKEN, field)
            .replace(SCRIPT_TOKEN, script_path.read_text()))


def build(data_path: Path, out_path: Path | None = None) -> Path:
    """Render a payload on disk into the dashboard page."""
    payload = json.loads(Path(data_path).read_text())
    out = Path(out_path or DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload))
    return out
