"""The dashboard page: what has to be in a payload, and what comes out.

The page is assembled rather than written, so the failures worth catching
here are the assembly ones - a payload the page cannot draw, a template
missing a placeholder, geometry that disagrees with the analysis - because
every one of them shows up in a browser as a blank panel rather than as an
error anybody sees.
"""

import json
import re

import pytest

from nfl_scouting import export, page
from nfl_scouting.config import END_ZONE_DEPTH, FIELD_LENGTH, FIELD_WIDTH


def minimal_payload():
    """The smallest payload :func:`page.validate` accepts."""
    return {
        "meta": {
            "season": 2023, "weeks": [1], "nPlays": 1, "nGames": 1,
            "nReceivers": 1, "nDefenders": 1, "completionRate": 0.5,
            "meanSeparationThrow": 4.0, "meanSeparationArrival": 3.0,
            "openThreshold": 3.0,
        },
        "plays": {
            "n": 1,
            "int16": {"down": "", "toGo": ""},
            "f32": {"losX": ""},
        },
        "receivers": [], "defenders": [], "routeCoverage": [],
        "coverageType": [], "strings": [],
        "model": {"scores": {}, "calibration": [], "importance": [],
                  "baselines": {}},
    }


class TestFieldGeometry:
    def test_it_is_the_field_the_analysis_measures(self):
        """The page and the metrics have to agree about where the field ends."""
        geometry = page.field_geometry()
        assert geometry["length"] == FIELD_LENGTH
        assert geometry["width"] == FIELD_WIDTH
        assert geometry["endZone"] == END_ZONE_DEPTH

    def test_the_goal_lines_fall_inside_the_field(self):
        geometry = page.field_geometry()
        near = geometry["endZone"]
        far = geometry["length"] - geometry["endZone"]
        assert 0 < near < far < geometry["length"]
        # A hundred yards of playing field between them, by definition.
        assert far - near == pytest.approx(100.0)

    def test_it_survives_a_json_round_trip(self):
        """It is injected as JSON, so it has to be JSON-serialisable."""
        assert json.loads(json.dumps(page.field_geometry())) == page.field_geometry()


class TestValidate:
    def test_a_complete_payload_passes(self):
        page.validate(minimal_payload())

    @pytest.mark.parametrize("key", page.REQUIRED_KEYS)
    def test_a_missing_top_level_key_is_named(self, key):
        payload = minimal_payload()
        del payload[key]
        with pytest.raises(SystemExit, match=key):
            page.validate(payload)

    def test_a_missing_meta_field_is_named(self):
        payload = minimal_payload()
        del payload["meta"]["openThreshold"]
        with pytest.raises(SystemExit, match="openThreshold"):
            page.validate(payload)

    def test_a_payload_with_no_plays_is_rejected(self):
        payload = minimal_payload()
        payload["plays"]["n"] = 0
        with pytest.raises(SystemExit, match="no plays"):
            page.validate(payload)

    @pytest.mark.parametrize("column", ["down", "toGo"])
    def test_the_line_to_gain_columns_are_required(self, column):
        """Without these the play explorer cannot draw the line to gain."""
        payload = minimal_payload()
        del payload["plays"]["int16"][column]
        with pytest.raises(SystemExit, match=column):
            page.validate(payload)

    def test_the_scrimmage_column_is_required(self):
        payload = minimal_payload()
        del payload["plays"]["f32"]["losX"]
        with pytest.raises(SystemExit, match="losX"):
            page.validate(payload)

    def test_geometry_offsets_must_cover_every_play(self):
        payload = minimal_payload()
        payload["geometry"] = {"offsets": [0, 1], "data": "", "scale": 10}
        with pytest.raises(SystemExit, match="2 offsets for 1 plays"):
            page.validate(payload)

    def test_geometry_nobody_can_be_drawn_from_is_rejected(self):
        payload = minimal_payload()
        payload["geometry"] = {"offsets": [-1], "data": "", "scale": 10}
        with pytest.raises(SystemExit, match="no play can be drawn"):
            page.validate(payload)

    def test_a_payload_without_geometry_is_fine(self):
        """--no-geometry is a supported export, not a broken one."""
        payload = minimal_payload()
        payload.pop("geometry", None)
        page.validate(payload)


class TestRender:
    def test_the_page_carries_all_three_parts(self, tmp_path):
        template = tmp_path / "t.html"
        template.write_text(
            '<div>/*__DATA__*/</div><div>/*__FIELD__*/</div>'
            '<script>/*__SCRIPT__*/</script>')
        script = tmp_path / "a.js"
        script.write_text("var answer = 42;")

        html = page.render(minimal_payload(), template=template, script=script)
        assert "var answer = 42;" in html
        assert '"season":2023' in html
        assert str(END_ZONE_DEPTH) in html
        # Nothing may be left unreplaced, or the page ships with a comment
        # where its data should be.
        for token in (page.DATA_TOKEN, page.SCRIPT_TOKEN, page.FIELD_TOKEN):
            assert token not in html

    def test_a_closing_tag_in_the_payload_cannot_break_out(self, tmp_path):
        """The payload sits in a script tag; only </ can end it early."""
        template = tmp_path / "t.html"
        template.write_text("/*__DATA__*/ /*__FIELD__*/ /*__SCRIPT__*/")
        script = tmp_path / "a.js"
        script.write_text("")

        payload = minimal_payload()
        payload["strings"] = ["</script><script>alert(1)</script>"]
        html = page.render(payload, template=template, script=script)
        assert "</script>" not in html
        assert "<\\/script>" in html

    @pytest.mark.parametrize("token", ["/*__DATA__*/", "/*__FIELD__*/",
                                       "/*__SCRIPT__*/"])
    def test_a_template_missing_a_placeholder_is_named(self, tmp_path, token):
        every = ["/*__DATA__*/", "/*__FIELD__*/", "/*__SCRIPT__*/"]
        template = tmp_path / "t.html"
        template.write_text(" ".join(t for t in every if t != token))
        script = tmp_path / "a.js"
        script.write_text("")
        with pytest.raises(SystemExit, match="placeholder"):
            page.render(minimal_payload(), template=template, script=script)

    def test_a_missing_template_is_named(self, tmp_path):
        script = tmp_path / "a.js"
        script.write_text("")
        with pytest.raises(SystemExit, match="template"):
            page.render(minimal_payload(), template=tmp_path / "gone.html",
                        script=script)

    def test_an_invalid_payload_never_reaches_the_template(self, tmp_path):
        template = tmp_path / "t.html"
        template.write_text("/*__DATA__*/ /*__FIELD__*/ /*__SCRIPT__*/")
        script = tmp_path / "a.js"
        script.write_text("")
        payload = minimal_payload()
        del payload["model"]
        with pytest.raises(SystemExit, match="model"):
            page.render(payload, template=template, script=script)


class TestShippedFrontEnd:
    """The real template and script, which is what actually gets published."""

    def test_they_are_in_the_repo(self):
        assert page.TEMPLATE.exists(), f"no template at {page.TEMPLATE}"
        assert page.SCRIPT.exists(), f"no script at {page.SCRIPT}"

    def test_the_template_has_every_placeholder(self):
        markup = page.TEMPLATE.read_text()
        for token in (page.DATA_TOKEN, page.SCRIPT_TOKEN, page.FIELD_TOKEN):
            assert token in markup, f"template is missing {token}"

    def test_the_script_reads_the_geometry_rather_than_repeating_it(self):
        """The point of injecting it is that the numbers appear once."""
        source = page.SCRIPT.read_text()
        assert "field-geometry" in source
        assert "var FIELD_LENGTH = 120" not in source

    def test_the_script_speaks_the_kinds_the_exporter_packs(self):
        """The blob carries a kind per player and nothing that says what one
        means, so the drawing's idea of the numbering has to be checked
        against the packing's rather than assumed to have kept up with it."""
        source = page.SCRIPT.read_text()
        for name, value in (("KIND_TARGET", export.KIND_TARGET),
                            ("KIND_COVERAGE", export.KIND_PRIMARY_COVERAGE),
                            ("KIND_OFFENCE", export.KIND_OTHER_OFFENCE),
                            ("KIND_DEFENCE", export.KIND_OTHER_DEFENCE)):
            assert f"{name} = {value}" in source, f"{name} is not {value} in app.js"

    def test_every_colour_the_script_asks_for_is_defined(self):
        """A custom property the template never sets resolves to the empty
        string, and a dot drawn in it is drawn in black on green grass."""
        markup = page.TEMPLATE.read_text()
        defined = set(re.findall(r"(--[a-z-]+)\s*:", markup))
        used = set(re.findall(r'"(--[a-z-]+)"', page.SCRIPT.read_text()))
        assert used <= defined, f"undefined in the template: {sorted(used - defined)}"
