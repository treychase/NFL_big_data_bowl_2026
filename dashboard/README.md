# Dashboard

`nfl-scouting.html` is the built scouting app: every targeted pass of the
season, filterable and drawn on the field with the flight animated, plus the
receiver and coverage tables, the route-by-coverage matrix, and the catch
probability model's own diagnostics. It is self-contained — open it in a
browser, no server and no network.

It is generated, not edited. The three parts it is generated from live here:

| File | What it is |
| --- | --- |
| `template.html` | the page's markup and styling |
| `app.js` | everything it does: filters, tables, and the field |
| `nfl-scouting.html` | the built result, which is what gets published |

`nfl_scouting/page.py` assembles them, and also injects the field geometry
from `nfl_scouting/config.py` rather than letting `app.js` carry its own copy
of it. That is what keeps the end zone the page shades and the end zone the
analysis measures against the same ten yards: change `END_ZONE_DEPTH` and the
picture moves with the numbers.

## Rebuilding

After a change to the analysis:

```bash
python -m nfl_scouting build
python -m nfl_scouting page
```

After a change to `template.html` or `app.js` alone, there is no need to refit
anything — render the payload you already have:

```bash
python -m nfl_scouting export --out /tmp/nfl_dashboard.json   # once
python -m nfl_scouting page --data /tmp/nfl_dashboard.json
```

`page` writes over `nfl-scouting.html` by default; pass `--out` for anywhere
else.

## Publishing

The page is published at
[treychase.github.io/projects/nfl-scouting.html](https://treychase.github.io/projects/nfl-scouting.html).
That site holds a copy of the built file and nothing else for this project —
no template, no script — so publishing is copying `nfl-scouting.html` into
`projects/` there. Everything about how the page looks and behaves is changed
here.
