# Dashboard

`nfl-scouting.html` is the built scouting app: every targeted pass of the
season, filterable and drawn on the field with the flight animated, plus the
receiver and coverage tables, the route-by-coverage matrix, and the catch
probability model's own diagnostics. It is self-contained — open it in a
browser, no server and no network.

It is generated, not edited. The page's template and script live with the
rest of the site build in
[treychase.github.io](https://github.com/treychase/treychase.github.io)
under `tools/`, which is where every project page on that site is built
from. To rebuild after a change to the analysis:

```bash
# here
python -m nfl_scouting build
python -m nfl_scouting export --out /tmp/nfl_dashboard.json

# in the website repo
python tools/build_nfl_scouting_page.py --data /tmp/nfl_dashboard.json
```

Then copy the result back over this file.
