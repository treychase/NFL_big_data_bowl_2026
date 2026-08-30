/* Receiver and coverage scouting - the whole page's behaviour.

   The payload arrives with its numeric columns packed as base64 typed
   arrays and its repeated text interned into one string table, so the first
   thing here is unpacking, and everything after it reads plain arrays. */
(function () {
"use strict";

var D = JSON.parse(document.getElementById("payload").textContent);
var S = D.strings;
var $ = function (id) { return document.getElementById(id); };

/* ---------- unpacking ------------------------------------------------ */

function unpack(b64, Type) {
  var bin = atob(b64), n = bin.length, bytes = new Uint8Array(n);
  for (var i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

var P = { n: D.plays.n };
Object.keys(D.plays.int16).forEach(function (k) {
  P[k] = unpack(D.plays.int16[k], Int16Array);
});
Object.keys(D.plays.f32).forEach(function (k) {
  P[k] = unpack(D.plays.f32[k], Float32Array);
});

/* Result flags were packed into one column as bits. */
var FLAG_COMPLETE = 1, FLAG_TD = 2, FLAG_INT = 4, FLAG_RIGHT = 8;
function flag(i, bit) { return (P.flags[i] & bit) !== 0; }

var GEO = D.geometry ? unpack(D.geometry.data, Int16Array) : null;
var GEO_SCALE = D.geometry ? D.geometry.scale : 10;

/* ---------- formatting ------------------------------------------------ */

function fmt(v, dp) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  return v.toFixed(dp === undefined ? 2 : dp);
}
function pct(v, dp) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  return (v * 100).toFixed(dp === undefined ? 1 : dp) + "%";
}
function signed(v, dp) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  return (v > 0 ? "+" : "") + v.toFixed(dp === undefined ? 2 : dp);
}
function str(i) { return S[i] || ""; }
function titleCase(t) {
  return String(t).replace(/_/g, " ").toLowerCase()
    .replace(/\b[a-z]/g, function (c) { return c.toUpperCase(); });
}

/* ---------- headline -------------------------------------------------- */

(function headline() {
  var m = D.meta;
  var cells = [
    ["Targets", m.nPlays.toLocaleString(), m.nGames + " games"],
    ["Completion rate", pct(m.completionRate), "league, tracked plays"],
    ["Separation at the throw", fmt(m.meanSeparationThrow) + " yd", "nearest defender"],
    ["Separation on arrival", fmt(m.meanSeparationArrival) + " yd", "when the ball gets there"],
    ["Catch probability AUC", fmt(D.model.scores.roc_auc, 3), "out of fold, by game"],
    ["Receivers", m.nReceivers, m.nDefenders + " defenders"]
  ];
  $("headline").innerHTML = cells.map(function (c) {
    return '<div><span class="k">' + c[0] + '</span>' +
           '<span class="v">' + c[1] + '</span>' +
           '<span class="n">' + c[2] + '</span></div>';
  }).join("");
  $("subline").textContent = "NFL Big Data Bowl 2026 · " + m.season +
    " season · weeks " + m.weeks[0] + "–" + m.weeks[m.weeks.length - 1];
  $("foot").innerHTML =
    "Source: NFL Big Data Bowl 2026 tracking, " + m.season + " season, weeks " +
    m.weeks[0] + "–" + m.weeks[m.weeks.length - 1] + " &middot; " +
    m.nPlays.toLocaleString() + " targeted passes &middot; " +
    "Coverage charged to the nearest defender at the release &middot; " +
    "Flight measures are differentiated from position at 10 Hz";
})();

/* ---------- tabs ------------------------------------------------------ */

Array.prototype.forEach.call(document.querySelectorAll('.tabs button'), function (b) {
  b.addEventListener("click", function () {
    Array.prototype.forEach.call(document.querySelectorAll('.tabs button'), function (o) {
      o.setAttribute("aria-selected", String(o === b));
    });
    Array.prototype.forEach.call(document.querySelectorAll('.tab'), function (s) {
      s.classList.toggle("on", s.id === "tab-" + b.dataset.tab);
    });
    if (b.dataset.tab === "model") drawModel();
    if (b.dataset.tab === "plays") drawField();
  });
});

/* ================= PLAY EXPLORER ================= */

function options(select, values, labels, all) {
  select.innerHTML = (all ? '<option value="">' + all + "</option>" : "") +
    values.map(function (v, i) {
      return '<option value="' + v + '">' + (labels ? labels[i] : v) + "</option>";
    }).join("");
}

function distinct(col) {
  var seen = {}, out = [];
  for (var i = 0; i < P.n; i++) {
    var v = str(col[i]);
    if (v && !seen[v]) { seen[v] = 1; out.push(v); }
  }
  return out.sort();
}

/* Third entry is the sort direction: -1 puts the biggest first. The default
   is the first entry, and catch over expected makes a better landing state
   than raw separation, whose top is a wall of uncontested checkdowns. */
var SORTS = [
  ["coe", "Hardest catches made", -1],
  ["sep", "Separation at the throw", -1],
  ["sepChg", "Separation lost in flight", 1],
  ["cp", "Longest odds", 1],
  ["airYds", "Air yards", -1],
  ["recCod", "Receiver change of direction", -1],
  ["recBurst", "Receiver burst", -1],
  ["covCorr", "Defender redirect to the ball", -1],
  ["covEff", "Defender pursuit efficiency", -1],
  ["yards", "Yards gained", -1]
];

options($("fWeek"), D.meta.weeks, D.meta.weeks.map(function (w) { return "Week " + w; }), "All weeks");
options($("fRoute"), distinct(P.route), distinct(P.route).map(titleCase), "All routes");
options($("fMan"), ["MAN_COVERAGE", "ZONE_COVERAGE"], ["Man", "Zone"], "Man and zone");
options($("fResult"), ["C", "I", "IN", "TD"],
        ["Complete", "Incomplete", "Intercepted", "Touchdown"], "Any result");
options($("fSort"), SORTS.map(function (s) { return s[0]; }),
        SORTS.map(function (s) { return s[1]; }));

var filtered = [], selected = 0;

function passesFilter(i, q, week, route, man, result) {
  if (week && P.week[i] !== week) return false;
  if (route && str(P.route[i]) !== route) return false;
  if (man && str(P.manZone[i]) !== man) return false;
  if (result === "C" && !flag(i, FLAG_COMPLETE)) return false;
  if (result === "I" && (flag(i, FLAG_COMPLETE) || flag(i, FLAG_INT))) return false;
  if (result === "IN" && !flag(i, FLAG_INT)) return false;
  if (result === "TD" && !flag(i, FLAG_TD)) return false;
  if (q) {
    var hay = (str(P.recv[i]) + " " + str(P.cov[i]) + " " +
               str(P.off[i]) + " " + str(P.def[i])).toLowerCase();
    if (hay.indexOf(q) === -1) return false;
  }
  return true;
}

function applyFilters() {
  var q = $("q").value.trim().toLowerCase();
  var week = parseInt($("fWeek").value, 10) || 0;
  var route = $("fRoute").value, man = $("fMan").value, result = $("fResult").value;
  var key = $("fSort").value;
  var dir = 1;
  for (var s = 0; s < SORTS.length; s++) if (SORTS[s][0] === key) dir = SORTS[s][2];

  filtered = [];
  for (var i = 0; i < P.n; i++) {
    if (passesFilter(i, q, week, route, man, result)) filtered.push(i);
  }
  var col = P[key];
  filtered.sort(function (a, b) {
    var va = col[a], vb = col[b];
    // Plays missing this measure sink to the bottom rather than leading.
    if (!isFinite(va) && !isFinite(vb)) return 0;
    if (!isFinite(va)) return 1;
    if (!isFinite(vb)) return -1;
    return dir * (va - vb);
  });

  $("playCount").textContent = filtered.length.toLocaleString() + " of " +
    P.n.toLocaleString() + " plays";
  renderPlayList();
  if (filtered.length) select(filtered[0]);
  else { $("playTitle").textContent = "No plays match those filters"; clearField(); }
}

function resultChip(i) {
  if (flag(i, FLAG_TD)) return '<span class="res t">TD</span>';
  if (flag(i, FLAG_INT)) return '<span class="res n">Int</span>';
  if (flag(i, FLAG_COMPLETE)) return '<span class="res c">Comp</span>';
  return '<span class="res i">Inc</span>';
}

function renderPlayList() {
  var shown = filtered.slice(0, 300);
  $("playList").innerHTML = shown.map(function (i) {
    return '<div class="play" data-i="' + i + '">' +
      '<div class="top"><span class="nm">' + str(P.recv[i]) + "</span>" + resultChip(i) + "</div>" +
      '<div class="meta">Wk ' + P.week[i] + " · " + str(P.off[i]) + " vs " + str(P.def[i]) +
      " · " + titleCase(str(P.route[i])) + " vs " + titleCase(str(P.manZone[i])) +
      " · " + fmt(P.sep[i], 1) + " yd sep · " + fmt(P.airYds[i], 0) + " air yd" +
      "</div></div>";
  }).join("") + (filtered.length > 300
    ? '<div class="play" style="cursor:default;color:var(--muted);font-family:var(--mono);' +
      'font-size:10px;letter-spacing:.1em;text-transform:uppercase">Showing the first 300 ' +
      "· narrow the filters to see the rest</div>" : "");

  Array.prototype.forEach.call($("playList").querySelectorAll(".play[data-i]"), function (el) {
    el.addEventListener("click", function () { select(parseInt(el.dataset.i, 10)); });
  });
}

function select(i) {
  selected = i;
  Array.prototype.forEach.call($("playList").querySelectorAll(".play"), function (el) {
    el.classList.toggle("sel", parseInt(el.dataset.i, 10) === i);
  });
  var td = flag(i, FLAG_TD) ? ", touchdown" : "";
  var res = flag(i, FLAG_INT) ? "intercepted"
          : flag(i, FLAG_COMPLETE) ? "complete for " + fmt(P.yards[i], 0) + " yards" + td
          : "incomplete";
  $("playTitle").textContent = str(P.recv[i]) + " — " + res;
  $("playMeta").textContent =
    "Week " + P.week[i] + " · " + str(P.off[i]) + " vs " + str(P.def[i]) +
    " · Q" + P.quarter[i] + " · " + P.down[i] + " and " + P.toGo[i] +
    " · " + titleCase(str(P.route[i])) + " vs " + titleCase(str(P.covType[i])) +
    (str(P.cov[i]) ? " · covered by " + str(P.cov[i]) + " (" + str(P.covPos[i]) + ")" : "");

  $("playStats").innerHTML = renderStatGroups(i);

  $("playNote").textContent = describe(i);
  loadGeometry(i);
}

/* ---------- play metrics, against the season ---------------------------- */
/* A number on its own says very little: 62 degrees of turn in the air is a
   lot or a little only against the other fourteen thousand targets. So each
   metric carries its percentile among every play that has one, shaded on the
   page's diverging ramp, and the metrics are grouped by whose they are - the
   throw, the receiver, the defender - in the colours those players are drawn
   in on the field above.
   
   The chip says where the number sits, not whether it is good. Which
   direction is good depends on who you are scouting: a defender's closing
   speed and a receiver's separation are both high-is-good, and both sit in
   the same play. Colouring by "good" would need the page to pick a side. */

var STAT_GROUPS = [
  ["The throw", "", [
    ["Air yards", "airYds", function (v) { return fmt(v, 1); }],
    ["Time in the air", "airTime", function (v) { return fmt(v, 1) + " s"; }],
    ["Catch probability", "cp", function (v) { return pct(v, 0); }],
    ["Over expected", "coe", function (v) { return signed(v, 2); }]
  ]],
  ["The receiver", "recv", [
    ["Separation at throw", "sep", function (v) { return fmt(v, 1) + " yd"; }],
    ["Separation on arrival", "sepArr", function (v) { return fmt(v, 1) + " yd"; }],
    ["Turn in flight", "recCod", function (v) { return fmt(v, 0) + "\u00B0"; }],
    ["Burst", "recBurst", function (v) { return fmt(v, 1) + " yd/s\u00B2"; }]
  ]],
  ["The defender", "cov", [
    ["Turn in flight", "covCod", function (v) { return fmt(v, 0) + "\u00B0"; }],
    ["Redirect to the ball", "covCorr", function (v) { return signed(v, 0) + "\u00B0"; }],
    ["Pursuit", "covEff", function (v) { return fmt(v, 2); }],
    ["Closing speed", "covClose", function (v) { return fmt(v, 1) + " yd/s"; }]
  ]]
];

/* Sorted copies, built once per metric on first use rather than all twelve up
   front: a viewer who never opens the play explorer never pays for them. */
var SORTED = {};
function sortedValues(key) {
  if (!SORTED[key]) {
    var out = [];
    for (var k = 0; k < P.n; k++) {
      var v = P[key][k];
      if (isFinite(v)) out.push(v);
    }
    out.sort(function (a, b) { return a - b; });
    SORTED[key] = out;
  }
  return SORTED[key];
}

/* Share of plays at or below this value, 0-100. Ties land at the midpoint of
   their own run, so a metric where a third of the league sits on one value
   does not report that third as the 99th percentile. */
function metricPercentile(key, value) {
  if (!isFinite(value)) return null;
  var a = sortedValues(key);
  if (!a.length) return null;
  var lo = 0, hi = a.length;
  while (lo < hi) { var m = (lo + hi) >> 1; if (a[m] < value) lo = m + 1; else hi = m; }
  var first = lo;
  hi = a.length;
  while (lo < hi) { var m2 = (lo + hi) >> 1; if (a[m2] <= value) lo = m2 + 1; else hi = m2; }
  return (first + lo) / 2 / a.length * 100;
}

function renderStatGroups(i) {
  return STAT_GROUPS.map(function (group) {
    var swatch = group[1]
      ? '<i style="background:var(--' + group[1] + ')"></i>'
      : '<i style="background:var(--other)"></i>';
    var cells = group[2].map(function (stat) {
      var value = P[stat[1]][i];
      var shown = isFinite(value) ? stat[2](value) : "\u2013";
      var p = metricPercentile(stat[1], value);
      var chip = p === null
        ? '<span class="p na">no rank</span>'
        : '<span class="p" style="background:' + ramp(p / 100) + ";color:" +
          textOnRamp(p / 100) + '">' + Math.round(p) + '</span>';
      return '<div class="stat' + (group[1] ? " " + group[1] : "") + '">' +
        '<span class="k">' + stat[0] + "</span>" +
        '<span class="vrow"><span class="v">' + shown + "</span>" + chip + "</span></div>";
    }).join("");
    return '<div class="sgroup"><h4>' + swatch + group[0] + "</h4>" +
           '<div class="grid4">' + cells + "</div></div>";
  }).join("");
}

function describe(i) {
  var sep = P.sep[i], chg = P.sepChg[i], coe = P.coe[i], corr = P.covCorr[i];
  var sentences = [];

  /* The release and what happened to that gap belong in one sentence, so
     the clauses are built as a pair rather than joined with a full stop. */
  if (isFinite(sep)) {
    var opening = sep >= D.meta.openThreshold
      ? "Open by " + fmt(sep, 1) + " yards when the ball came out"
      : "Covered to " + fmt(sep, 1) + " yards at the release";
    if (isFinite(chg)) {
      opening += chg < -0.5
        ? ", and the defender closed " + fmt(-chg, 1) + " of that in the air"
        : chg > 0.5
          ? ", and he pulled away by another " + fmt(chg, 1)
          : ", and the gap held through the flight";
    }
    sentences.push(opening);
  }
  if (isFinite(corr)) {
    sentences.push(corr > 20
      ? "The defender turned hard back to the ball, taking " + fmt(corr, 0) +
        "° off his angle to it"
      : corr < -20
        ? "The defender never got turned around, losing " + fmt(-corr, 0) +
          "° off the ball's line"
        : "The defender held his angle to the ball");
  }
  if (isFinite(coe) && isFinite(P.cp[i])) {
    sentences.push("The throw was " + pct(P.cp[i], 0) + " to be caught, and it was "
      + (flag(i, FLAG_COMPLETE) ? "caught" : "not"));
  }
  return sentences.length ? sentences.join(". ") + "." : "";
}

/* ---------- the field ------------------------------------------------- */

/* Tracking coordinates run the length of the field in x and across it in y,
   so the playing field is the hundred yards between the two goal lines and
   the depth outside each of them is end zone. The numbers come from
   nfl_scouting/config.py, injected at build time, so the picture and the
   analysis cannot disagree about where the field ends. */
var FIELD = JSON.parse(document.getElementById("field-geometry").textContent);
var FIELD_LENGTH = FIELD.length, FIELD_WIDTH = FIELD.width;
var GOAL_NEAR = FIELD.endZone, GOAL_FAR = FIELD.length - FIELD.endZone;

var geo = null, frame = 0, timer = null;

/* The four kinds the geometry packs, in the order it packs them: the
   targeted receiver, the defender charged with covering him, the rest of the
   offence, and the rest of the defence. The two the page is about get their
   own colours; everybody else is drawn by his side of the ball, so every
   defender in the coverage reads as a defender rather than as one of a set
   of anonymous grey dots. */
var KIND_TARGET = 0, KIND_COVERAGE = 1, KIND_OFFENCE = 2, KIND_DEFENCE = 3;
var KIND_COLOR = ["--recv", "--cov", "--other", "--cov-soft"];
function kindColor(kind) { return cssVar(KIND_COLOR[kind] || "--other"); }
function isLead(kind) { return kind === KIND_TARGET || kind === KIND_COVERAGE; }

/* The play is recorded in two phases at two rates: everyone from the snap to
   the throw at PRE_STEP seconds a sample, then the players near the ball
   through the flight at FLIGHT_STEP. The animation runs on one clock at the
   flight's rate, so a tick is always the same slice of real time; positions
   between two pre-snap samples are drawn by interpolating them, which is a
   rendering choice about a smooth path and not a claim that anybody was
   measured there. */
var PRE_STEP = (D.geometry && D.geometry.preStep) || 0.3;
var FLIGHT_STEP = (D.geometry && D.geometry.flightStep) || 0.1;
var PRE_TICKS = Math.max(1, Math.round(PRE_STEP / FLIGHT_STEP));

function loadGeometry(i) {
  geo = null;
  if (GEO && D.geometry.offsets[i] >= 0) {
    var o = D.geometry.offsets[i];
    var nPre = GEO[o], nPreFrames = GEO[o + 1], nFlight = GEO[o + 2];
    var p = o + 3;

    var pre = [];
    for (var s = 0; s < nPre; s++) {
      /* inAir says the tracking followed this player past the release, so
         his flight block draws him from there on. Everyone else stops at
         the throw, which is the last thing recorded about him. */
      var kind = GEO[p], inAir = GEO[p + 1] === 1; p += 2;
      var path = [];
      for (var q = 0; q < nPreFrames; q++, p += 2) {
        path.push({ x: GEO[p] / GEO_SCALE, y: GEO[p + 1] / GEO_SCALE });
      }
      pre.push({ kind: kind, inAir: inAir, path: path });
    }

    var flight = [], maxLen = 0;
    for (var f = 0; f < nFlight; f++) {
      var kind = GEO[p], len = GEO[p + 1];
      p += 2;
      var pts = [];
      for (var k = 0; k < len; k++, p += 2) {
        pts.push({ x: GEO[p] / GEO_SCALE, y: GEO[p + 1] / GEO_SCALE });
      }
      flight.push({ kind: kind, pts: pts });
      if (len > maxLen) maxLen = len;
    }

    /* The targeted receiver's own pre-snap path is the route, and the last
       sample of every path is that player at the release. */
    var target = null;
    pre.forEach(function (q) { if (q.kind === KIND_TARGET) target = q; });

    var preTicks = Math.max(0, (nPreFrames - 1)) * PRE_TICKS;
    /* The two the page is about are drawn last so a safety crossing behind
       them cannot sit on top of the pair the play is about. */
    var order = pre.filter(function (q) { return !isLead(q.kind); })
                   .concat(pre.filter(function (q) { return isLead(q.kind); }));

    geo = {
      pre: pre, preDrawOrder: order, flight: flight,
      route: target ? target.path : [],
      preFrames: nPreFrames, preTicks: preTicks,
      flightFrames: maxLen,
      frames: preTicks + Math.max(0, maxLen - 1)
    };
  }
  frame = 0;
  $("scrub").max = geo ? geo.frames : 1;
  $("scrub").value = 0;
  stop();
  drawField();
}

/* Where a pre-snap player is at tick t of the run-up, interpolating between
   the two samples that bracket it. */
function preAt(path, tick) {
  var exact = tick / PRE_TICKS;
  var a = Math.min(Math.floor(exact), path.length - 1);
  var b = Math.min(a + 1, path.length - 1);
  var u = exact - a;
  return { x: path[a].x + (path[b].x - path[a].x) * u,
           y: path[a].y + (path[b].y - path[a].y) * u };
}

var DOWN_NAMES = ["", "1ST", "2ND", "3RD", "4TH"];
function ordinalDown(d) { return DOWN_NAMES[d] || String(d); }

/* The x of the line to gain, or null when the goal line already is it.
   FLAG_RIGHT is the same flag the drawing uses to put the offence
   left-to-right, so it is also the one that says which way "forward" is. */
function firstDownLine(i) {
  var los = P.losX[i], toGo = P.toGo[i];
  if (!isFinite(los) || !isFinite(toGo) || toGo <= 0) return null;
  var right = flag(i, FLAG_RIGHT);
  var x = right ? los + toGo : los - toGo;
  var goal = right ? GOAL_FAR : GOAL_NEAR;
  if (right ? x >= goal : x <= goal) return null;   /* goal to go */
  return x;
}

/* A full-height line with a small label sitting on it. The label gets its own
   plate rather than being drawn straight onto the grass, because a yard
   number or a route can pass behind it and the two would overprint. */
function markLine(g, X, W, color, label) {
  if (X < -2 || X > W + 2) return;
  g.strokeStyle = color;
  g.lineWidth = 2.4;
  g.beginPath(); g.moveTo(X, 0); g.lineTo(X, g.canvas.height); g.stroke();

  g.font = "700 10px " + cssVar("--mono");
  var w = g.measureText(label).width + 10;
  /* Keep the plate inside the canvas when the line is near an edge. */
  var left = Math.max(1, Math.min(X - w / 2, W - w - 1));
  g.fillStyle = color;
  g.fillRect(left, 6, w, 15);
  g.fillStyle = "#14201B";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(label, left + w / 2, 14);
  g.textBaseline = "alphabetic";
}

/* Resolved once each. The field asks for one of these per player per frame
   now that the whole coverage is drawn in colour, and getComputedStyle is a
   style recalculation every time it is called. */
var CSS_CACHE = {};
function cssVar(name) {
  if (!(name in CSS_CACHE)) {
    CSS_CACHE[name] =
      getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  return CSS_CACHE[name];
}

function clearField() {
  var c = $("field"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
}

function drawField() {
  var c = $("field"), g = c.getContext("2d");
  var i = selected;
  if (!isFinite(P.landX[i])) { clearField(); return; }

  /* The view follows the play rather than showing all 120 yards, so a five
     yard hitch is not drawn as four pixels of movement. By default the box
     is built from the ball, the route and the flight - not from every
     tracked player - because a safety standing forty yards off would
     otherwise set the zoom for a one yard screen and shrink the actual play
     to a dot. "Whole coverage" is the other side of that trade, and it is
     the reader's to make. Players outside the frame simply are not drawn. */
  /* The line to gain. Which way the offence is going decides the sign, and
     the flag that decides it is the same one that flips the drawing below.
     In a goal to go situation the line to gain *is* the goal line, which is
     already drawn heavier than anything else, so it comes back null rather
     than as a second line laid over the first. */
  var firstDownX = firstDownLine(i);

  /* The line to gain earns its place in the fit box even though it is not
     part of the action: on a third and long checkdown it is the whole point
     of the play, and it sits outside a box built from the ball and the route
     alone on a third of the season's targets. It costs a couple of yards of
     zoom to include and answers "did this get there" without arithmetic. */
  var xs = [P.losX[i], P.landX[i]], ys = [P.landY[i]];
  if (firstDownX !== null) xs.push(firstDownX);
  if (geo) {
    geo.route.forEach(function (q) { xs.push(q.x); ys.push(q.y); });
    geo.flight.forEach(function (t) {
      t.pts.forEach(function (q) { xs.push(q.x); ys.push(q.y); });
    });
    /* Half the coverage runs outside that box on a typical play, which is
       the cost of framing the action rather than the field. The toggle buys
       the rest of it back at about twice the width - the whole defence, at
       half the zoom - and it is off by default because most plays are worth
       more as the close view. */
    if ($("fitAll").checked) {
      geo.pre.forEach(function (q) {
        q.path.forEach(function (a) { xs.push(a.x); ys.push(a.y); });
      });
    }
  }
  /* Fit the action to the canvas. The window has to be widened to the
     canvas's own aspect ratio, or a short throw - where everything happens
     inside twenty yards - gets drawn into a square in the middle and leaves
     half the canvas empty. */
  var pad = 5, minSpan = 14;
  var W = c.width, H = c.height, aspect = W / H;
  var x0 = Math.min.apply(null, xs) - pad, x1 = Math.max.apply(null, xs) + pad;
  var y0 = Math.min.apply(null, ys) - pad, y1 = Math.max.apply(null, ys) + pad;
  var midX = (x0 + x1) / 2, midY = (y0 + y1) / 2;
  var spanX = Math.max(x1 - x0, minSpan);
  var spanY = Math.max(y1 - y0, minSpan / aspect);
  if (spanX / spanY < aspect) spanX = spanY * aspect; else spanY = spanX / aspect;
  x0 = midX - spanX / 2; x1 = midX + spanX / 2;
  y0 = midY - spanY / 2; y1 = midY + spanY / 2;

  var scale = W / spanX;
  var right = flag(i, FLAG_RIGHT);
  /* Always draw the offence moving left to right. */
  function px(x) { return (right ? (x - x0) : (x1 - x)) * scale; }
  function py(y) { return (right ? (y1 - y) : (y - y0)) * scale; }

  g.fillStyle = cssVar("--field");
  g.fillRect(0, 0, W, H);

  /* The end zones, drawn before anything else so the lines land on top.
     Tracking coordinates run 0-120 along the field: the goal lines are at
     10 and 110, and the ten yards beyond each are end zone. The view
     usually sits in midfield and never sees them, but a red zone target
     is exactly the play where knowing where the back line is matters. */
  g.fillStyle = cssVar("--field-deep");
  [[0, GOAL_NEAR], [GOAL_FAR, FIELD_LENGTH]].forEach(function (zone) {
    if (zone[1] < x0 || zone[0] > x1) return;
    var a = px(zone[0]), b = px(zone[1]);
    g.fillRect(Math.min(a, b), 0, Math.abs(b - a), H);
  });

  /* Yard lines every five, numbers every ten. Both are clipped to the field
     of play: an end zone carries no five yard lines, and the goal line
     itself is unnumbered rather than a "0". */
  g.strokeStyle = "rgba(236,239,230,.20)";
  g.fillStyle = "rgba(236,239,230,.45)";
  g.font = "600 11px " + cssVar("--mono");
  g.textAlign = "center";
  for (var yd = Math.ceil(x0 / 5) * 5; yd <= x1; yd += 5) {
    if (yd < GOAL_NEAR || yd > GOAL_FAR) continue;
    var X = px(yd);
    g.lineWidth = (yd % 10 === 0) ? 1.4 : 0.7;
    g.beginPath(); g.moveTo(X, 0); g.lineTo(X, H); g.stroke();
    if (yd % 10 === 0 && yd > GOAL_NEAR && yd < GOAL_FAR) {
      var number = yd <= 60 ? yd - GOAL_NEAR : GOAL_FAR - yd;
      g.fillText(String(number), X, H - 8);
    }
  }

  /* Goal lines and the back of each end zone. The goal line is the one
     line on the field a defence is actually defending, so it is drawn
     brighter and heavier than the yard lines it sits among. */
  [GOAL_NEAR, GOAL_FAR].forEach(function (line) {
    if (line < x0 || line > x1) return;
    g.strokeStyle = "rgba(236,239,230,.85)";
    g.lineWidth = 2.4;
    g.beginPath(); g.moveTo(px(line), 0); g.lineTo(px(line), H); g.stroke();
  });
  [0, FIELD_LENGTH].forEach(function (line) {
    if (line < x0 || line > x1) return;
    g.strokeStyle = "rgba(236,239,230,.55)";
    g.lineWidth = 1.8;
    g.beginPath(); g.moveTo(px(line), 0); g.lineTo(px(line), H); g.stroke();
  });

  /* END ZONE across each one, but only when enough of it is on screen for
     the word to sit inside its own paint rather than run out over a goal
     line. */
  [[0, GOAL_NEAR], [GOAL_FAR, FIELD_LENGTH]].forEach(function (zone) {
    var a = px(zone[0]), b = px(zone[1]);
    var left = Math.min(a, b), width = Math.abs(b - a);
    var visible = Math.min(left + width, W) - Math.max(left, 0);
    if (visible < 74) return;
    g.save();
    g.translate((Math.max(left, 0) + Math.min(left + width, W)) / 2, H / 2);
    g.rotate(-Math.PI / 2);
    g.fillStyle = "rgba(236,239,230,.42)";
    g.font = "700 13px " + cssVar("--mono");
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText("END ZONE", 0, 0);
    g.restore();
    g.textBaseline = "alphabetic";
  });

  /* Sidelines. */
  g.strokeStyle = "rgba(236,239,230,.30)";
  g.lineWidth = 1.6;
  [0, FIELD_WIDTH].forEach(function (side) {
    if (side < y0 || side > y1) return;
    var Y = py(side);
    g.beginPath(); g.moveTo(0, Y); g.lineTo(W, Y); g.stroke();
  });
  /* Hash marks at the NFL's 70'9" inbound lines. */
  g.strokeStyle = "rgba(236,239,230,.16)";
  g.lineWidth = 1;
  [23.58, 29.72].forEach(function (hy) {
    if (hy < y0 || hy > y1) return;
    var Y = py(hy);
    g.beginPath(); g.moveTo(0, Y); g.lineTo(W, Y); g.stroke();
  });

  /* Line of scrimmage and line to gain, in the colours a broadcast uses for
     them. Drawn after the field markings and before the players, so a
     receiver is never hidden behind a line. */
  markLine(g, px(P.losX[i]), W, cssVar("--los"), "LOS");
  if (firstDownX !== null) {
    markLine(g, px(firstDownX), W, cssVar("--first"),
             ordinalDown(P.down[i]) + " & " + P.toGo[i]);
  }

  if (!geo) {
    g.fillStyle = "rgba(236,239,230,.7)";
    g.font = "600 12px " + cssVar("--mono");
    g.fillText("No tracking geometry for this play", W / 2, H / 2);
    return;
  }

  /* Where the play has got to. Both halves of the drawing below read these,
     so they are settled once here rather than recomputed per block. */
  var inRunUp = frame < geo.preTicks;
  var preTick = Math.min(frame, geo.preTicks);

  /* The route, drawn only as far as the play has actually got, so the run-up
     traces itself out rather than sitting there finished from the first
     frame. */
  if (geo.route.length) {
    var drawn = Math.min(preTick / PRE_TICKS, geo.route.length - 1);
    g.strokeStyle = "rgba(223,162,44,.55)";
    g.lineWidth = 2;
    g.beginPath();
    for (var rk = 0; rk <= Math.floor(drawn); rk++) {
      var q = geo.route[rk];
      if (rk === 0) g.moveTo(px(q.x), py(q.y)); else g.lineTo(px(q.x), py(q.y));
    }
    var head = preAt(geo.route, preTick);
    g.lineTo(px(head.x), py(head.y));
    g.stroke();
  }

  /* Everyone the tracking carries, moving from the snap: every defender in
     coverage purple rather than as one anonymous grey dot, so the whole
     coverage is watchable and not just the man on the target.

     After the throw the ones the tracking follows into the air are drawn by
     their flight paths below and skipped here. The rest have no recorded
     position past the release - the competition's output file follows only a
     few players into the air - so they are drawn hollow at the spot the
     release left them, which says "last seen here" rather than pretending to
     a measurement nobody took. */
  geo.preDrawOrder.forEach(function (q) {
    var at = preAt(q.path, preTick);
    if (at.x < x0 || at.x > x1 || at.y < y0 || at.y > y1) return;
    if (!inRunUp && q.inAir) return;      /* his flight path draws him */
    var lead = isLead(q.kind);
    var r = lead ? 6.5 : q.kind === KIND_DEFENCE ? 5 : 4;
    var X = px(at.x), Y = py(at.y);
    if (!inRunUp) {
      /* Stopped: an open ring, in his own colour so he is still identifiable
         as the defender he was, at the width the live dots are drawn. */
      g.strokeStyle = kindColor(q.kind);
      g.lineWidth = 1.6;
      g.globalAlpha = q.kind === KIND_OFFENCE ? 0.45 : 0.75;
      g.beginPath(); g.arc(X, Y, r, 0, 6.284); g.stroke();
      g.globalAlpha = 1;
      return;
    }
    g.fillStyle = kindColor(q.kind);
    g.globalAlpha = q.kind === KIND_OFFENCE ? 0.55 : 1;
    g.beginPath(); g.arc(X, Y, r, 0, 6.284); g.fill();
    g.globalAlpha = 1;
    /* The two the page is about carry a light rim, which is what keeps them
       the subject now that everybody else is in colour too. */
    if (lead) {
      g.strokeStyle = "rgba(236,239,230,.85)";
      g.lineWidth = 1.5;
      g.beginPath(); g.arc(X, Y, r, 0, 6.284); g.stroke();
    }
  });

  /* Ball landing spot. */
  var bx = px(P.landX[i]), by = py(P.landY[i]);
  g.strokeStyle = cssVar("--ball");
  g.lineWidth = 2;
  g.beginPath(); g.arc(bx, by, 8, 0, 6.284); g.stroke();
  g.beginPath();
  g.moveTo(bx - 5, by - 5); g.lineTo(bx + 5, by + 5);
  g.moveTo(bx + 5, by - 5); g.lineTo(bx - 5, by + 5);
  g.stroke();

  /* Flight paths, drawn up to the current frame - and not at all until the
     ball is out, or every player the tracking follows into the air would sit
     as a second dot on his own release spot while he is still running the
     route. */
  if (!inRunUp) geo.flight.forEach(function (t) {
    var color = kindColor(t.kind);
    var lead = isLead(t.kind);
    var r = lead ? 6.5 : t.kind === KIND_DEFENCE ? 5 : 4;
    var upto = Math.max(0, Math.min(frame - geo.preTicks, t.pts.length - 1));
    g.strokeStyle = color;
    g.lineWidth = lead ? 2.5 : t.kind === KIND_DEFENCE ? 2 : 1.5;
    g.globalAlpha = t.kind === KIND_OFFENCE ? 0.55 : lead ? 1 : 0.8;
    g.beginPath();
    for (var k = 0; k <= upto; k++) {
      var q = t.pts[k];
      if (k === 0) g.moveTo(px(q.x), py(q.y)); else g.lineTo(px(q.x), py(q.y));
    }
    g.stroke();
    var head = t.pts[upto];
    g.fillStyle = color;
    g.beginPath(); g.arc(px(head.x), py(head.y), r, 0, 6.284); g.fill();
    g.globalAlpha = 1;
    if (lead) {
      g.strokeStyle = "rgba(236,239,230,.85)";
      g.lineWidth = 1.5;
      g.beginPath(); g.arc(px(head.x), py(head.y), r, 0, 6.284); g.stroke();
    }
  });

  /* One clock, zeroed at the throw: negative through the run-up, positive
     while the ball is in the air. */
  var seconds = (frame - geo.preTicks) * FLIGHT_STEP;
  var air = Math.max(0, geo.flightFrames - 1) * FLIGHT_STEP;
  $("clock").textContent = seconds < 0
    ? Math.abs(seconds).toFixed(1) + " s before the throw"
    : seconds.toFixed(1) + " s of " + air.toFixed(1) + " s in the air";
}

function stop() {
  if (timer) { clearInterval(timer); timer = null; }
  $("playBtn").textContent = "Play";
}

$("playBtn").addEventListener("click", function () {
  if (timer) { stop(); return; }
  if (!geo) return;
  if (frame >= geo.frames - 1) frame = 0;
  $("playBtn").textContent = "Pause";
  timer = setInterval(function () {
    frame++;
    if (frame >= geo.frames - 1) { frame = geo.frames - 1; stop(); }
    $("scrub").value = frame;
    drawField();
  }, 100);
});

$("scrub").addEventListener("input", function () {
  stop();
  frame = parseInt($("scrub").value, 10);
  drawField();
});

/* The zoom, not the clock: redraw the frame the play is already on. */
$("fitAll").addEventListener("change", drawField);

["q", "fWeek", "fRoute", "fMan", "fResult", "fSort"].forEach(function (id) {
  $(id).addEventListener("input", applyFilters);
});
$("reset").addEventListener("click", function () {
  $("q").value = ""; $("fWeek").value = ""; $("fRoute").value = "";
  $("fMan").value = ""; $("fResult").value = ""; $("fSort").value = SORTS[0][0];
  applyFilters();
});

/* ================= SORTABLE TABLES ================= */

function sortableTable(el, rows, columns, state) {
  function render() {
    var sorted = rows.slice().sort(function (a, b) {
      var va = a[state.key], vb = b[state.key];
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      if (typeof va === "string") return state.dir * va.localeCompare(vb);
      return state.dir * (va - vb);
    });
    el.innerHTML =
      "<thead><tr>" + columns.map(function (c) {
        var dir = c.key === state.key ? (state.dir === 1 ? "asc" : "desc") : "";
        return '<th data-key="' + c.key + '"' + (dir ? ' data-dir="' + dir + '"' : "") +
               (c.title ? ' title="' + c.title + '"' : "") + ">" + c.label + "</th>";
      }).join("") + "</tr></thead><tbody>" +
      sorted.map(function (r) {
        return "<tr>" + columns.map(function (c) {
          return "<td>" + c.fmt(r[c.key], r) + "</td>";
        }).join("") + "</tr>";
      }).join("") + "</tbody>";

    Array.prototype.forEach.call(el.querySelectorAll("th"), function (th) {
      th.addEventListener("click", function () {
        var key = th.dataset.key;
        if (state.key === key) state.dir = -state.dir;
        else { state.key = key; state.dir = typeof rows[0][key] === "string" ? 1 : -1; }
        render();
      });
    });
  }
  render();
}

var text = function (v) { return v === null || v === undefined ? "–" : v; };
var num = function (dp) { return function (v) { return fmt(v, dp); }; };
var rate = function (dp) { return function (v) { return pct(v, dp); }; };
var plus = function (dp) { return function (v) { return signed(v, dp); }; };
var pctPlus = function (v) {
  return v === null || !isFinite(v) ? "–" : (v > 0 ? "+" : "") + (v * 100).toFixed(1);
};

/* ---------- receivers -------------------------------------------------- */

var RECV_COLUMNS = [
  { key: "player_name", label: "Receiver", fmt: text },
  { key: "position", label: "Pos", fmt: text },
  { key: "targets", label: "Tgt", fmt: num(0) },
  { key: "separation_at_throw_yd", label: "Sep throw", fmt: num(2),
    title: "Yards to the nearest defender when the ball was released" },
  { key: "separation_at_arrival_yd", label: "Sep arrival", fmt: num(2),
    title: "Yards to the nearest tracked defender when the ball got there" },
  { key: "separation_change_yd", label: "Δ in flight", fmt: plus(2),
    title: "Negative means the defender closed while the ball was in the air" },
  { key: "open_rate_at_throw", label: "Open rate", fmt: rate(0),
    title: "Share of targets with at least three yards of separation at the throw" },
  { key: "catch_rate", label: "Catch %", fmt: rate(1) },
  { key: "expected_catch_rate", label: "Expected", fmt: rate(1),
    title: "What the model gave those throws, from the release only" },
  { key: "catch_rate_over_expected", label: "CROE", fmt: pctPlus,
    title: "Catch rate over expected, in points" },
  { key: "yards_per_target", label: "Yds/tgt", fmt: num(1) },
  { key: "passer_rating_when_targeted", label: "Rating", fmt: num(1),
    title: "Passer rating on throws to this receiver" },
  { key: "air_accel_burst_yps2", label: "Burst", fmt: num(2),
    title: "Best half second of speed gain while the ball was in the air, yd/s²" },
  { key: "air_speed_max_yps", label: "Top speed", fmt: num(2),
    title: "Fastest tenth of a second in flight, yd/s" },
  { key: "air_cod_total_deg", label: "Turn", fmt: num(1),
    title: "Total degrees of heading change while the ball was in the air" },
  { key: "air_pursuit_efficiency", label: "To the ball", fmt: num(2),
    title: "Ground closed on the landing spot over ground covered; one is a straight line" }
];

var DEF_COLUMNS = [
  { key: "player_name", label: "Defender", fmt: text },
  { key: "position", label: "Pos", fmt: text },
  { key: "targets_covered", label: "Tgt", fmt: num(0) },
  { key: "passer_rating_allowed", label: "Rating allowed", fmt: num(1),
    title: "Passer rating on throws where he was the nearest defender" },
  { key: "completion_rate_allowed", label: "Comp % allowed", fmt: rate(1) },
  { key: "yards_per_target_allowed", label: "Yds/tgt", fmt: num(1) },
  { key: "touchdowns_allowed", label: "TD", fmt: num(0) },
  { key: "interceptions", label: "Int", fmt: num(0) },
  { key: "separation_allowed_at_throw_yd", label: "Sep allowed", fmt: num(2) },
  { key: "separation_allowed_at_arrival_yd", label: "On arrival", fmt: num(2) },
  { key: "open_rate_allowed_at_throw", label: "Open rate", fmt: rate(0) },
  { key: "catch_rate_over_expected_allowed", label: "CROE allowed", fmt: pctPlus,
    title: "Negative means fewer catches than the geometry of those throws deserved" },
  { key: "air_bearing_correction_deg", label: "Redirect", fmt: plus(1),
    title: "Degrees of angle to the ball he took out during the flight; positive is turning onto it" },
  { key: "air_cod_total_deg", label: "Turn", fmt: num(1) },
  { key: "air_pursuit_efficiency", label: "To the ball", fmt: num(2) },
  { key: "air_closing_speed_yps", label: "Closing", fmt: num(2),
    title: "Yards per second closed on the landing spot" }
];

function setupPlayerTab(rows, columns, ids, defaultKey, defaultDir, minKey) {
  var positions = [];
  rows.forEach(function (r) {
    if (r.position && positions.indexOf(r.position) === -1) positions.push(r.position);
  });
  positions.sort();
  options($(ids.pos), positions, positions, "All positions");
  $(ids.min).value = Math.min.apply(null, rows.map(function (r) { return r[minKey]; }));

  var state = { key: defaultKey, dir: defaultDir };
  function refresh() {
    var q = $(ids.q).value.trim().toLowerCase();
    var pos = $(ids.pos).value;
    var min = parseInt($(ids.min).value, 10) || 0;
    var shown = rows.filter(function (r) {
      if (pos && r.position !== pos) return false;
      if (r[minKey] < min) return false;
      if (q && r.player_name.toLowerCase().indexOf(q) === -1) return false;
      return true;
    });
    $(ids.count).textContent = shown.length + " players";
    if (!shown.length) { $(ids.table).innerHTML = ""; return; }
    sortableTable($(ids.table), shown, columns, state);
  }
  [ids.q, ids.pos, ids.min].forEach(function (id) {
    $(id).addEventListener("input", refresh);
  });
  refresh();
}

setupPlayerTab(D.receivers, RECV_COLUMNS,
  { q: "rq", pos: "rPos", min: "rMin", count: "rCount", table: "rTable" },
  "targets", -1, "targets");

setupPlayerTab(D.defenders, DEF_COLUMNS,
  { q: "dq", pos: "dPos", min: "dMin", count: "dCount", table: "dTable" },
  "passer_rating_allowed", 1, "targets_covered");

/* ================= ROUTE x COVERAGE ================= */

var MEASURES = [
  ["catch_rate", "Catch rate", 1, "rate"],
  ["passer_rating", "Passer rating", 1, "num1"],
  ["separation_at_throw_yd", "Separation at the throw", 1, "num2"],
  ["separation_at_arrival_yd", "Separation on arrival", 1, "num2"],
  ["separation_change_yd", "Separation lost in flight", 1, "plus2"],
  ["open_rate_at_throw", "Open rate", 1, "rate"],
  ["yards_per_target", "Yards per target", 1, "num1"],
  ["interception_rate", "Interception rate", -1, "rate2"],
  ["mean_air_yards", "Air yards", 1, "num1"],
  ["rec_cod_total_deg", "Receiver turn in flight", 1, "num1"],
  ["rec_accel_burst_yps2", "Receiver burst", 1, "num2"],
  ["cov_cod_total_deg", "Defender turn in flight", 1, "num1"],
  ["cov_pursuit_efficiency", "Defender pursuit to the ball", 1, "num2"]
];
var CELL_FORMAT = {
  rate: function (v) { return pct(v, 0); },
  rate2: function (v) { return pct(v, 1); },
  num1: function (v) { return fmt(v, 1); },
  num2: function (v) { return fmt(v, 2); },
  plus2: function (v) { return signed(v, 2); }
};

options($("mMetric"), MEASURES.map(function (m) { return m[0]; }),
        MEASURES.map(function (m) { return m[1]; }));
options($("mSplit"), ["manZone", "covType"], ["Man vs zone", "Coverage shell"]);

/* Blue to red through a near-neutral middle: readable for a red-green
   colourblind reader, which red-to-green is not. The three stops come off
   the CSS variables so the heat map here and the percentile chips on the
   play explorer cannot drift apart. */
function rgbOf(name) {
  var hex = cssVar(name);
  return [1, 3, 5].map(function (i) { return parseInt(hex.substr(i, 2), 16); });
}
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  var a = rgbOf("--cold"), b = rgbOf("--mid"), c = rgbOf("--hot");
  var lo = t < 0.5 ? a : b, hi = t < 0.5 ? b : c;
  var u = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return "rgb(" + lo.map(function (v, i) {
    return Math.round(v + (hi[i] - v) * u);
  }).join(",") + ")";
}
/* Dark or light text, whichever survives on the ramp colour underneath. */
function textOnRamp(t) {
  var m = /(\d+),(\d+),(\d+)/.exec(ramp(t));
  var l = (0.2126 * m[1] + 0.7152 * m[2] + 0.0722 * m[3]) / 255;
  return l > 0.55 ? "#14201B" : "#ffffff";
}

function drawMatrix() {
  var metric = $("mMetric").value;
  var split = $("mSplit").value;
  var rows = split === "manZone" ? D.routeCoverage : D.coverageType;
  var colKey = split === "manZone" ? "team_coverage_man_zone" : "team_coverage_type";
  var spec = MEASURES.filter(function (m) { return m[0] === metric; })[0];
  var format = CELL_FORMAT[spec[3]];

  var routes = [], columns = [], byKey = {};
  rows.forEach(function (r) {
    var route = r.route_of_targeted_receiver, col = r[colKey];
    if (routes.indexOf(route) === -1) routes.push(route);
    if (columns.indexOf(col) === -1) columns.push(col);
    byKey[route + "|" + col] = r;
  });
  routes.sort(); columns.sort();

  var values = rows.map(function (r) { return r[metric]; })
                   .filter(function (v) { return v !== null && isFinite(v); });
  var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
  var span = (hi - lo) || 1;

  var html = "<thead><tr><th></th>" + columns.map(function (c) {
    return "<th>" + titleCase(c) + "</th>";
  }).join("") + "</tr></thead><tbody>";
  routes.forEach(function (route) {
    html += "<tr><th style='text-align:left'>" + titleCase(route) + "</th>";
    columns.forEach(function (col) {
      var cell = byKey[route + "|" + col];
      if (!cell || cell[metric] === null || !isFinite(cell[metric])) {
        html += "<td class='empty'>–</td>";
      } else {
        var t = (cell[metric] - lo) / span;
        if (spec[2] === -1) t = 1 - t;
        var light = t > 0.62 || t < 0.18;
        html += "<td style=\"background:" + ramp(t) + ";color:" +
          (light ? "#F3F1EC" : "#14201B") + "\" title=\"" +
          titleCase(route) + " vs " + titleCase(col) + " · " +
          cell.plays + " plays\">" + format(cell[metric]) +
          "<br><span style='font-size:9.5px;font-weight:400;opacity:.9'>" +
          cell.plays + "</span></td>";
      }
    });
    html += "</tr>";
  });
  $("mTable").innerHTML = html + "</tbody>";

  $("mRamp").style.background =
    "linear-gradient(90deg," + ramp(0) + "," + ramp(0.5) + "," + ramp(1) + ")";
  $("mLow").textContent = format(spec[2] === -1 ? hi : lo);
  $("mHigh").textContent = format(spec[2] === -1 ? lo : hi);
  $("mCount").textContent = rows.length + " cells, each at least " +
    Math.min.apply(null, rows.map(function (r) { return r.plays; })) + " plays";
  $("mNote").textContent =
    "Small number in each cell is the number of targeted passes behind it. " +
    "Cells thinner than the minimum are not shown at all rather than shown as noise. " +
    (spec[2] === -1 ? "Darker purple is better here, which for this measure means lower."
                    : "Darker purple is higher.");
}
["mMetric", "mSplit"].forEach(function (id) {
  $(id).addEventListener("change", drawMatrix);
});

/* ================= MODEL ================= */

var modelDrawn = false;
function drawModel() {
  if (modelDrawn) return;
  modelDrawn = true;

  /* --- calibration --- */
  var c = $("calib"), g = c.getContext("2d");
  var W = c.width, H = c.height, pad = 54;
  g.clearRect(0, 0, W, H);
  function X(v) { return pad + v * (W - pad - 18); }
  function Y(v) { return H - pad - v * (H - pad - 18); }

  g.strokeStyle = cssVar("--line"); g.lineWidth = 1;
  g.fillStyle = cssVar("--muted");
  g.font = "600 10px " + cssVar("--mono");
  for (var t = 0; t <= 1.0001; t += 0.25) {
    g.globalAlpha = 0.5;
    g.beginPath(); g.moveTo(X(0), Y(t)); g.lineTo(X(1), Y(t)); g.stroke();
    g.globalAlpha = 1;
    g.textAlign = "right"; g.textBaseline = "middle";
    g.fillText((t * 100).toFixed(0) + "%", X(0) - 8, Y(t));
    g.textAlign = "center"; g.textBaseline = "top";
    g.fillText((t * 100).toFixed(0) + "%", X(t), Y(0) + 8);
  }
  /* Perfect calibration. */
  g.strokeStyle = cssVar("--muted");
  g.setLineDash([5, 4]); g.globalAlpha = 0.6;
  g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(1), Y(1)); g.stroke();
  g.setLineDash([]); g.globalAlpha = 1;

  var pts = D.model.calibration;
  g.strokeStyle = cssVar("--field"); g.lineWidth = 2.5;
  g.beginPath();
  pts.forEach(function (p, i) {
    var x = X(p.mean_predicted), y = Y(p.observed);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  });
  g.stroke();
  g.fillStyle = cssVar("--gold");
  pts.forEach(function (p) {
    g.beginPath(); g.arc(X(p.mean_predicted), Y(p.observed), 4.5, 0, 6.284); g.fill();
  });

  g.fillStyle = cssVar("--muted");
  g.textAlign = "center"; g.textBaseline = "bottom";
  g.fillText("PREDICTED CATCH PROBABILITY", W / 2, H - 8);
  g.save();
  g.translate(14, H / 2); g.rotate(-Math.PI / 2);
  g.textBaseline = "top";
  g.fillText("OBSERVED CATCH RATE", 0, 0);
  g.restore();

  /* --- scores --- */
  var rows = [{ name: "Gradient boosting", s: D.model.scores }];
  Object.keys(D.model.baselines).forEach(function (k) {
    rows.push({
      name: k === "base_rate" ? "League base rate" : "Logistic regression",
      s: D.model.baselines[k]
    });
  });
  $("scoreTable").innerHTML =
    "<thead><tr><th>Model</th><th>AUC</th><th>Brier</th><th>Log loss</th><th>Acc</th></tr></thead><tbody>" +
    rows.map(function (r) {
      return "<tr><td>" + r.name + "</td><td>" + fmt(r.s.roc_auc, 3) + "</td><td>" +
        fmt(r.s.brier, 4) + "</td><td>" + fmt(r.s.log_loss, 4) + "</td><td>" +
        fmt(r.s.accuracy, 3) + "</td></tr>";
    }).join("") +
    "<tr><td colspan='5' style='text-align:left;color:var(--muted);font-size:11.5px;" +
    "padding-top:10px'>" + D.model.features + " features, " + D.model.folds.length +
    " folds grouped by game, base rate " + pct(D.model.baseRate, 1) +
    "</td></tr></tbody>";

  /* --- importance --- */
  var ic = $("imp"), ig = ic.getContext("2d");
  var IW = ic.width, IH = ic.height, left = 210;
  ig.clearRect(0, 0, IW, IH);
  var imp = D.model.importance.slice(0, 12);
  var max = Math.max.apply(null, imp.map(function (r) { return r.importance; }));
  var barH = Math.min(24, (IH - 26) / imp.length - 6);
  imp.forEach(function (r, i) {
    var y = 10 + i * ((IH - 26) / imp.length);
    var w = Math.max(1, (r.importance / max) * (IW - left - 54));
    ig.fillStyle = cssVar("--field");
    ig.fillRect(left, y, w, barH);
    ig.fillStyle = cssVar("--ink");
    ig.font = "500 11px " + cssVar("--sans");
    ig.textAlign = "right"; ig.textBaseline = "middle";
    ig.fillText(prettyFeature(r.feature), left - 10, y + barH / 2);
    ig.fillStyle = cssVar("--muted");
    ig.font = "600 10px " + cssVar("--mono");
    ig.textAlign = "left";
    ig.fillText(r.importance.toFixed(3), left + w + 7, y + barH / 2);
  });
}

var FEATURE_NAMES = {
  coverage_position_advantage_yd: "Defender's edge to the ball",
  target_dist_to_spot_at_throw_yd: "Receiver to landing spot",
  coverage_off_bearing_at_throw_deg: "Defender turned off the ball",
  nearest_defender_to_spot_yd: "Nearest defender to the spot",
  target_speed_at_throw_yps: "Receiver speed at release",
  land_sideline_dist_yd: "Ball to the sideline",
  separation_at_throw_yd: "Separation at the throw",
  air_yards_yd: "Air yards",
  coverage_speed_at_throw_yps: "Defender speed at release",
  throw_distance_yd: "Throw distance",
  target_depth_yd: "Receiver depth",
  target_sideline_dist_yd: "Receiver to the sideline",
  air_time_s: "Time in the air",
  defenders_near_spot: "Defenders near the spot",
  route_of_targeted_receiver: "Route",
  team_coverage_man_zone: "Man or zone",
  team_coverage_type: "Coverage shell",
  target_position: "Receiver position",
  target_height_in: "Receiver height",
  target_weight_lb: "Receiver weight",
  n_route_runners: "Route runners",
  n_coverage_defenders: "Coverage defenders",
  offense_formation: "Formation",
  receiver_alignment: "Receiver alignment",
  pass_location_type: "Pass location",
  down: "Down",
  yards_to_go: "Yards to go",
  target_accel_at_throw_yps2: "Receiver acceleration at release",
  coverage_dist_to_spot_at_throw_yd: "Defender to landing spot"
};
function prettyFeature(f) { return FEATURE_NAMES[f] || titleCase(f); }

/* ---------- go ---------- */
applyFilters();
drawMatrix();
window.addEventListener("resize", function () { if (geo) drawField(); });
})();
