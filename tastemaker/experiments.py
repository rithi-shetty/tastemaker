"""Experiments that test the feed's design choices instead of assuming them.

  compare          Six policies on the same users: taps, saves, retention, diversity
  short_vs_long    Would a 1-week or 4-week test pick the right policy?
  cold_start       How fast does the feed learn a new user, with and without a prior?
  diversity_dial   What the diversity guardrail trades: taps vs retention
  robustness       Is the result luck from one simulated world?
  stress_tests     Worlds where the assumptions behind Tastemaker do not hold
  test_power       How many users an A/B test needs to see the difference
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import policies as P
from . import simulate as sim
from .config import BASE, SCENARIOS, World

MAIN = ["random", "popular", "greedy", "epsilon", "thompson", "tastemaker"]
KEY = ["greedy", "thompson", "tastemaker"]
METRICS = ["ctr", "ctr_week1", "saves_per_user", "saves_week1_per_user", "saves_final_week_per_user",
           "retention_d7", "retention_d30", "retention_end", "engaged_end", "category_entropy",
           "dismiss_rate", "brand_gini", "liked_share"]


def run_all(w: World = BASE, names=MAIN, seed: int = 0, personas=None, **kw):
    world = sim.build_world(w, seed=w.seed + seed)
    return {n: sim.run(P.make(n, **kw.get(n, {})), w, seed=seed, personas=personas, world=world) for n in names}


def _mean_sd(rows: list[dict]) -> dict:
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows]))}
            for k in METRICS}


def compare(w: World = BASE, seeds=range(3)) -> dict:
    """Headline table plus the daily curves used in the charts (from the first seed)."""
    per = {n: [] for n in MAIN}
    curves = {}
    for s in seeds:
        res = run_all(w, MAIN, seed=s)
        for n, r in res.items():
            per[n].append(r.final)
            if s == seeds[0]:
                curves[n] = {"active": (r.daily["active"] / w.n_users).tolist(),
                             "cum_saves": (np.cumsum(r.daily["saves"]) / w.n_users).tolist(),
                             "ctr": (r.daily["taps"] / np.maximum(r.daily["shown"], 1)).tolist()}
    table = {n: _mean_sd(rows) for n, rows in per.items()}
    return {"table": table, "curves": curves, "seeds": len(list(seeds))}


def short_vs_long(cmp: dict) -> dict:
    """Which policy each test length would pick, using the daily curves."""
    c = cmp["curves"]
    out = {}
    for label, day in (("1 week", 7), ("2 weeks", 14), ("4 weeks", 28), ("8 weeks", 59)):
        by_saves = max(KEY, key=lambda n: c[n]["cum_saves"][day - 1])
        by_ctr = max(KEY, key=lambda n: np.mean(c[n]["ctr"][:day]))
        out[label] = {"day": day, "winner_saves": by_saves, "winner_ctr": by_ctr,
                      "saves": {n: c[n]["cum_saves"][day - 1] for n in KEY},
                      "active": {n: c[n]["active"][day - 1] for n in KEY}}
    g, t = np.array(c["greedy"]["cum_saves"]), np.array(c["tastemaker"]["cum_saves"])
    ahead = np.flatnonzero(t > g)
    overtake = None
    for d in ahead:            # first day after which Tastemaker stays ahead
        if (t[d:] > g[d:]).all():
            overtake = int(d) + 1
            break
    return {"tests": out, "overtake_day": overtake}


def cold_start(w: World = BASE, seed: int = 0) -> dict:
    """Tap rate on a user's n-th card: population prior vs no prior vs no personalisation."""
    world = sim.build_world(w, seed=w.seed + seed)
    out = {}
    for label, pol in (("With population prior", P.make("tastemaker")),
                       ("No prior (learn each user from scratch)", P.make("tastemaker", prior_strength=0)),
                       ("Popular (no personalisation)", P.make("popular"))):
        b = sim.run(pol, w, seed=seed, world=world).final["by_index"]
        out[label] = (b["taps"] / np.maximum(b["shown"], 1)).tolist()
    return out


def diversity_dial(w: World = BASE, penalties=(0, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0), seeds=range(2)) -> list[dict]:
    rows = []
    for lam in penalties:
        f = [sim.run(P.make("tastemaker", diversity_penalty=lam), w, seed=s,
                     world=sim.build_world(w, seed=w.seed + s)).final for s in seeds]
        rows.append({"penalty": lam, **{k: float(np.mean([x[k] for x in f])) for k in METRICS}})
    return rows


def robustness(w: World = BASE, seeds=range(10)) -> dict:
    rows = []
    for s in seeds:
        res = run_all(replace(w, seed=1000 + s), KEY, seed=s)
        rows.append({n: {k: res[n].final[k] for k in ("saves_per_user", "retention_end", "ctr", "ctr_week1",
                                                       "saves_week1_per_user")} for n in KEY})
    win = lambda a, b, k: int(sum(r[a][k] > r[b][k] for r in rows))
    return {"n": len(rows), "runs": rows,
            "tastemaker_beats_greedy_saves": win("tastemaker", "greedy", "saves_per_user"),
            "tastemaker_beats_greedy_retention": win("tastemaker", "greedy", "retention_end"),
            "greedy_beats_tastemaker_week1_saves": win("greedy", "tastemaker", "saves_week1_per_user"),
            "tastemaker_beats_thompson_saves": win("tastemaker", "thompson", "saves_per_user"),
            "tastemaker_beats_thompson_retention": win("tastemaker", "thompson", "retention_end")}


def stress_tests() -> list[dict]:
    out = []
    for name, w in SCENARIOS.items():
        res = run_all(w, KEY)
        out.append({"scenario": name, **{n: {k: res[n].final[k] for k in ("saves_per_user", "retention_end", "ctr")}
                                         for n in KEY}})
    return out


def test_power(cmp: dict, w: World = BASE, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Users per arm to detect day-30 retention differences (two-proportion z-test)."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    T = cmp["table"]

    def n_for(a, b, k="retention_d30"):
        p1, p2 = T[a][k]["mean"], T[b][k]["mean"]
        d = abs(p1 - p2)
        return int(np.ceil(z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d ** 2)) if d > 0 else None

    return {"greedy_vs_tastemaker": n_for("greedy", "tastemaker"),
            "thompson_vs_tastemaker": n_for("thompson", "tastemaker"),
            "d30": {n: T[n]["retention_d30"]["mean"] for n in KEY}}


def journeys(w: World = BASE, n_days: int = 28) -> dict:
    """Card-by-card journeys for the demo, one example person per interest group.

    Every user is simulated under both feeds for four weeks. Most people end
    up the same either way, so for each group the demo shows the most common
    *difference* (left the click-maximiser, stayed with Tastemaker) and reports
    how often it happens and how often the reverse happens. Within that
    pattern it picks the person with the most typical tap counts.
    """
    from .config import CATEGORIES
    cata, users = sim.build_world(w)
    short = replace(w, days=n_days)
    every = {f"u{u}": u for u in range(users.taste.shape[0])}
    res = {n: sim.run(P.make(n), short, world=(cata, users), personas=every) for n in ("greedy", "tastemaker")}
    off = {n: np.array([next((e["d"] for e in res[n].journeys[f"u{u}"] if e.get("off")), -1)
                        for u in range(len(every))]) for n in res}
    taps = {n: np.array([sum(e.get("t", 0) for e in res[n].journeys[f"u{u}"]) for u in range(len(every))]) for n in res}
    out = {"personas": {}, "days": n_days, "cards_per_day": w.cards_per_day}
    k = users.n_interests
    for label, group in (("Asha: loves one thing", k == 1), ("Kabir: two interests", k == 2),
                         ("Meera: three interests", k == 3)):
        g_off, t_off = off["greedy"][group], off["tastemaker"][group]
        outcomes = {"greedy_off_tastemaker_on": float(((g_off >= 0) & (t_off < 0)).mean()),
                    "both_on": float(((g_off < 0) & (t_off < 0)).mean()),
                    "both_off": float(((g_off >= 0) & (t_off >= 0)).mean()),
                    "tastemaker_off_greedy_on": float(((g_off < 0) & (t_off >= 0)).mean())}
        common = "greedy_off_tastemaker_on"
        ids = np.flatnonzero(group)
        match = {"greedy_off_tastemaker_on": (off["greedy"] >= 7) & (off["tastemaker"] < 0),
                 "both_on": (off["greedy"] < 0) & (off["tastemaker"] < 0),
                 "both_off": (off["greedy"] >= 0) & (off["tastemaker"] >= 0),
                 "tastemaker_off_greedy_on": (off["greedy"] < 0) & (off["tastemaker"] >= 0)}[common][ids]
        cand = ids[match] if match.any() else ids
        # Among matching people, the one with the most typical tap counts under both feeds.
        score = sum(abs(taps[n][cand] - np.median(taps[n][ids])) for n in res)
        u = int(cand[np.argmin(score)])
        out["personas"][label] = {"likes": [CATEGORIES[c][0] for c in np.flatnonzero(users.liked[u])],
                                  "group_size": int(group.sum()), "pattern": common, "pattern_share": outcomes[common],
                                  "outcomes": outcomes,
                                  **{n: res[n].journeys[f"u{u}"] for n in res}}
    return out
