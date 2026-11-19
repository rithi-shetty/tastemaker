"""README charts, drawn from reports/results.json."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, LINE = "#1F2937", "#6B7280", "#E5E7EB"
COL = {"random": "#9CA3AF", "popular": "#6B7280", "greedy": "#DC2626", "epsilon": "#F59E0B",
       "thompson": "#2563EB", "tastemaker": "#7C3AED"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.8,
})


def _save(fig, out, name):
    fig.savefig(out / name, facecolor="white")
    plt.close(fig)


def crossover(R, out):
    c, lab = R["compare"]["curves"], R["meta"]["labels"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for n in ("greedy", "thompson", "tastemaker", "popular"):
        d = range(1, len(c[n]["cum_saves"]) + 1)
        a1.plot(d, c[n]["cum_saves"], color=COL[n], lw=2.2 if n in ("greedy", "tastemaker") else 1.4, label=lab[n])
        a2.plot(d, [x * 100 for x in c[n]["active"]], color=COL[n], lw=2.2 if n in ("greedy", "tastemaker") else 1.4)
    od = R["short_vs_long"]["overtake_day"]
    if od:
        a1.axvline(od, color=MUTED, ls=":", lw=1)
        a1.text(od + 1, a1.get_ylim()[1] * 0.08, f"day {od}", color=MUTED, fontsize=9)
    a1.set_title("Saves per user, cumulative")
    a1.set_xlabel("Day")
    a2.set_title("Users who still have the feed on (%)")
    a2.set_xlabel("Day")
    a1.legend(frameon=False, fontsize=9)
    _save(fig, out, "crossover.png")


def policies(R, out):
    T, lab = R["compare"]["table"], R["meta"]["labels"]
    order = ["random", "popular", "greedy", "epsilon", "thompson", "tastemaker"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (k, title, fmt) in zip(axes, (("ctr", "Tap rate", lambda v: f"{v:.1%}"),
                                         ("saves_per_user", "Saves per user, 60 days", lambda v: f"{v:.1f}"),
                                         ("retention_end", "Feed still on at day 60", lambda v: f"{v:.0%}"))):
        vals = [T[n][k]["mean"] for n in order]
        ax.barh([lab[n] for n in order], vals, color=[COL[n] for n in order])
        for i, v in enumerate(vals):
            ax.text(v, i, " " + fmt(v), va="center", fontsize=9, color=INK)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xticks([])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_xlim(0, max(vals) * 1.3)
    _save(fig, out, "policies.png")


def cold_start(R, out):
    C = R["cold_start"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    cols = ["#7C3AED", "#F59E0B", "#6B7280"]
    import numpy as np
    for (lab, ys), col in zip(C.items(), cols):
        sm = np.convolve(ys, np.ones(5) / 5, mode="valid")          # 5-card rolling average
        ax.plot(range(3, len(sm) + 3), sm * 100, color=col, lw=2, label=lab)
    ax.set_xlabel("Card number in the user's history")
    ax.set_ylabel("Tap rate (%)")
    ax.set_title("Cold start: how fast the feed learns a new user")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(1, -0.22, "5-card rolling average", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    _save(fig, out, "cold_start.png")


def dial(R, out):
    D = R["diversity_dial"]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = [d["penalty"] for d in D]
    ax.plot(x, [d["retention_end"] * 100 for d in D], color="#7C3AED", marker="o", lw=2, label="Feed still on at day 60 (%)")
    ax.plot(x, [d["ctr"] * 100 for d in D], color="#DC2626", marker="s", lw=2, label="Tap rate (%)")
    ax.plot(x, [d["saves_per_user"] for d in D], color="#059669", marker="^", lw=2, label="Saves per user")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xticks(x, [str(v) for v in x])
    ax.set_xlabel("Diversity penalty")
    ax.set_title("The diversity dial: retention rises, taps barely move")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "diversity_dial.png")


def stress(R, out):
    S = R["stress"]
    if not S:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    labels = [s["label"] for s in S][::-1]
    y = range(len(S))
    for off, n in ((-0.27, "greedy"), (0, "thompson"), (0.27, "tastemaker")):
        ax.barh([i + off for i in y], [s[n]["retention_end"] * 100 for s in S][::-1], 0.26, color=COL[n],
                label=R["meta"]["labels"][n])
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("Feed still on at day 60 (%)")
    ax.set_title("Stress tests")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    _save(fig, out, "stress.png")


def journey(J: dict, cats: list, out: Path, who: str = "Kabir: two interests"):
    """The README hero: one typical person's four weeks under each feed."""
    from matplotlib.patches import Rectangle
    P = J["personas"][who]
    days, per = J["days"], J["cards_per_day"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.4), sharey=True)
    for ax, pol, title in zip(axes, ("greedy", "tastemaker"), ("Greedy (max taps)", "Tastemaker")):
        ev = [e for e in P[pol] if not e.get("off")]
        off = next((e["d"] for e in P[pol] if e.get("off")), None)
        for e in ev:
            col = cats[e["c"]][2]
            ax.add_patch(Rectangle((e["s"] + 0.08, days - 1 - e["d"] + 0.08), 0.84, 0.84, color=col,
                                   alpha=1.0 if e["t"] else 0.18, lw=0))
            if e["v"]:
                ax.text(e["s"] + 0.5, days - 1 - e["d"] + 0.47, "★", ha="center", va="center", color="white", fontsize=8)
        if off is not None:
            y = days - 1 - off
            ax.axhline(y, color="#DC2626", lw=1.6)
            ax.text(per / 2, y - 0.6, f"Switched the feed off after day {off + 1}", ha="center", va="top",
                    color="#DC2626", fontsize=10, fontweight="bold")
        taps = sum(e["t"] for e in ev)
        saves = sum(e["v"] for e in ev)
        status = ", still using the feed" if off is None else ", then gone"
        ax.set_title(f"{title}\n{taps} taps, {saves} saves{status}", fontsize=11)
        ax.set_xlim(0, per)
        ax.set_ylim(0, days)
        ax.set_xticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks([days - 1 - d + 0.5 for d in range(0, days, 7)], [f"Day {d + 1}" for d in range(0, days, 7)])
    liked = ", ".join(P["likes"])
    fig.suptitle(f"Same person, two feeds: four weeks of lock-screen cards (likes {liked})", x=0.02, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    used = sorted({e["c"] for pol in ("greedy", "tastemaker") for e in P[pol] if "c" in e})
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cats[c][2], label=cats[c][0]) for c in used], loc="lower center", ncol=min(6, len(used)),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.02, -0.11, "Each row is a day of six cards. Solid = tapped, faded = ignored, ★ = saved. Simulated person, "
             "typical of the most common difference between the two feeds.", fontsize=8.5, color=MUTED)
    _save(fig, out, "journey.png")


def draw_all(R, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for fn in (crossover, policies, cold_start, dial, stress):
        fn(R, out)
