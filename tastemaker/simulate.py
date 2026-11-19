"""The simulated feed: users, cards, and what happens when a policy shows a card.

Everything is vectorised over users, so a 30-day month for 4,000 users
(about 700,000 cards) runs in a couple of seconds per policy.

Policies only ever see what a real feed would log: which card was shown,
and whether it was tapped, saved or dismissed. Tastes, annoyance and the
churn model are hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BASE, N_CAT, World


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Catalogue:
    cat: np.ndarray       # (I,) category of each item
    brand: np.ndarray     # (I,) brand id
    quality: np.ndarray   # (I,) intrinsic appeal, hidden from policies


@dataclass
class Users:
    taste: np.ndarray     # (U, C) extra tap log-odds per category, hidden
    liked: np.ndarray     # (U, C) bool, hidden
    offset: np.ndarray    # (U,) how tap-happy the user is, hidden
    n_interests: np.ndarray


def build_world(w: World = BASE, seed: int | None = None):
    rng = np.random.default_rng(w.seed if seed is None else seed)
    n_items = N_CAT * w.items_per_category
    cat = np.repeat(np.arange(N_CAT), w.items_per_category)
    brand_pop = 1.0 / np.arange(1, w.n_brands + 1) ** 1.1
    brand = rng.choice(w.n_brands, n_items, p=brand_pop / brand_pop.sum())
    quality = rng.normal(0, w.quality_sd, n_items)
    cata = Catalogue(cat, brand, quality)
    users = draw_users(w, rng, w.n_users)
    return cata, users


def draw_users(w: World, rng, n: int) -> Users:
    k = rng.integers(w.interests_min, w.interests_max + 1, n)
    # Some categories are more commonly liked than others.
    pop = np.array([1.4, 1.1, 0.9, 1.0, 0.8, 1.2, 0.9, 0.9, 1.3, 1.0, 0.6, 0.7])
    liked = np.zeros((n, N_CAT), bool)
    for u in range(n):
        liked[u, rng.choice(N_CAT, k[u], replace=False, p=pop / pop.sum())] = True
    strength = rng.uniform(0.7, 1.3, (n, N_CAT)) * w.interest_strength
    taste = np.where(liked, strength, 0.0)
    offset = rng.normal(0, 0.35, n)
    return Users(taste, liked, offset, k)


@dataclass
class Log:
    """What the feed can observe, accumulated per user and globally."""
    show: np.ndarray      # (U, C) cards shown per category
    tap: np.ndarray       # (U, C)
    save: np.ndarray      # (U, C)
    dismiss: np.ndarray   # (U, C)
    seen: np.ndarray      # (U, I) bool
    recent: np.ndarray    # (U, W) categories of the last W cards, -1 = empty
    item_show: np.ndarray # (I,)
    item_tap: np.ndarray  # (I,)
    day: int = 0
    slot: int = 0


def new_log(n_users: int, n_items: int, window: int) -> Log:
    z = lambda: np.zeros((n_users, N_CAT))
    return Log(z(), z(), z(), z(), np.zeros((n_users, n_items), bool), np.full((n_users, window), -1),
               np.zeros(n_items), np.zeros(n_items))


def recent_count(log: Log, cats: np.ndarray | None = None) -> np.ndarray:
    """(U, C) how many of the last W cards were from each category."""
    out = np.zeros((log.recent.shape[0], N_CAT))
    for c in range(N_CAT):
        out[:, c] = (log.recent == c).sum(1)
    return out


@dataclass
class Result:
    policy: str
    daily: dict              # per-day arrays
    final: dict              # summary metrics
    journeys: dict           # persona id -> list of card events


def run(policy, w: World = BASE, seed: int = 0, personas: dict | None = None, world=None) -> Result:
    """Run one policy for w.days days. Same seed = same users, same coin flips across policies."""
    cata, users = world if world is not None else build_world(w)
    users = Users(users.taste.copy(), users.liked.copy(), users.offset, users.n_interests)  # runs never share state
    rng = np.random.default_rng(10_000 + seed)
    U, I = users.taste.shape[0], len(cata.cat)
    log = new_log(U, I, w.fatigue_window)
    active = np.ones(U, bool)
    annoy = np.zeros(U)
    churn_day = np.full(U, -1)
    last_tap_day = np.full(U, -100)
    daily = {k: np.zeros(w.days) for k in ("active", "shown", "taps", "saves", "dismisses", "churned")}
    K = 60                                   # cold-start curve: tap rate by how many cards a user has seen
    n_seen = np.zeros(U, int)
    by_index = {"shown": np.zeros(K), "taps": np.zeros(K)}
    journeys = {pid: [] for pid in (personas or {})}
    pid_of = {u: pid for pid, u in (personas or {}).items()}
    policy.reset(U, I, cata, rng)
    for day in range(w.days):
        log.day = day
        if w.tastes_drift and day == w.days // 2:
            # Half the users discover new interests; old ones fade.
            move = rng.random(U) < 0.5
            fresh = draw_users(w, rng, int(move.sum()))
            users.taste[move], users.liked[move] = fresh.taste, fresh.liked
        for slot in range(w.cards_per_day):
            log.slot = slot
            idx = np.flatnonzero(active)
            if not len(idx):
                break
            item = policy.choose(log, idx)
            c = cata.cat[item]
            # --- the user's hidden response
            repeats = (log.recent[idx] == c[:, None]).sum(1)
            extra = np.maximum(0, repeats - w.fatigue_free)
            seen_before = log.seen[idx, item]
            liked = users.liked[idx, c]
            logit = (w.base_logit + users.offset[idx] + users.taste[idx, c] + cata.quality[item]
                     - w.fatigue_ctr * extra - 1.0 * seen_before)
            r = rng.random((3, len(idx)))
            tap = r[0] < sigmoid(logit)
            save = tap & (r[1] < sigmoid(w.save_logit + np.where(liked, 0.8, -1.6) + cata.quality[item]))
            dismiss = ~tap & (r[2] < sigmoid(w.dismiss_logit + np.where(liked, -2.0, 1.2)))
            fatigue = 0.0 if w.patient_users else w.fatigue_annoy * extra
            annoy[idx] += fatigue + w.irrelevant_annoy * ~liked + w.dismiss_annoy * dismiss
            # --- what the feed logs
            log.show[idx, c] += 1
            log.tap[idx, c] += tap
            log.save[idx, c] += save
            log.dismiss[idx, c] += dismiss
            log.seen[idx, item] = True
            log.recent[idx] = np.column_stack([log.recent[idx, 1:], c])
            np.add.at(log.item_show, item, 1)
            np.add.at(log.item_tap, item, tap)
            last_tap_day[idx[tap]] = day
            k = n_seen[idx]
            early = k < K
            np.add.at(by_index["shown"], k[early], 1)
            np.add.at(by_index["taps"], k[early], tap[early])
            n_seen[idx] += 1
            daily["shown"][day] += len(idx)
            daily["taps"][day] += tap.sum()
            daily["saves"][day] += save.sum()
            daily["dismisses"][day] += dismiss.sum()
            policy.update(log, idx, item, tap, save, dismiss)
            for j in np.flatnonzero(np.isin(idx, list(pid_of))) if pid_of else []:
                u = idx[j]
                journeys[pid_of[u]].append({"d": day, "s": slot, "c": int(c[j]), "i": int(item[j]),
                                            "t": int(tap[j]), "v": int(save[j]), "x": int(dismiss[j]),
                                            "a": round(float(annoy[u]), 3)})
        # --- end of day: some users switch the feed off
        daily["active"][day] = active.sum()
        p_off = np.clip(w.churn_base + w.churn_per_annoy * annoy, 0, 1)
        off = active & (rng.random(U) < p_off)
        churn_day[off] = day
        active &= ~off
        annoy *= w.annoy_decay
        daily["churned"][day] = off.sum()
        for pid, u in (personas or {}).items():
            if off[u]:
                journeys[pid].append({"d": day, "off": 1})
        policy.end_of_day(log)
    final = summarise(daily, log, users, cata, active, last_tap_day, w)
    final["churn_day"] = churn_day
    final["by_index"] = by_index
    return Result(policy.name, daily, final, journeys)


def summarise(daily, log, users, cata, active, last_tap_day, w: World) -> dict:
    U = len(active)
    shown, taps = daily["shown"].sum(), daily["taps"].sum()
    wk1 = slice(0, 7)
    # Category diversity: entropy of each user's shown-category mix, averaged.
    p = log.show / np.maximum(log.show.sum(1, keepdims=True), 1)
    ent = -(p * np.log(np.where(p > 0, p, 1))).sum(1)
    # Brand exposure concentration (Gini over brands).
    exp_brand = np.bincount(cata.brand, weights=log.item_show, minlength=int(cata.brand.max()) + 1)
    g = np.sort(exp_brand)
    n = len(g)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(g) / (n * g.sum())) if g.sum() else 0.0
    engaged = active & (last_tap_day >= w.days - 7)   # tapped in the final week
    return {
        "users": U,
        "ctr": float(taps / shown),
        "ctr_week1": float(daily["taps"][wk1].sum() / daily["shown"][wk1].sum()),
        "saves_per_user": float(daily["saves"].sum() / U),
        "saves_week1_per_user": float(daily["saves"][wk1].sum() / U),
        "taps_per_user": float(taps / U),
        "dismiss_rate": float(daily["dismisses"].sum() / shown),
        "retention_end": float(active.mean()),
        "retention_d30": float(daily["active"][29] / U) if w.days > 30 else float(active.mean()),
        "retention_d7": float(daily["active"][7] / U) if w.days > 7 else float(active.mean()),
        "engaged_end": float(engaged.mean()),
        "saves_final_week_per_user": float(daily["saves"][-7:].sum() / U),
        "category_entropy": float(ent.mean() / np.log(N_CAT)),
        "brand_gini": gini,
        "liked_share": float((log.show * users.liked).sum() / log.show.sum()),
    }
