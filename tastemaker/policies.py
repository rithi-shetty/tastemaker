"""Six ways to choose the next card.

Each policy picks a category for every active user, then the best card in
that category the user has not seen (by population tap rate, with a little
noise so the catalogue keeps getting explored). They differ only in how they
pick the category, which is where personalisation lives:

  Random        any category, uniformly
  Popular       the most-tapped cards overall; no personalisation
  Greedy        the user's highest estimated tap rate, always
  Epsilon       Greedy, but 10% of the time a random category
  Thompson      sample each category's tap rate from its posterior, pick the
                highest; explores in proportion to uncertainty
  Tastemaker    Thompson on a value score (taps, saves, dismisses) with a
                diversity guardrail that stops one category filling the feed

All personalised policies start new users from the population's tap rates
(the "prior"), which is how they handle cold start.
"""
from __future__ import annotations

import numpy as np

from .config import N_CAT


class Policy:
    name = "policy"
    label = "Policy"

    def __init__(self, prior_strength: float = 4.0):
        self.m = prior_strength

    def reset(self, U, I, cata, rng):
        self.cata, self.rng, self.I = cata, rng, I
        self.item_cat = cata.cat
        self.cat_mask = np.stack([cata.cat == c for c in range(N_CAT)])   # (C, I)
        self.pop_c = np.full(N_CAT, 0.08)                                  # population tap rate per category

    # ---- shared helpers
    def item_scores(self, log):
        # Population tap rate per item, shrunk towards its category's rate.
        prior = self.pop_c[self.item_cat]
        return (log.item_tap + 10 * prior) / (log.item_show + 10)

    def pick_item(self, log, idx, cats):
        """Best unseen card in each chosen category (seen cards only if the category is exhausted)."""
        if getattr(self, "_jitter_day", None) != log.day:      # small daily shuffle among near-equal cards
            self._jitter, self._jitter_day = self.rng.normal(0, 0.01, self.I), log.day
        s = (self.item_scores(log) + self._jitter)[None, :] - 1.0 * log.seen[idx]
        if cats is not None:
            s = np.where(self.cat_mask[cats], s, -np.inf)
        return np.argmax(s, axis=1)

    def posterior(self, log, idx):
        """Beta posterior over each user's tap rate per category, centred on the population rate."""
        if self.m == 0:                      # no population prior: every user starts from scratch
            return log.tap[idx] + 1.0, log.show[idx] - log.tap[idx] + 1.0
        a = log.tap[idx] + self.m * self.pop_c
        b = log.show[idx] - log.tap[idx] + self.m * (1 - self.pop_c)
        return a, b

    def choose(self, log, idx):
        return self.pick_item(log, idx, self.category(log, idx))

    def category(self, log, idx):
        raise NotImplementedError

    def update(self, log, idx, item, tap, save, dismiss):
        pass

    def end_of_day(self, log):
        tot_show = log.show.sum(0)
        self.pop_c = (log.tap.sum(0) + 8 * 0.08) / (tot_show + 8)


class Random(Policy):
    name, label = "random", "Random"

    def category(self, log, idx):
        return self.rng.integers(0, N_CAT, len(idx))


class Popular(Policy):
    name, label = "popular", "Popular"

    def choose(self, log, idx):
        return self.pick_item(log, idx, None)


class Greedy(Policy):
    name, label = "greedy", "Greedy (max taps)"

    def category(self, log, idx):
        a, b = self.posterior(log, idx)
        return np.argmax(a / (a + b) + self.rng.normal(0, 1e-6, a.shape), axis=1)


class Epsilon(Greedy):
    name, label = "epsilon", "Epsilon-greedy"

    def __init__(self, eps: float = 0.10, **kw):
        super().__init__(**kw)
        self.eps = eps

    def category(self, log, idx):
        c = super().category(log, idx)
        explore = self.rng.random(len(idx)) < self.eps
        return np.where(explore, self.rng.integers(0, N_CAT, len(idx)), c)


class Thompson(Policy):
    name, label = "thompson", "Thompson sampling"

    def category(self, log, idx):
        a, b = self.posterior(log, idx)
        return np.argmax(self.rng.beta(a, b), axis=1)


class Tastemaker(Policy):
    """Thompson sampling on long-term value, with a diversity guardrail.

    value = sampled tap rate × (1 + save_weight × save rate) − dismiss_weight × dismiss rate
            − diversity_penalty × repeats beyond the allowance in the last 12 cards

    The guardrail is a product rule, not a learned model: the feed cannot see
    boredom directly, so it caps monotony on purpose and the experiments
    measure what that costs in taps and buys in retention.
    """
    name, label = "tastemaker", "Tastemaker"

    def __init__(self, diversity_penalty: float = 0.25, allowance: int = 3, save_weight: float = 1.0,
                 dismiss_weight: float = 0.5, **kw):
        super().__init__(**kw)
        self.lam, self.allow = diversity_penalty, allowance
        self.ws, self.wd = save_weight, dismiss_weight

    def category(self, log, idx):
        a, b = self.posterior(log, idx)
        theta = self.rng.beta(a, b)
        save_rate = (log.save[idx] + 1) / (log.tap[idx] + 4)
        dismiss_rate = (log.dismiss[idx] + 0.2) / (log.show[idx] + 4)
        value = theta * (1 + self.ws * save_rate) - self.wd * dismiss_rate * theta.mean(1, keepdims=True)
        rec = log.recent[idx]
        repeats = np.stack([(rec == c).sum(1) for c in range(N_CAT)], axis=1)
        value = value - self.lam * np.maximum(0, repeats - self.allow) * theta.mean(1, keepdims=True)
        return np.argmax(value, axis=1)


ALL = [Random, Popular, Greedy, Epsilon, Thompson, Tastemaker]
LABELS = {p.name: p.label for p in ALL}


def make(name: str, **kw) -> Policy:
    return {p.name: p for p in ALL}[name](**kw)
