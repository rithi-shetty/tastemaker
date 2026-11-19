"""Guarantees the experiments rely on."""
from dataclasses import replace

import numpy as np
import pytest

from tastemaker import experiments as ex
from tastemaker import policies as P
from tastemaker import simulate as sim
from tastemaker.config import BASE, N_CAT

SMALL = replace(BASE, n_users=300, days=14)


@pytest.fixture(scope="module")
def world():
    return sim.build_world(SMALL)


def test_world_shape(world):
    cata, users = world
    assert len(cata.cat) == N_CAT * SMALL.items_per_category
    assert users.taste.shape == (SMALL.n_users, N_CAT)
    k = users.liked.sum(1)
    assert (k >= SMALL.interests_min).all() and (k <= SMALL.interests_max).all()


@pytest.mark.parametrize("name", [p.name for p in P.ALL])
def test_every_policy_runs_and_counts_add_up(world, name):
    r = sim.run(P.make(name), SMALL, world=world)
    d = r.daily
    assert d["shown"].sum() > 0
    assert d["taps"].sum() <= d["shown"].sum()
    assert d["saves"].sum() <= d["taps"].sum()
    assert (np.diff(d["active"]) <= 0).all(), "users who switch the feed off never come back"
    assert 0 <= r.final["ctr"] <= 1


def test_runs_are_reproducible(world):
    a = sim.run(P.make("tastemaker"), SMALL, world=world).final
    b = sim.run(P.make("tastemaker"), SMALL, world=world).final
    assert a["saves_per_user"] == b["saves_per_user"] and a["retention_end"] == b["retention_end"]


def test_runs_do_not_mutate_the_shared_world(world):
    _, users = world
    before = users.taste.copy()
    sim.run(P.make("greedy"), replace(SMALL, tastes_drift=True), world=world)
    assert np.array_equal(users.taste, before)


def test_chosen_card_is_in_the_chosen_category(world):
    cata, _ = world
    pol = P.make("thompson")
    pol.reset(SMALL.n_users, len(cata.cat), cata, np.random.default_rng(0))
    log = sim.new_log(SMALL.n_users, len(cata.cat), SMALL.fatigue_window)
    idx = np.arange(SMALL.n_users)
    cats = np.random.default_rng(1).integers(0, N_CAT, len(idx))
    items = pol.pick_item(log, idx, cats)
    assert (cata.cat[items] == cats).all()


def test_unseen_cards_come_first(world):
    cata, _ = world
    pol = P.make("greedy")
    pol.reset(SMALL.n_users, len(cata.cat), cata, np.random.default_rng(0))
    log = sim.new_log(SMALL.n_users, len(cata.cat), SMALL.fatigue_window)
    idx = np.arange(5)
    cats = np.zeros(5, int)
    first = pol.pick_item(log, idx, cats)
    log.seen[idx, first] = True
    second = pol.pick_item(log, idx, cats)
    assert (first != second).all()


def test_no_prior_starts_from_a_flat_belief(world):
    cata, _ = world
    pol = P.make("thompson", prior_strength=0)
    pol.reset(SMALL.n_users, len(cata.cat), cata, np.random.default_rng(0))
    log = sim.new_log(SMALL.n_users, len(cata.cat), SMALL.fatigue_window)
    a, b = pol.posterior(log, np.arange(3))
    assert (a == 1).all() and (b == 1).all()


def test_diversity_guardrail_adds_variety(world):
    plain = sim.run(P.make("tastemaker", diversity_penalty=0), SMALL, world=world).final
    capped = sim.run(P.make("tastemaker", diversity_penalty=2.0), SMALL, world=world).final
    assert capped["category_entropy"] > plain["category_entropy"]


def test_patient_users_feel_no_fatigue(world):
    tired = sim.run(P.make("greedy"), SMALL, world=world).final
    patient = sim.run(P.make("greedy"), replace(SMALL, patient_users=True), world=world).final
    assert patient["retention_end"] > tired["retention_end"]


def test_policies_never_see_hidden_state():
    hidden = {"taste", "liked", "offset", "quality", "annoy"}
    import inspect
    src = inspect.getsource(P)
    assert not any(f".{h}" in src for h in hidden - {"quality"})
    assert "quality" not in src.replace("quality so the catalogue", "")


def test_short_vs_long_finds_the_crossover():
    cmp = {"curves": {n: {"cum_saves": list(np.linspace(0, v, 60)), "ctr": [0.2] * 60, "active": [1.0] * 60}
                      for n, v in (("greedy", 30), ("thompson", 31), ("tastemaker", 32))}}
    cmp["curves"]["greedy"]["cum_saves"] = [min(1.0 * d, 30) for d in range(60)]
    out = ex.short_vs_long(cmp)
    assert out["tests"]["1 week"]["winner_saves"] == "greedy"
    assert out["overtake_day"] is not None
