"""Every assumption behind the simulated feed, in one place.

The world is a lock-screen discovery feed: each time a user unlocks their
phone they see one card (a look, a product, a beauty tip). They can tap it,
save it to shop later, dismiss it as "not interested", or do nothing.
Every few days they decide whether to keep the feed switched on.

Nothing here is real user data. The parameters are chosen so that the
behaviours a discovery product has to handle are present and visible:
strong individual tastes, cold-start users, fatigue from repetition, and
churn that clicks alone do not reveal. The most important assumption, the
one the main result rests on, is marked ASSUMPTION below.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

CATEGORIES = [
    # name, emoji, colour (for the demo)
    ("Ethnic wear", "🥻", "#C2410C"),
    ("Streetwear", "🧢", "#334155"),
    ("Workwear", "👔", "#1D4ED8"),
    ("Athleisure", "🏃", "#059669"),
    ("Party wear", "✨", "#9333EA"),
    ("Footwear", "👟", "#B45309"),
    ("Bags", "👜", "#BE185D"),
    ("Jewellery", "💍", "#CA8A04"),
    ("Skincare", "🧴", "#0E7490"),
    ("Makeup", "💄", "#DB2777"),
    ("Fragrance", "🌸", "#E11D48"),
    ("Home decor", "🪴", "#4D7C0F"),
]
N_CAT = len(CATEGORIES)


@dataclass(frozen=True)
class World:
    seed: int = 7
    n_users: int = 3000
    days: int = 60
    cards_per_day: int = 6          # lock-screen unlocks that show a card
    items_per_category: int = 40
    n_brands: int = 60

    # Users: how many categories they really like, and how strongly.
    interests_min: int = 1
    interests_max: int = 3
    interest_strength: float = 2.6   # extra log-odds of tapping a liked category
    base_logit: float = -3.1         # log-odds of tapping a random card
    quality_sd: float = 0.45         # item-to-item appeal within a category
    save_logit: float = -1.3         # log-odds of saving a tapped card in a liked category
    dismiss_logit: float = -2.4      # log-odds of dismissing an irrelevant card

    # Fatigue: seeing the same category again and again.
    fatigue_window: int = 12          # recent cards a user remembers
    fatigue_free: int = 4             # repeats of one category in the window that do not bother anyone
    fatigue_ctr: float = 0.06         # tap log-odds lost per extra repeat (small)

    # ASSUMPTION: repetition hurts retention far more than it hurts taps.
    # Users keep tapping their favourite category but grow bored of a feed
    # that shows nothing else, and boredom is what makes them switch it off.
    # This is the hidden long-term cost a click-maximising feed cannot see.
    fatigue_annoy: float = 0.012      # annoyance added per extra repeat
    irrelevant_annoy: float = 0.045   # annoyance added by a card from a category they dislike
    dismiss_annoy: float = 0.06       # annoyance added each time they dismiss
    annoy_decay: float = 0.80         # share of annoyance that carries to the next day
    churn_base: float = 0.002         # daily chance of switching the feed off anyway
    churn_per_annoy: float = 0.012    # extra daily switch-off chance per unit of annoyance

    # Personas whose journeys are recorded for the demo.
    tastes_drift: bool = False        # stress test: half the users' interests change halfway
    patient_users: bool = False       # stress test: nobody tires of repetition


BASE = World()

SCENARIOS = {
    "base": BASE,
    "patient_users": replace(BASE, patient_users=True),
    "tastes_drift": replace(BASE, tastes_drift=True),
    "few_interests": replace(BASE, interests_min=1, interests_max=1),
    "broad_interests": replace(BASE, interests_min=4, interests_max=5),
}
SCENARIO_LABELS = {
    "base": "Base case",
    "patient_users": "Nobody tires of repetition",
    "tastes_drift": "Tastes change halfway through",
    "few_interests": "Everyone likes one thing",
    "broad_interests": "Everyone likes many things",
}
