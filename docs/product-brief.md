# Product brief: Tastemaker

## Context

Lock-screen and home-screen feeds show one card at a time to people who did not open an app to shop. Each card either earns a little attention or spends a little patience. The feed has to learn each person's taste from a handful of taps, with no search query to go on, and it can lose the user entirely: switching the feed off takes two taps in settings.

## Users and jobs

| User | Job to be done |
|---|---|
| **Person using the phone** | "Show me things I'd actually like, without turning my lock screen into an ad wall." |
| **Brand or creator** | "Get my products in front of people likely to want them." |
| **Feed product team** | "Grow the number of people who keep the feed on and act on it." |

## Problem

The obvious objective, click-through rate, rewards showing each person more of whatever they tapped last. That works for a week. Then the feed narrows to one category, people get bored, and some switch it off. Clicks fall only slightly, so a click dashboard barely notices. A short A/B test, judged on clicks or early saves, ships the narrowing feed.

## Goals and metrics

| Level | Metric | Why |
|---|---|---|
| **North star** | Engaged users at day 60: feed still on, and tapped at least once in the last week | Captures both value and the cost of annoying people |
| Value | Saves per user (intent to shop) | The commercial outcome for brands |
| Guardrails | Dismiss rate; category variety; brand exposure concentration | Early signs of annoyance and narrowing; fairness to smaller brands |
| Diagnostic only | Tap rate | Useful, but blind to boredom; never the objective on its own |

## Solution

1. **Learn each person's taste with Thompson sampling.** It explores categories in proportion to uncertainty, so it learns faster than "always show the best guess" without wasting cards on random ones.
2. **Start new people from the population** (a prior), so the feed is not starting from zero.
3. **Optimise value, not taps.** Saves count more than taps; "not interested" counts against a category.
4. **A diversity guardrail.** The feed cannot see boredom, so a product rule limits how often one category can appear in the last 12 cards. The penalty is a dial the team can tune, and the experiments show what each setting buys and costs.

## Non-goals

Ranking within a category (uses popularity), creative generation, pricing, ads auction, and anything that needs identity across apps.

## Risks and how the experiments address them

| Risk | Test |
|---|---|
| The boredom assumption is wrong | Stress test "nobody tires of repetition": the guardrail then costs saves, and Greedy wins. This is the first thing a real pilot must check |
| Short tests pick the wrong winner | Short-vs-long analysis; long-running holdout recommended |
| Cold start is slow | Cold-start curve, with and without the population prior |
| Narrowing hurts small brands | Brand exposure concentration (Gini) per policy |

## Launch plan

1. **Measure boredom before building the fix.** On the current feed, check whether people who saw long runs of one category switch the feed off more often than people who did not, holding their taste fixed. If not, the guardrail is not worth its cost.
2. **A/B test for 8 weeks, not 1.** Primary metric: engaged users at day 60. Also report week-1 results, to show the team how misleading they are.
3. **Keep a 5% long-term holdout** on the old feed after launch, so slow effects stay visible.
