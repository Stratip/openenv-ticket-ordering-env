---
title: Ticket Ordering Environment Server
emoji: 🎟️
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ticket Ordering Environment

**Ticket Ordering** is an environment for turning a messy “who shouts loudest” backlog into a **repeatable, auditable ordering** of work items (GitHub issues, Jira tickets, support requests, bugs, etc.) using an **objective scoring rubric**.

It’s especially useful when **QA testers**, **engineering**, and **third-party stakeholders** (customers, partners, vendors) have **contradictory priorities** depending on the situation:

- QA may optimize for *risk reduction* and *release confidence*.
- Third parties may optimize for *their own timelines*, *contracts*, and *visibility*.
- Engineering may optimize for *system health*, *tech debt*, and *delivery cost*.

When incentives diverge, “priority” becomes a negotiation. This project makes priority **measurable** by asking an agent (human or model) to assign scores *relative to a shared criteria string* and provides feedback via a reward function—so you can iterate toward a ranking that’s **consistent**, not political.

## Table of contents

- [Why prioritization breaks](#why-prioritization-breaks)
- [Observation and action (plain English)](#observation-and-action-plain-english)
- [Motivation](#motivation)
- [API surface](#api-surface)
- [Example](#example)
- [Environment Details](#environment-details)
- [Benchmark results](#benchmark-results)
- [Env logic without HTTP](#env-logic-without-http)
- [Project Structure](#project-structure)

## Why prioritization breaks

Teams usually fail at prioritization for one of two reasons:

- **Subjectivity**: two people read the same ticket and give different priorities for valid reasons.
- **Context drift**: the “right” priority changes with release phase, customer escalations, regressions, or risk posture.

An “objective scorer” doesn’t remove judgment—it **structures it**. You get:

- **Comparable decisions**: scores are on the same scale across tickets.
- **Explainability**: the criteria and summaries capture the *why*, not just the number.
- **Repeatability**: rerun ordering when constraints change (release week vs normal week) and measure deltas.
- **Less conflict**: disagreements become “which criterion weights are wrong?” rather than “your priority is wrong.”

## Observation and action (plain English)

### **What you see each step**

- A **partial view** of the episode: not every ticket in one shot.
- **`ordering_criteria`** — shared rubric for the whole episode.
- **`candidate_ticket`** — the item currently under review.
- **`reference_tickets`** — a small anchor set the agent asked for on the prior step.
- **`ticket_heuristics`** — running table of **priority + short summary** per ticket id already touched.
- **`total_tickets`** — size of the backlog to be ordered; note this down as **n** (number of tickets to order).
- **`completed_iterations`** — how many steps have run so far.

### **What you do each step**

- Set **`candidate_priority`** and **`candidate_summary`** for the current candidate.
- Choose **`next_reference_ids`** and **`next_candidate_id`** to steer the **next** observation.
- Set **`end_ordering`** to finish the episode early when the ranking is good enough.
- Otherwise the run stops when **`completed_iterations` reaches `max_steps`** — so no single observation has to materialize all **n** full tickets at once.

## Motivation

In decoder-only transformers, self-attention over context length *L* scales as **O(L²)** per layer (pairwise token interactions). If the whole backlog is placed in one prompt, *L* grows with the amount of material in that prompt, so each forward pass becomes relatively expensive. At larger backlogs this pattern is often **slow** and **resource-heavy** compared to keeping contexts short.

This setup does **not** ask the model to ingest every ticket at once. Each observation is limited to a **candidate**, a **small reference set**, and a **capped** summary of what has already been decided. **n** (the **`total_tickets`** field) then enters mainly through the allowed number of steps (**`max_steps`**, and optional **`end_ordering`**), rather than through how many tickets must sit in a single attention matrix. The hope is to ease the **quadratic-in-context-length** cost that arises when a prompt is sized to include the whole backlog at once (often discussed as **O(n²)** when *L* grows with **n**).

**`max_steps`** is configurable by design. Setting it on the order of **n** gives a short, **one-shot**-like horizon; setting it on the order of **n²** leaves more room for cross-comparison if that is what you need. Those are **examples** only—the same API also supports values in between and other settings without branching the implementation.

Per-step reward is marginal improvement in ordering quality minus a step cost (see the reward subsection below). Structurally, that should favor moves that still meaningfully improve the ranking over extra steps that barely change it. A single end-to-end prompt that asks for a full ordering in one reply does **not** offer an equivalent **step-level** signal; it can still work well in small cases, but it **cannot**, on its own, express the same graded feedback across intermediate decisions.

## API surface

- `TicketOrderingAction` / `TicketOrderingObservation` — episode fields below
- Step reward — formula below

## Example

```python
from ticket_ordering import TicketOrderingAction, TicketOrderingEnv
import random

env = TicketOrderingEnv(base_url="http://localhost:8000")
result = env.reset()
obs = result.observation

result = env.step(
    TicketOrderingAction(
        candidate_priority=50,
        candidate_summary="issue",
        next_reference_ids=[],
        next_candidate_id=random.choice(list(obs.ticket_heuristics.keys())),
        end_ordering=False,
    )
)
print(result.observation, result.reward)
env.close()
```

## Environment Details

### Action: `TicketOrderingAction`

| Field | Role |
| --- | --- |
| `candidate_priority` (int) | Score vs other tickets |
| `candidate_summary` (str) | Short context for later compare |
| `next_reference_ids` (list[int]) | Ref ticket IDs for next step |
| `next_candidate_id` (int) | Next candidate |
| `end_ordering` (bool) | Terminate episode |

### Observation: `TicketOrderingObservation`

| Field | Role |
| --- | --- |
| `reward` (float) | Δ global optimality − small step cost |
| `done` (bool) | `end_ordering` or max steps |
| `metadata` (dict) | Extra |
| `ordering_criteria` (str) | Rubric |
| `reference_tickets` | Refs for current candidate |
| `candidate_ticket` | Current candidate |
| `ticket_heuristics` | Prior scores by id |
| `total_tickets` | Count to order |
| `completed_iterations` | Steps so far |

### Reward

Per step: `action_optimality_weight * Δoptimality - step_penalty_min_gain_fraction * smallest_optimality_quantum(n)`, `n` = ticket count, `smallest_optimality_quantum(n) = 2 / footrule_max_distance(n)` (min gap between distinct optimality scores). Penalty ∈ `(0,1)` in quantum units → flat step net negative; improving step beats same # no-ops. Return telescopes: net optimality gain − step costs.

## Benchmark results

Normalized episode scores (`inference.py`: reward sum scaled [0,1] per episode bounds), synthetic problems:

| Model | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| `llama-3.1-8b-instant` | 0.753 | 0.741 | 0.694 |

## Env logic without HTTP

```bash
python3 server/ticket_ordering_environment.py
```

Checks reset, step, state, rewards.

## Project Structure

```
ticket_ordering/
├── client.py
├── docker-build.sh
├── .dockerignore
├── inference.py
├── __init__.py
├── load-env-vars.sh
├── models.py
├── openenv.yaml
├── problem_generator.py
├── pyproject.toml
├── README.md
├── server/
│   ├── app.py
│   ├── Dockerfile
│   ├── __init__.py
│   ├── requirements.txt
│   └── ticket_ordering_environment.py
├── uv.lock
└── validation-script.sh
```
