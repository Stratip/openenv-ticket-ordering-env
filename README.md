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
