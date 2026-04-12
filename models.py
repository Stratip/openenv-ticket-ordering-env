# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Ticket Ordering Environment.

The ticket_ordering environment is an environment that presents an environment for optimally ordering
tickets (e.g. Jira tickets, Github issues) according to an arbitrary (but clearly specified) criteria.
"""


from typing import Optional

from pydantic import BaseModel, Field, model_validator
from openenv.core.env_server import State, Observation, Action


def footrule_max_distance(n: int) -> float:
    """Maximum Spearman footrule distance for permutations of length ``n`` (env normalizer)."""
    if n <= 1:
        return 1.0
    return (n * n) / 2.0 if n % 2 == 0 else (n * n - 1) / 2.0


def smallest_optimality_quantum(n: int) -> float:
    """Smallest positive gap between two distinct optimality scores for ``n`` tickets."""
    return 2.0 / footrule_max_distance(n)


class ThreadComment(BaseModel):
    user: str = Field(min_length=1, max_length=16, description="Username of user commenting on the thread")
    content: str = Field(min_length=1, max_length=1024, description="Content of the user's comment on the thread.")

class TicketHeuristic(BaseModel):
    priority: int = Field(default=0, description="Relative priority of the ticket. Priority is only meaningful relative to other tickets.")
    summary: str = Field(default="", max_length=32, description="Summary of the ticket. Useful for agents to store context for tricky tickets.")
    times_assigned: int = Field(default=0, description="Times the heuristic has been assigned. Loose metric of 'certainty' (more assignments ~= more certain)")

class Ticket(BaseModel):
    id: int = Field(description="ID of the ticket, random / meaningless unique identifier.")
    thread: list[ThreadComment] = Field(description="Thread of comments on the ticket.")

    heuristic: TicketHeuristic = Field(description="Heuristic of the ticket.")




class TicketOrderingConfig(BaseModel):
    max_steps: int = Field(
        default=50,
        ge=1,
        description="Fixed maximum steps per episode when max_steps_scale is not set.",
    )
    max_steps_scale: Optional[float] = Field(
        default=2.0,
        description=(
            "When set, episode step budget scales with backlog size n: "
            "round(max_steps_scale * n**max_steps_n_exponent), clamped to at least 1 "
            "and optionally max_steps_cap. Ignores fixed max_steps for the terminal condition. "
            "Default 2 with max_steps_n_exponent 1 is linear 2 steps per ticket."
        ),
    )
    max_steps_n_exponent: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Exponent on n when using max_steps_scale (1 ≈ O(n) horizon, 2 ≈ O(n²) headroom)."
        ),
    )
    max_steps_cap: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional upper bound on the proportional step budget (only used when max_steps_scale is set).",
    )
    max_reference_tickets: int = Field(default=5, description="Maximum number of references an agent can request on a step.")
    max_heuristics: int = Field(default=10, description="Maximum number of heuristics in an observation on a step.")
    step_penalty_min_gain_fraction: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Per-step cost = fraction * smallest_optimality_quantum(n). "
            "Must stay < 1 so any strict improvement yields positive net reward vs flat steps."
        ),
    )
    action_optimality_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight on marginal optimality change (how much this action improved or hurt the ordering).",
    )

    @model_validator(mode="after")
    def _validate_proportional_steps(self) -> "TicketOrderingConfig":
        if self.max_steps_scale is not None and self.max_steps_scale <= 0:
            raise ValueError("max_steps_scale must be positive when set")
        return self


class TicketOrderingState(State):
    """State for the Ticket Ordering environment."""
    ordering_criteria: str = Field(max_length=32, description="Criteria by which tickets are to be ordered.")
    optimally_ordered_ticket_ids: list[int] = Field(description="IDs of the optimal ticket at each position / index.")
    tickets: list[Ticket] = Field(min_length=2, max_length=128, description="Tickets to be ordered by the agent.")
    optimality: float = Field(ge=0.0, le=1.0, description="How close the tickets are to being optimal, can be thought of as a percentage.")


class TicketOrderingObservation(Observation):
    """Observation from the Ticket Ordering environment."""
    ordering_criteria: str = Field(
        max_length=32,
        description="Criteria by which tickets should be ordered."
    )
    reference_tickets: list[Ticket] = Field(
        description="Subset of tickets provided as references to help evaluate and compare the current candidate ticket."
    )
    candidate_ticket: Ticket = Field(
        description="The ticket currently being evaluated and assigned a heuristic by the agent."
    )
    ticket_heuristics: dict[int, TicketHeuristic] = Field(
        description="Known heuristics for previously evaluated tickets, keyed by ticket ID, used for relative comparison."
    )
    total_tickets: int = Field(
        default=0,
        description="Total number of tickets in the environment that need to be ordered."
    )
    completed_iterations: int = Field(
        default=0,
        description="Number of ordering steps completed so far in the current episode."
    )
    max_steps: int = Field(
        ge=1,
        description="Step budget for this episode; episode ends when completed_iterations reaches this (unless end_ordering).",
    )


class TicketOrderingAction(Action):
    """Action for the Ticket Ordering environment."""
    candidate_priority: int = Field(
        description="Assigned priority score for the candidate ticket, relative to other tickets."
    )
    candidate_summary: str = Field(
        description="Short summary describing the candidate ticket, capturing key context for future comparisons."
    )
    next_reference_ids: list[int] = Field(
        description="IDs of tickets to request as references in the next step, used to guide further comparisons."
    )
    next_candidate_id: int = Field(
        description="ID of the next ticket to evaluate as the candidate in the following step."
    )
    end_ordering: bool = Field(
        description="Whether the agent thinks it has finished ordering all tickets and wants to terminate the episode."
    )
