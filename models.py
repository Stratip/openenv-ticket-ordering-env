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


from pydantic import BaseModel, Field
from openenv.core.env_server import State, Observation, Action


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
    max_steps: int = Field(default=50, description="Maximum number of steps in an episode.")
    max_reference_tickets: int = Field(default=5, description="Maximum number of references an agent can request on a step.")
    max_heurestics: int = Field(default=10, description="Maximum number of heuristics in an observation on a step.")

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
