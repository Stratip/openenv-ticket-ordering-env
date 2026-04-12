# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ticket Ordering Environment Implementation.
"""


import heapq
import logging
import numpy as np
from uuid import uuid4
from copy import deepcopy

from openenv.core.env_server.interfaces import Any, Environment, Optional


logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)


try:
    from models import (
        Ticket,
        TicketHeuristic,
        TicketOrderingConfig,
        TicketOrderingState,
        TicketOrderingObservation,
        TicketOrderingAction,
        footrule_max_distance,
        smallest_optimality_quantum,
    )
    from problem_generator import generate_problem_statement, GenerationDifficulty
except ImportError:
    from ..models import (
        Ticket,
        TicketHeuristic,
        TicketOrderingConfig,
        TicketOrderingState,
        TicketOrderingObservation,
        TicketOrderingAction,
        footrule_max_distance,
        smallest_optimality_quantum,
    )
    from ..problem_generator import generate_problem_statement, GenerationDifficulty


class TicketOrderingEnvironment(Environment):
    """
    An environment that orders tickets.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the ticket_ordering environment."""

        self._config = TicketOrderingConfig()

        self._reset_count = 0

        self.rng = np.random.default_rng(42)

        self._state = TicketOrderingState(
            episode_id=str(uuid4()),
            step_count=0,
            ordering_criteria="",
            optimally_ordered_ticket_ids=[],
            tickets=[
                Ticket(id=0, thread=[], heuristic=TicketHeuristic()),
                Ticket(id=1, thread=[], heuristic=TicketHeuristic())
            ],
            optimality=0.5,
        )
        self._state_id_index_map: dict[int, int] = {}

        self._current_candidate: Ticket = Ticket(
            id=0,
            thread=[],
            heuristic=TicketHeuristic(priority=0, summary="")
        )
        self._current_references: list[Ticket] = []
        self._current_heuristics: dict[int, TicketHeuristic] = {}


    def reset(
        self,
        seed: Optional[int] = 42,
        episode_id: Optional[str] = str(uuid4()),
        difficulty: Optional[int] = GenerationDifficulty.Medium.value,
        **kwargs: Any,
    ) -> TicketOrderingObservation:
        """
        Reset the environment.

        Returns:
            TicketOrderingObservation
        """

        self._reset_count += 1

        self.rng = np.random.default_rng(seed if seed is not None else 42)

        generated_criteria, generated_tickets = generate_problem_statement(
            GenerationDifficulty(difficulty) if difficulty is not None else GenerationDifficulty.Medium
        )
        shuffled_tickets = deepcopy(generated_tickets)
        self.rng.shuffle(shuffled_tickets)
        optimal_ticket_ids = [ticket.id for ticket in generated_tickets]

        self._state = TicketOrderingState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            ordering_criteria=generated_criteria,
            optimally_ordered_ticket_ids=optimal_ticket_ids,
            tickets=shuffled_tickets,
            optimality=self.compute_optimality(shuffled_tickets, optimal_ticket_ids),
        )
        self._state_id_index_map = self._make_id_index_map(self._state.tickets)

        self._current_candidate = self._state.tickets[0]
        self._current_references = self._state.tickets[0:1]
        self._current_heuristics = self.select_heuristics(self._state)


        return TicketOrderingObservation(
            done=False,
            reward=0.0,
            metadata={},

            ordering_criteria=self._state.ordering_criteria,
            reference_tickets=self._current_references,
            candidate_ticket=self._current_candidate,
            ticket_heuristics=self._current_heuristics,
            total_tickets=len(self._state.tickets),
            completed_iterations=self._state.step_count
        )


    def step(self, action: TicketOrderingAction) -> TicketOrderingObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: TicketOrderingAction containing the assigned priority and summary for the candidate ticket,
            next reference ticket ids, next candidate ticket id and whether or not ordering should stop (in case
            the agent decides ordering has "reached optimality" at any point in time)

        Returns:
            TicketOrderingObservation
        """

        previous_state = deepcopy(self._state)

        updated_state = self.get_updated_state(previous_state, self._current_candidate, action)

        self._current_candidate = self.select_candidate(updated_state, action)
        self._current_references = self.select_references(updated_state, action)
        self._current_heuristics = self.select_heuristics(updated_state)

        self._state = deepcopy(updated_state)

        observation = self.construct_observation(previous_state, updated_state, action)

        return observation


    @property
    def state(self) -> TicketOrderingState:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state    


    def select_candidate(self, state: TicketOrderingState, action: TicketOrderingAction) -> Ticket:
        id_index_map = self._make_id_index_map(state.tickets)

        if action.next_candidate_id in id_index_map:
            candidate_ticket_index = id_index_map[action.next_candidate_id]
        else:
            candidate_ticket_index = id_index_map[state.tickets[0].id]

        return deepcopy(state.tickets[candidate_ticket_index])


    def select_references(self, state: TicketOrderingState, action: TicketOrderingAction) -> list[Ticket]:
        id_index_map = self._make_id_index_map(state.tickets)
        
        reference_ticket_indices = []
        for id in action.next_reference_ids[:self._config.max_reference_tickets]:
            if id in id_index_map:
                reference_ticket_indices.append(id_index_map[id])
            else:
                reference_ticket_indices.append(id_index_map[state.tickets[0].id])

        return deepcopy(
            [state.tickets[index] for index in reference_ticket_indices]
        )


    def select_heuristics(self, state: TicketOrderingState) -> dict[int, TicketHeuristic]:
        heuristics = {}

        assert self._config.max_heurestics >= 2
        assert self._config.max_heurestics % 2 == 0

        n = self._config.max_heurestics // 2
        largest = heapq.nlargest(n, state.tickets, key=lambda x: (x.heuristic.times_assigned, self.rng.random()))
        smallest = heapq.nsmallest(n, state.tickets, key=lambda x: (x.heuristic.times_assigned, self.rng.random()))

        heuristics.update(
            {ticket.id: ticket.heuristic for ticket in largest}
        )
        heuristics.update(
            {ticket.id: ticket.heuristic for ticket in smallest}
        )

        return heuristics


    def construct_observation(
        self,
        previous_state: TicketOrderingState,
        new_state: TicketOrderingState,
        action: TicketOrderingAction,
    ) -> TicketOrderingObservation:
        observation = TicketOrderingObservation(
            done=action.end_ordering or (new_state.step_count >= self._config.max_steps),
            reward=self.construct_reward(previous_state, new_state),

            ordering_criteria=new_state.ordering_criteria,
            reference_tickets=self._current_references,
            candidate_ticket=self._current_candidate,
            ticket_heuristics=self._current_heuristics,

            total_tickets=len(new_state.tickets),
            completed_iterations=new_state.step_count
        )

        return observation


    def construct_reward(
        self,
        previous_state: TicketOrderingState,
        new_state: TicketOrderingState,
    ) -> float:
        """
        Per-step reward = (weight * marginal optimality) - step_penalty(n).

        Step cost scales with ticket count: smallest distinct optimality jump is
        ``2 / footrule_max_distance(n)``; penalty is a fraction of that, so any step that
        improves optimality beats the same number of flat steps.

        Episode return telescopes to:
            sum_t r_t = w * (final_optimality - initial_optimality) - sum_t penalty(n)
        """
        w = self._config.action_optimality_weight
        n = len(new_state.tickets)
        min_gain = smallest_optimality_quantum(n)
        lam = self._config.step_penalty_min_gain_fraction * min_gain
        delta = new_state.optimality - previous_state.optimality
        return w * delta - lam


    def get_updated_state(self, previous_state: TicketOrderingState, candidate: Ticket, action: TicketOrderingAction) -> TicketOrderingState:
        id_index_map = self._make_id_index_map(previous_state.tickets)
        updated_state = deepcopy(previous_state)

        updated_state.step_count += 1

        candidate_ticket_index = id_index_map[candidate.id]
        updated_state.tickets[candidate_ticket_index].heuristic.priority = action.candidate_priority
        updated_state.tickets[candidate_ticket_index].heuristic.summary = action.candidate_summary
        updated_state.tickets[candidate_ticket_index].heuristic.times_assigned += 1

        updated_state.tickets = self.reorder_tickets(updated_state.tickets)

        updated_state.optimality = self.compute_optimality(updated_state.tickets, updated_state.optimally_ordered_ticket_ids)

        return updated_state


    def compute_optimality(self, tickets: list[Ticket], optimal_ids: list[int]):
        normalized_distance = self.normalized_spearman_footrule_distance(
            [ticket.id for ticket in tickets],
            [id for id in optimal_ids],
        )
        return 1.0 - normalized_distance


    def _make_id_index_map(self, tickets: list[Ticket]) -> dict[int, int]:
        id_index_map = {ticket.id: index for index, ticket in enumerate(tickets)}
        return id_index_map


    def reorder_tickets(self, tickets: list[Ticket]) -> list[Ticket]:
        copied_tickets = deepcopy(tickets)
        copied_tickets.sort(key=lambda ticket: ticket.heuristic.priority)
        return copied_tickets


    def normalized_spearman_footrule_distance(self, a: list[int], b: list[int]) -> float:
        assert len(a) == len(b)
        assert len(a) > 0

        a_value_index_map = {
            value: index for index, value in enumerate(a)
        }

        accumulated_distance = 0.0
        for index, value in enumerate(b):
            a_index = a_value_index_map[value]
            distance = float(abs(index - a_index))
            accumulated_distance += distance

        n = len(a)
        max_distance = footrule_max_distance(n)

        normalized_distance = accumulated_distance / max_distance

        return normalized_distance
