# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv server implementation for ticket-ordering episodes.

Exposes :class:`TicketOrderingEnvironment` for HTTP/WebSocket drivers; scoring helpers
live on the same class so clients can align normalization with :meth:`TicketOrderingEnvironment.construct_reward`.
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
    Multi-step environment: agent assigns priorities and summaries to tickets under a
    text criterion, receives observations with a candidate and references, and is scored
    by how much optimality (Spearman footrule vs ground-truth order) improves each step.

    Each :meth:`reset` samples a synthetic problem (difficulty-controlled), shuffles tickets,
    and sets the step budget from :class:`~models.TicketOrderingConfig`. Each :meth:`step`
    applies the action, reorders tickets by heuristic priority, recomputes optimality, and
    returns reward plus the next observation.

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: When ``True``, the OpenEnv server may attach multiple
            WebSocket clients with isolated instances (factory mode). This class keeps
            per-instance state only.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """
        Build default config, RNG, and placeholder state before the first :meth:`reset`.

        The placeholder state satisfies validators; real episodes replace it on ``reset``.
        """

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
        self._episode_max_steps = self._resolve_episode_max_steps(len(self._state.tickets))

    def _resolve_episode_max_steps(
        self,
        n: int,
        *,
        max_steps: Optional[int] = None,
        max_steps_scale: Optional[float] = None,
        max_steps_cap: Optional[int] = None,
        max_steps_n_exponent: Optional[float] = None,
    ) -> int:
        """Compute step budget from config and optional reset() overrides."""
        cfg = self._config
        ms = cfg.max_steps if max_steps is None else max_steps
        scale = cfg.max_steps_scale if max_steps_scale is None else max_steps_scale
        mcap = cfg.max_steps_cap if max_steps_cap is None else max_steps_cap
        exp = cfg.max_steps_n_exponent if max_steps_n_exponent is None else max_steps_n_exponent

        if scale is not None:
            if scale <= 0:
                raise ValueError("max_steps_scale must be positive when set")
            raw = scale * (float(n) ** exp)
            steps = max(1, int(round(raw)))
            if mcap is not None:
                steps = min(steps, mcap)
        else:
            steps = max(1, ms)

        if steps < n:
            logger.warning(
                f"Episode max_steps ({steps}) is less than ticket count ({n}); "
                "the horizon may end before every ticket has been a candidate.",
            )
        return steps

    def reset(
        self,
        seed: Optional[int] = 42,
        episode_id: Optional[str] = str(uuid4()),
        difficulty: Optional[int] = GenerationDifficulty.Medium.value,
        max_steps: Optional[int] = None,
        max_steps_scale: Optional[float] = None,
        max_steps_cap: Optional[int] = None,
        max_steps_n_exponent: Optional[float] = None,
        **kwargs: Any,
    ) -> TicketOrderingObservation:
        """
        Start a new episode: generate criteria and tickets, shuffle, set optimality, step cap.

        Args:
            seed: NumPy RNG seed for shuffling and heuristic subsampling; ``None`` uses 42.
            episode_id: Optional stable id for logging; default is a new UUID string.
            difficulty: ``GenerationDifficulty`` enum value (0=easy, 1=medium, 2=hard).
            max_steps: Fixed horizon when ``max_steps_scale`` is unset in config; overridden
                by proportional scaling when scale is set.
            max_steps_scale: If not ``None``, episode length scales with ticket count ``n``
                (rounded ``scale * n**exponent``, optional cap); overrides fixed ``max_steps``
                for the terminal condition.
            max_steps_cap: Optional upper bound on scaled step budget.
            max_steps_n_exponent: Exponent on ``n`` when using ``max_steps_scale``.
            **kwargs: Ignored; reserved for OpenEnv compatibility.

        Returns:
            Initial :class:`~models.TicketOrderingObservation` with ``done=False``,
            ``reward=0.0``, criteria, first candidate, references, heuristics subset, and
            ``max_steps`` set to this episode's budget.
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

        self._episode_max_steps = self._resolve_episode_max_steps(
            len(shuffled_tickets),
            max_steps=max_steps,
            max_steps_scale=max_steps_scale,
            max_steps_cap=max_steps_cap,
            max_steps_n_exponent=max_steps_n_exponent,
        )

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
            completed_iterations=self._state.step_count,
            max_steps=self._episode_max_steps,
        )


    def _validate_action_against_heuristics(self, action: TicketOrderingAction) -> None:
        """Require next candidate and every next reference id to appear in the step's heuristic keys."""
        allowed = frozenset(self._current_heuristics.keys())
        if action.next_candidate_id not in allowed:
            raise ValueError(
                f"next_candidate_id {action.next_candidate_id} is not in ticket_heuristics keys {sorted(allowed)}"
            )
        bad_refs = [rid for rid in action.next_reference_ids if rid not in allowed]
        if bad_refs:
            raise ValueError(
                f"next_reference_ids {bad_refs} are not in ticket_heuristics keys {sorted(allowed)}"
            )

    def step(self, action: TicketOrderingAction) -> TicketOrderingObservation:  # type: ignore[override]
        """
        Apply ``action`` to the current candidate, advance state, and return the next observation.

        Args:
            action: Priority and summary for the current candidate, next reference ticket ids
                (subset of keys from the last observation's ``ticket_heuristics``), next
                candidate id (also a key in ``ticket_heuristics``), and ``end_ordering`` to
                request early termination.

        Returns:
            :class:`~models.TicketOrderingObservation` with updated candidate, references,
            heuristics, step count, per-step ``reward``, and ``done`` if ``end_ordering`` or
            the step budget is exhausted.

        Raises:
            ValueError: If ``next_candidate_id`` or any ``next_reference_ids`` entry is not
                among the ids in ``ticket_heuristics`` for this step.
        """
        self._validate_action_against_heuristics(action)

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
        Snapshot of the live episode: criteria, ground-truth order, tickets, optimality, step count.

        Returns:
            :class:`~models.TicketOrderingState` after the last completed ``reset`` or ``step``.
        """
        return self._state


    def select_candidate(self, state: TicketOrderingState, action: TicketOrderingAction) -> Ticket:
        id_index_map = self._make_id_index_map(state.tickets)
        candidate_ticket_index = id_index_map[action.next_candidate_id]
        return deepcopy(state.tickets[candidate_ticket_index])


    def select_references(self, state: TicketOrderingState, action: TicketOrderingAction) -> list[Ticket]:
        id_index_map = self._make_id_index_map(state.tickets)
        reference_ticket_indices = [
            id_index_map[rid]
            for rid in action.next_reference_ids[: self._config.max_reference_tickets]
        ]
        return deepcopy([state.tickets[index] for index in reference_ticket_indices])


    def select_heuristics(self, state: TicketOrderingState) -> dict[int, TicketHeuristic]:
        heuristics = {}

        assert self._config.max_heuristics >= 2
        assert self._config.max_heuristics % 2 == 0

        n = self._config.max_heuristics // 2
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
            done=action.end_ordering or (new_state.step_count >= self._episode_max_steps),
            reward=self.construct_reward(previous_state, new_state),

            ordering_criteria=new_state.ordering_criteria,
            reference_tickets=self._current_references,
            candidate_ticket=self._current_candidate,
            ticket_heuristics=self._current_heuristics,

            total_tickets=len(new_state.tickets),
            completed_iterations=new_state.step_count,
            max_steps=self._episode_max_steps,
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

    @staticmethod
    def compute_episode_return_bounds(
        n_tickets: int,
        num_steps: int,
        *,
        config: Optional[TicketOrderingConfig] = None,
    ) -> tuple[float, float]:
        """
        Min/max possible undiscounted episode return for ``num_steps`` steps at fixed ``n_tickets``.

        Same telescoping identity as :meth:`construct_reward`: with optimality in ``[0, 1]``,
        ``sum_t r_t = w * (final_optimality - initial_optimality) - num_steps * lam`` where
        ``w`` is ``action_optimality_weight`` and ``lam`` matches per-step penalty in
        :meth:`construct_reward` (uses ``smallest_optimality_quantum(n_tickets)``).

        Args:
            n_tickets: Backlog size ``n`` used for the per-step penalty quantum (must match
                the episode's ticket count for correct bounds).
            num_steps: Episode horizon (e.g. observation ``max_steps``) used to multiply the
                per-step penalty in the bound.
            config: Reward/step hyperparameters; default matches a fresh
                :class:`~models.TicketOrderingConfig` (callers should pass the server config
                if it differs from defaults).

        Returns:
            ``(min_return, max_return)`` suitable for linearly rescaling summed step rewards
            to ``[0, 1]``.
        """
        cfg = config or TicketOrderingConfig()
        w = cfg.action_optimality_weight
        lam = cfg.step_penalty_min_gain_fraction * smallest_optimality_quantum(n_tickets)
        return (-w - lam * num_steps, w - lam * num_steps)

    def episode_return_bounds(self, n_tickets: int, num_steps: int) -> tuple[float, float]:
        """
        Episode return bounds using this environment's loaded :class:`~models.TicketOrderingConfig`.

        Args:
            n_tickets: Ticket count for the quantum in the penalty term.
            num_steps: Step budget for the episode.

        Returns:
            Same as :meth:`compute_episode_return_bounds` with ``config=self._config``.
        """
        return self.compute_episode_return_bounds(n_tickets, num_steps, config=self._config)

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
