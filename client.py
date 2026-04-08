# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Ticket Ordering Environment Client."""


from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import TicketOrderingAction, TicketOrderingObservation, TicketOrderingState


class TicketOrderingEnv(
    EnvClient[TicketOrderingAction, TicketOrderingObservation, TicketOrderingState]
):
    """
    Client for the Ticket Ordering Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """


    def _step_payload(self, action: TicketOrderingAction) -> Dict:
        """
        Convert TicketOrderingAction to JSON payload for step message.

        Args:
            action: TicketOrderingAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()


    def _parse_result(self, payload: Dict) -> StepResult[TicketOrderingObservation]:
        """
        Parse server response into StepResult[TicketOrderingObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TicketOrderingObservation
        """

        observation = TicketOrderingObservation.model_validate(payload.get("observation"))

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )


    def _parse_state(self, payload: Dict) -> TicketOrderingState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with:
                episode_id,
                step_count,
                ordering_criteria: Criteria by which the ordering is to take place.
                optimally_ordered_ticket_ids: IDs of the tickets assuming them in their optimal positions.
                tickets: The tickets that the agent operates on during an episode. Initially shuffled randomly.
                optimality: 1.0 - Normalized Spearman footrule distance between optimal ticket IDs and actual tickets. Should always be between 0.0 and 1.0.
        """

        state = TicketOrderingState.model_validate(payload)

        return state
