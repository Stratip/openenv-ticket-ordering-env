import os
import asyncio
import textwrap
import numpy as np
from openai import OpenAI
from typing import List, Optional, Dict, Any, Tuple

from client import TicketOrderingEnv
from problem_generator import GenerationDifficulty
from models import (
    TicketHeuristic,
    TicketOrderingAction,
    TicketOrderingObservation,
    TicketOrderingConfig,
    smallest_optimality_quantum,
)


backup_rng = np.random.default_rng(42)


ENV_BASE = "https://startripper-openenv-ticket-ordering-env.hf.space"

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
MAX_STEPS = 15
TEMPERATURE = 0.1 # Kind of makes these models TOO deterministic / repeat things, oh well, rules are rules.
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.75
UNASSIGNED_HEURISTIC = "unassigned"

_REWARD_CFG = TicketOrderingConfig()


def _episode_return_bounds(n_tickets: int, steps: int) -> Tuple[float, float]:
    """Match env: per-step cost = fraction * smallest_optimality_quantum(n)."""
    lam = _REWARD_CFG.step_penalty_min_gain_fraction * smallest_optimality_quantum(n_tickets)
    return -1.0 - lam * steps, 1.0 - lam

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a ticket prioritization agent.

    Given:
    - ordering criteria
    - reference tickets
    - a candidate ticket
    - heuristics from n most- and least-assigned tickets (times_assigned = prior steps as candidate; 0 until one completes, including first-time candidate)
    - total number of tickets
    - iterations completed so far

    Your job:
    - Assign a priority score to the candidate (0 to 100) (int, 0 = not important AT ALL, 100 = EXTREMELY important)
    - Write a short summary for the candidate (<=32 chars) (Part of that ticket's heuristic), this is to help choose the next reference and candidate ticket.
    - Select next reference ticket ids (must be one of the keys from the heuristics)
    - Select next candidate ticket id (must be one of the keys from the heuristics)
    - Decide whether to end ordering (there is no need to end after iterations completed = total tickets since cross comparing tickets still may be valuable)

    Respond with no fancy formatting, backticks or anything else, respond ONLY with PURE valid JSON in this format:
    {
      "candidate_priority": int,
      "candidate_summary": string,
      "next_reference_ids": [int],
      "next_candidate_id": int,
      "end_ordering": bool
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def serialize_heuristic(h: TicketHeuristic) -> Dict[str, Any]:
    ta = h.times_assigned
    return {
        "priority": UNASSIGNED_HEURISTIC if ta == 0 else h.priority,
        "summary": UNASSIGNED_HEURISTIC if ta == 0 else h.summary,
        "times_assigned": ta,
    }


def serialize_ticket(ticket: Any) -> Dict[str, Any]:
    return {
        "id": ticket.id,
        "thread": [{"user": c.user, "content": c.content} for c in ticket.thread],
        "heuristic": serialize_heuristic(ticket.heuristic),
    }


def build_user_prompt(obs: Any) -> str:
    return textwrap.dedent(
        f"""
        Ordering criteria: {obs.ordering_criteria}

        Reference tickets:
        {[serialize_ticket(t) for t in obs.reference_tickets]}

        Candidate ticket:
        {serialize_ticket(obs.candidate_ticket)}

        Existing heuristics:
        { {k: serialize_heuristic(v) for k, v in obs.ticket_heuristics.items()} }

        Total tickets: {obs.total_tickets}
        Completed iterations: {obs.completed_iterations}

        Decide the next action.
        """
    ).strip()


def get_model_action(client: OpenAI, obs: TicketOrderingObservation) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        import json

        return json.loads(text)
    except Exception:
        return {
            "candidate_priority": int(backup_rng.integers(low=1, high=101)),
            "candidate_summary": "issue",
            "next_reference_ids": [],
            "next_candidate_id": backup_rng.choice(list(obs.ticket_heuristics.keys())),
            "end_ordering": False,
        }


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    with TicketOrderingEnv(base_url=ENV_BASE).sync() as env:
    # with TicketOrderingEnv(base_url="localhost:8001").sync() as env:
        for task in [GenerationDifficulty.Easy, GenerationDifficulty.Medium, GenerationDifficulty.Hard]:
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=f"ticket_ordering_env_{task.name}", env="ticket_ordering_env", model=MODEL_NAME)

            try:
                result = env.reset(difficulty=task.value)
                obs = result.observation

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    action_dict = get_model_action(client, obs)

                    action = TicketOrderingAction(
                        candidate_priority=int(action_dict.get("candidate_priority", 0)),
                        candidate_summary=str(action_dict.get("candidate_summary", ""))[:32],
                        next_reference_ids=list(action_dict.get("next_reference_ids", [])),
                        next_candidate_id=int(action_dict.get("next_candidate_id", 0)),
                        end_ordering=bool(action_dict.get("end_ordering", False)),
                    )

                    result = env.step(action)
                    obs = result.observation

                    reward = result.reward or 0.0
                    done = result.done
                    error = None

                    rewards.append(reward)
                    steps_taken = step

                    log_step(
                        step=step,
                        action=str(action_dict),
                        reward=reward,
                        done=done,
                        error=error,
                    )

                    if done:
                        break

                ep_min, ep_max = _episode_return_bounds(
                    max(obs.total_tickets, 2),
                    _REWARD_CFG.max_steps,
                )
                rewards_sum = sum(rewards)
                rewards_sum -= ep_min
                rewards_sum /= ep_max - ep_min
                score = min(max(rewards_sum, 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD

            finally:
                try:
                    env.close()
                except Exception:
                    pass
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
