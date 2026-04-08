import re
import numpy as np
from enum import Enum
from typing import DefaultDict
from models import ThreadComment, Ticket, TicketHeuristic


rng = np.random.default_rng(42)


class GenerationDifficulty(Enum):
    Easy = 0
    Medium = 1
    Hard = 2


NAMES = [
    "Aarav", "Emma", "Liam", "Olivia", "Noah", "Ava", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amir", "Fatima", "Hassan", "Layla", "Omar", "Yasmin", "Ali", "Zara", "Ibrahim", "Noor",
    "Wei", "Yuki", "Hiroshi", "Mei", "Sora", "Jin", "Minho", "Haruto", "Aiko", "Ren",
    "Carlos", "Sofia", "Mateo", "Lucia", "Diego", "Valentina", "Juan", "Camila", "Luis", "Elena",
    "Ethan", "Abigail", "James", "Emily", "Benjamin", "Ella", "Lucas", "Scarlett", "Henry", "Grace",
    "Arjun", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Kiran", "Isha", "Rahul", "Neha",
    "Leo", "Chloe", "Gabriel", "Lily", "Samuel", "Zoe", "Daniel", "Hannah", "Matthew", "Aria",
    "Alexander", "Nina", "Mikhail", "Anastasia", "Ivan", "Svetlana", "Dmitry", "Olga", "Sergey", "Irina",
    "Kwame", "Ama", "Kofi", "Zuri", "Abena", "Tariq", "Amina", "Malik", "Imani", "Nia",
    "Oscar", "Freja", "Lars", "Ingrid", "Bjorn", "Astrid", "Erik", "Sigrid", "Magnus", "Elin"
]


DIFFICULTY_UNCERTAINTY_MAP = {
    GenerationDifficulty.Easy: 0.0,
    GenerationDifficulty.Medium: 0.0833,
    GenerationDifficulty.Hard: 0.1666,
}


CRITERIA = [
    "severity",        # how bad
    "fix ease",        # how easy to fix
    "scope backend",   # backend impact
    "scope frontend",  # frontend impact
    "user impact",     # perceived impact
]
CRITERIA_DIST_RANGES = {
    "severity": (0.15, 1.9),
    "fix ease": (0.2, 1.7),
    "scope backend": (0.5, 1.0),
    "scope frontend": (0.0, 0.5),
    "user impact": (0.5, 2.9),
}

ISSUE_TYPES = [
    ("crash", {
        "severity": 0.9,
        "fix ease": 0.8,
        "user impact": 0.9,
    }),
    ("failure", {
        "severity": 0.75,
        "fix ease": 0.7,
        "user impact": 0.8,
    }),
    ("bug", {
        "severity": 0.4,
        "fix ease": 0.4,
        "user impact": 0.5,
    }),
    ("slowdown", {
        "severity": 0.35,
        "fix ease": 0.5,
        "user impact": 0.6,
    }),
    ("feature request", {
        "severity": 0.05,
        "fix ease": 0.1,
        "user impact": 0.3,
    }),
]

SYSTEM_PARTS = [
    ("lookup API", {
        "scope backend": 1.0,
        "scope frontend": 0.1,
        "user impact": 0.1,
    }),
    ("bot API", {
        "scope backend": 1.0,
        "scope frontend": 0.0,
        "user impact": 0.1,
    }),
    ("login system", {
        "scope backend": 0.5,
        "scope frontend": 0.5,
        "user impact": 1.0,
    }),
    ("automatic curation", {
        "scope backend": 0.8,
        "scope frontend": 0.3,
        "user impact": 0.25,
    }),
    ("relationship routing", {
        "scope backend": 0.75,
        "scope frontend": 0.25,
        "user impact": 0.2,
    }),
    ("trend tracker", {
        "scope backend": 0.9,
        "scope frontend": 0.2,
        "user impact": 0.2,
    }),
]

MODIFIERS = [
    ("minor", {
        "severity": 0.1,
        "fix ease": 0.9,
        "user impact": 0.1,
    }),
    ("intermittent", {
        "severity": 0.2,
        "fix ease": 0.4,
        "user impact": 0.2,
    }),
    ("random", {
        "severity": 0.5,
        "fix ease": 0.1,
        "user impact": 0.45,
    }),
    ("unexpected", {
        "severity": 0.8,
        "fix ease": 0.5,
        "user impact": 0.75,
    }),
    ("severe", {
        "severity": 0.75,
        "fix ease": 0.5,
        "user impact": 0.8,
    }),
    ("critical", {
        "severity": 0.9,
        "fix ease": 0.5,
        "user impact": 0.9,
    }),
    ("disastrous", {
        "severity": 1.0,
        "fix ease": 0.5,
        "user impact": 1.0,
    }),
]

CONTEXTS = [
    "during normal usage",
    "under heavy user load",
    "under heavy API load",
    "fixed time after deployment",
    "after db migration",
]

ACTIONS = [
    "pressing action button {random}",
    "opening dashboard",
    "registering on platform",
    "updating {random} multiple times within {random2} seconds",
    "removing post",
    "replying to user",
    "submitting poll",
    "making post"
]

NEW_TEMPLATES = [
    "{modifier} {issue} affecting {system} {context}",
    "{issue} in {system} triggered by {action}",
    "{modifier} {issue} when {action} in {system}",
    
    "users experience {modifier} {issue} in {system} {context}",
    "multiple users report {issue} while {action}",
    "user reports {issue} after {action} {context}",

    "{action} leads to {modifier} {issue} in {system}",
    "{system} shows {modifier} behavior when {action}",
    
    "{issue} detected in {system} {context}",
    "{modifier} degradation in {system} {context}",
    
    "{issue} in {system} after {action} {context}",
    "{modifier} issue observed in {system} when {action} {context}",

    "{modifier} {issue} impacting users during {context}",
    "{issue} causing failures in {system} under {context}",
]


def fill_action(action_template):
    fillers = [
        "profile", "settings", "feed", "post", "account",
        "preferences", "notification settings"
    ]
    numbers = ["2", "3", "5", "10", "30"]

    result = action_template
    result = result.replace("{random}", rng.choice(fillers))
    result = result.replace("{random2}", rng.choice(numbers))
    return result


def maybe(value, probability=0.7):
    return value if rng.random() < probability else ""


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def combine_scores(
    issue, system, modifier,
    
    uncertainty = 0.0,
    importances = {
        "severity": 0.2,
        "fix ease": 0.2,
        "scope backend": 0.2,
        "scope frontend": 0.2,
        "user impact": 0.2,
    }
):
    scores = DefaultDict(float)

    for key, value in issue.items():
        scores[key] += value
    for key, value in system.items():
        scores[key] += value
    for key, value in modifier.items():
        scores[key] += value

    combined_score = 0.0
    for criteria in CRITERIA:
        _min, _max  = CRITERIA_DIST_RANGES[criteria]
        scores[criteria] -= _min
        scores[criteria] /= _max - _min

        scores[criteria] += rng.uniform(-uncertainty, +uncertainty)

        scores[criteria] = min(scores[criteria], 1.0)
        scores[criteria] = max(scores[criteria], 0.0)

        scores[criteria] *= importances[criteria]

        combined_score += scores[criteria]

    combined_score = min(combined_score, 1.0)
    combined_score = max(combined_score, 0.0)

    return combined_score


def generate_ticket_data(
    uncertainty = 0.0,
    importances = {
        "severity": 0.2,
        "fix ease": 0.2,
        "scope backend": 0.2,
        "scope frontend": 0.2,
        "user impact": 0.2,
    }
):
    template = rng.choice(NEW_TEMPLATES)

    issue_name, issue_vals = rng.choice(ISSUE_TYPES) # type: ignore
    system_name, system_vals = rng.choice(SYSTEM_PARTS) # type: ignore
    modifier_name, modifier_vals = rng.choice(MODIFIERS) # type: ignore
    context = rng.choice(CONTEXTS)
    action_template = rng.choice(ACTIONS)

    action = fill_action(action_template)

    text = template.format(
        modifier=maybe(modifier_name),
        issue=issue_name,
        system=system_name,
        context=context,
        action=action,
    )
    combined_score = combine_scores(
        issue_vals, system_vals, modifier_vals,
        uncertainty=uncertainty,
        importances=importances
    )

    return clean_text(text), combined_score


def generate_problem_statement(difficulty: GenerationDifficulty = GenerationDifficulty.Medium) -> tuple[str, list[Ticket]]:
    rng = np.random.default_rng(42 + difficulty.value)

    criteria_str = rng.choice(CRITERIA)
    criteria_importances = {
        "severity": 0.2,
        "fix ease": 0.2,
        "scope backend": 0.2,
        "scope frontend": 0.2,
        "user impact": 0.2,
    }
    for key in criteria_importances:
        if key == criteria_str: criteria_importances[key] = 0.8
        else: criteria_importances[key] = 0.05

    uncertainty = DIFFICULTY_UNCERTAINTY_MAP[difficulty]

    ids = set()
    tickets_with_scores = []
    for _ in range(rng.integers(5, 10)):
        while True:
            id = rng.integers(0, 15, dtype=int)
            if id not in ids:
                ids.add(id)
                break

        text, score = generate_ticket_data(uncertainty, criteria_importances)

        ticket = Ticket(
            id=id,
            thread=[ThreadComment(user=rng.choice(NAMES), content=text)],
            heuristic=TicketHeuristic()
        )

        tickets_with_scores.append((ticket, score))

    tickets_with_scores.sort(key=lambda x: x[1])

    tickets = [t for t, _ in tickets_with_scores]

    return criteria_str, tickets
