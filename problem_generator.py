from __future__ import annotations

from enum import Enum

try:
    # When run from within the `ticket_ordering/` package.
    from models import ThreadComment, Ticket, TicketHeuristic
except ModuleNotFoundError as e:  # pragma: no cover
    # When imported from `ticket_ordering.server` (relative import context).
    # Only fall back when *this* module can't be found; don't mask missing dependencies.
    if e.name not in {"models"}:
        raise
    from .models import ThreadComment, Ticket, TicketHeuristic


class GenerationDifficulty(Enum):
    Easy = 0
    Medium = 1
    Hard = 2


def _ticket(ticket_id: int, thread: list[tuple[str, str]]) -> Ticket:
    return Ticket(
        id=ticket_id,
        thread=[ThreadComment(user=user, content=content) for user, content in thread],
        heuristic=TicketHeuristic(),
    )


def _easy_problem() -> tuple[str, list[Ticket]]:
    # Easy: 1:1 correlation between what you read and the correct ordering.
    # Ordering criteria is explicit and each ticket states its severity clearly.
    criteria = "severity"

    # Tickets are returned in optimal order (lowest severity -> highest severity).
    tickets = [
        _ticket(
            10,
            [
                ("Ava", "Minor cosmetic: button label is slightly misaligned on the settings page. No functional impact."),
            ],
        ),
        _ticket(
            11,
            [
                ("Noah", "Low severity bug: typo in confirmation email subject line. Everything still sends correctly."),
            ],
        ),
        _ticket(
            12,
            [
                ("Mia", "Medium severity: search sometimes returns stale results until refresh. Workaround: refresh page."),
            ],
        ),
        _ticket(
            13,
            [
                ("Liam", "High severity: checkout fails for some users with 'payment method unavailable'. Blocks purchases."),
            ],
        ),
        _ticket(
            14,
            [
                ("Emma", "Critical severity: app crashes on launch for all Android users after latest release. Blocks all usage."),
            ],
        ),
    ]
    return criteria, tickets


def _medium_problem() -> tuple[str, list[Ticket]]:
    # Medium: the target criteria is clear, but there is mild misdirection:
    # - some tickets are scary but affect few users
    # - some are boring but affect many users
    criteria = "user impact"

    # Tickets are returned in optimal order (lowest user impact -> highest user impact).
    tickets = [
        _ticket(
            20,
            [
                ("Sofia", "Edge case: admin-only export occasionally omits one optional column. Affects a single internal admin."),
            ],
        ),
        _ticket(
            21,
            [
                ("Diego", "Rare crash: app closes when opening a specific legacy report. Only happens on an old device model."),
            ],
        ),
        _ticket(
            22,
            [
                ("Priya", "Moderate impact: password reset email is delayed by ~5 minutes during peak hours. Users can still log in otherwise."),
            ],
        ),
        _ticket(
            23,
            [
                ("Ethan", "High impact: notifications are not delivered reliably, causing users to miss replies. Many users report it daily."),
            ],
        ),
        _ticket(
            24,
            [
                ("Zara", "Very high impact: new user signup frequently fails with 'Try again later'. Large portion of new users affected."),
            ],
        ),
        _ticket(
            25,
            [
                ("Layla", "Extreme impact: login fails intermittently for a large segment of users. Reports across regions; users cannot access accounts."),
            ],
        ),
    ]
    return criteria, tickets


def _hard_problem() -> tuple[str, list[Ticket]]:
    # Hard: cross-ticket references and clarifications.
    # Users may exaggerate, misunderstand, or report symptoms; other tickets clarify root cause/scope.
    criteria = "user impact"

    # Tickets are returned in optimal order (lowest user impact -> highest user impact),
    # but you need to read carefully because early reports can be misleading.
    tickets = [
        _ticket(
            30,
            [
                ("Ren", "User report: 'Everything is broken' after clicking Save in settings. Claims app is unusable."),
                ("Aiko", "Support follow-up: reproduced once. Looks like only the settings page shows an error toast; navigation still works."),
                ("Ren", "Additional detail: error text says 'Invalid timezone'. User had entered 'PST' in a freeform field."),
            ],
        ),
        _ticket(
            31,
            [
                ("Carlos", "Report: profile pictures not updating. Looks like a caching issue; eventually correct after a few minutes."),
                ("Elena", "Engineering note: affects only the thumbnail URL; full-size image updates immediately."),
            ],
        ),
        _ticket(
            32,
            [
                ("Hassan", "Complaint: 'Notifications are totally down'."),
                ("Noor", "Triage: actually only push notifications on iOS 16; in-app notifications still appear."),
                ("Hassan", "Reference: similar symptoms mentioned in ticket #34, but that one seems Android-only."),
            ],
        ),
        _ticket(
            33,
            [
                ("Arjun", "Users report they are being 'charged twice' at checkout."),
                ("Grace", "Finance: no double charges in payment provider logs. Likely duplicate confirmation screens."),
                ("Arjun", "Support: users see two success messages if they refresh quickly."),
                ("Grace", "Engineering: this is a UI idempotency issue; money is not duplicated. Still causes high confusion and support load."),
            ],
        ),
        _ticket(
            34,
            [
                ("Mateo", "Android users: app freezes on startup on slow networks. 'Stuck forever'."),
                ("Emily", "Clarification: only first launch after install; subsequent launches are fine. Workaround: wait ~30 seconds."),
                ("Mateo", "Reference: ticket #35 suggests the underlying API is timing out for many requests, not just first launch."),
            ],
        ),
        _ticket(
            35,
            [
                ("Wei", "Multiple users: login intermittently fails with 'Something went wrong'."),
                ("Hannah", "On-call: correlates with auth service timeouts; affects all platforms during bursts."),
                ("Wei", "Cross-check: explains the startup freeze in ticket #34 (waiting on auth bootstrap)."),
                ("Hannah", "Impact: users cannot access accounts during timeout windows; widespread reports."),
            ],
        ),
    ]
    return criteria, tickets


def generate_problem_statement(
    difficulty: GenerationDifficulty = GenerationDifficulty.Medium,
) -> tuple[str, list[Ticket]]:
    """
    Generate a deterministic ticket-ordering problem.

    Important: the returned ticket list is already in the optimal order expected by the environment.
    The server will shuffle tickets for the agent, but it uses this list order as ground truth.
    """
    if difficulty == GenerationDifficulty.Easy:
        return _easy_problem()
    if difficulty == GenerationDifficulty.Medium:
        return _medium_problem()
    if difficulty == GenerationDifficulty.Hard:
        return _hard_problem()

    # Defensive fallback: treat unknown as medium.
    return _medium_problem()
