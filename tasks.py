"""
tasks.py — Task definitions for the Customer Support Ticket Routing Environment.

Each task set contains:
  - A list of ticket dicts: {"text": str, "label": str}
  - Three difficulty levels: easy, medium, hard
"""

from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Ticket = Dict[str, str]   # {"text": ..., "label": ...}
TaskSet = Dict[str, Any]  # {"name": str, "description": str, "tickets": [...]}


# ---------------------------------------------------------------------------
# EASY — tickets have clear, unambiguous intent and explicit keywords
# ---------------------------------------------------------------------------
EASY_TASK: TaskSet = {
    "name": "easy",
    "description": (
        "Straightforward tickets with explicit keywords that make the correct "
        "category immediately obvious."
    ),
    "tickets": [
        {
            "text": (
                "I was charged twice for my subscription this month. "
                "Please refund the duplicate payment immediately."
            ),
            "label": "billing",
        },
        {
            "text": (
                "My internet connection keeps dropping every few minutes. "
                "I've already restarted the router but the problem persists."
            ),
            "label": "technical",
        },
        {
            "text": (
                "I would like to know the store's opening hours for the "
                "holiday season."
            ),
            "label": "general",
        },
        {
            "text": (
                "There is smoke coming out of my device and it feels very hot. "
                "This is an emergency — please help right now!"
            ),
            "label": "urgent",
        },
        {
            "text": (
                "Can you send me a copy of my latest invoice? I need it for "
                "my tax records."
            ),
            "label": "billing",
        },
        {
            "text": (
                "The app keeps crashing whenever I try to open the settings "
                "screen on my Android phone."
            ),
            "label": "technical",
        },
        {
            "text": (
                "What documents do I need to provide to create a new account?"
            ),
            "label": "general",
        },
        {
            "text": (
                "My account has been hacked! Someone is making purchases "
                "without my permission. Please lock it immediately!"
            ),
            "label": "urgent",
        },
    ],
}


# ---------------------------------------------------------------------------
# MEDIUM — tickets are slightly ambiguous; one category dominates but others
# are plausible
# ---------------------------------------------------------------------------
MEDIUM_TASK: TaskSet = {
    "name": "medium",
    "description": (
        "Slightly ambiguous tickets where the correct category is clear with "
        "careful reading but surface-level keywords could mislead."
    ),
    "tickets": [
        {
            "text": (
                "I updated my payment method last week but I'm still seeing "
                "the old card being charged. Is there a system delay, or is "
                "this a bug in your billing portal?"
            ),
            "label": "billing",
        },
        {
            "text": (
                "Ever since the latest software update my printer stopped "
                "working with your application. I've reinstalled the driver "
                "twice with no luck."
            ),
            "label": "technical",
        },
        {
            "text": (
                "I heard you launched a loyalty rewards program. Could you "
                "explain how points are earned and whether existing customers "
                "qualify?"
            ),
            "label": "general",
        },
        {
            "text": (
                "A customer in our store is threatening staff and refusing to "
                "leave. We need guidance on next steps right away."
            ),
            "label": "urgent",
        },
        {
            "text": (
                "My bill this quarter seems higher than usual. I didn't change "
                "my plan, so I'm wondering if there's an error or a hidden fee "
                "that was added."
            ),
            "label": "billing",
        },
        {
            "text": (
                "The two-factor authentication SMS is not arriving on my phone. "
                "I've checked my signal and the number on file is correct."
            ),
            "label": "technical",
        },
        {
            "text": (
                "Do you offer student discounts, and if so, what proof of "
                "enrollment is required to apply?"
            ),
            "label": "general",
        },
        {
            "text": (
                "Our production server went down 10 minutes ago and we're "
                "losing thousands of dollars per minute. We need an engineer "
                "on the line immediately."
            ),
            "label": "urgent",
        },
        {
            "text": (
                "I cancelled my subscription three weeks ago but was still "
                "billed this month. I have the cancellation confirmation email."
            ),
            "label": "billing",
        },
        {
            "text": (
                "Your API keeps returning a 503 error for the past hour. "
                "All other services are running fine on our end."
            ),
            "label": "technical",
        },
    ],
}


# ---------------------------------------------------------------------------
# HARD — complex, multi-intent, or emotionally charged tickets that require
# careful reasoning to identify the primary category
# ---------------------------------------------------------------------------
HARD_TASK: TaskSet = {
    "name": "hard",
    "description": (
        "Complex, multi-intent, or emotionally charged tickets. The agent must "
        "identify the SINGLE primary category despite competing signals."
    ),
    "tickets": [
        {
            "text": (
                "I've been a loyal customer for five years and this is the "
                "third time I've been overcharged. Your technical support told "
                "me last month it was a software glitch, but I'm still being "
                "billed incorrectly. I want a full refund and a written "
                "explanation or I'm contacting my bank."
            ),
            "label": "billing",
        },
        {
            "text": (
                "After the system migration you pushed last night, none of our "
                "team can log in. We're a hospital and our patient management "
                "software is completely inaccessible. Patient safety is at risk."
            ),
            "label": "urgent",
        },
        {
            "text": (
                "I'm trying to understand your enterprise pricing tiers. Your "
                "website mentions volume discounts but doesn't specify the "
                "thresholds, and the sales rep quoted me a different number "
                "than what's in the contract I received."
            ),
            "label": "billing",
        },
        {
            "text": (
                "The webhook integration we set up is firing duplicate events "
                "intermittently — roughly 1 in 50 requests triggers a double "
                "callback. The issue seems to happen under high load. Here's "
                "our request ID: REQ-4892. We're on the Pro plan."
            ),
            "label": "technical",
        },
        {
            "text": (
                "I need to know your data retention policy, specifically "
                "whether customer PII is deleted within 30 days of account "
                "closure as required by GDPR. Our legal team is asking and "
                "we're up against a compliance deadline."
            ),
            "label": "general",
        },
        {
            "text": (
                "A fraud alert on my account has frozen all transactions. "
                "I'm traveling abroad right now and I can't access funds or "
                "pay for my hotel. I need this resolved in the next 30 minutes "
                "or I'll be stranded."
            ),
            "label": "urgent",
        },
        {
            "text": (
                "Your mobile app charged me for a premium feature I never "
                "enabled. When I try to request a refund through the app, it "
                "crashes. I've also submitted a ticket through your portal but "
                "got no response in 5 days. This is unacceptable."
            ),
            "label": "billing",
        },
        {
            "text": (
                "We're running a load test and noticed your service degrades "
                "significantly above 500 concurrent requests — latency spikes "
                "from 120 ms to over 4 seconds. Is there a rate limit we're "
                "hitting, or is this a scalability issue on your side? We're "
                "evaluating whether to continue with your platform."
            ),
            "label": "technical",
        },
        {
            "text": (
                "I want to understand the SLA differences between your "
                "Business and Enterprise plans, particularly around uptime "
                "guarantees and incident response times. My company is in "
                "procurement talks and needs this information for a board "
                "presentation tomorrow."
            ),
            "label": "general",
        },
        {
            "text": (
                "A pipeline running critical nightly batch jobs failed silently "
                "— no alerts were triggered. We only discovered it because a "
                "downstream report was missing. The silent failure means we "
                "have corrupted data in production. We need an engineer and a "
                "data recovery plan immediately."
            ),
            "label": "urgent",
        },
    ],
}


# ---------------------------------------------------------------------------
# Public registry — index by name for easy lookup
# ---------------------------------------------------------------------------
ALL_TASKS: Dict[str, TaskSet] = {
    "easy":   EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard":   HARD_TASK,
}


def get_task(name: str) -> TaskSet:
    """Return a task set by name.  Raises KeyError for unknown names."""
    if name not in ALL_TASKS:
        raise KeyError(
            f"Unknown task '{name}'. Available tasks: {list(ALL_TASKS.keys())}"
        )
    return ALL_TASKS[name]


def list_tasks() -> List[str]:
    """Return sorted list of available task names."""
    return sorted(ALL_TASKS.keys())
