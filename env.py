"""
env.py — Customer Support Ticket Routing Environment (OpenEnv compliant).

The agent receives a support ticket (string) as its observation and must
classify it into one of four routing categories.  A grader compares the
agent's choice to the ground-truth label and returns a scalar reward.

Action space  : ["billing", "technical", "general", "urgent"]
Observation   : ticket text (str)
Reward        : 1.0 (correct) | 0.5 (partial) | 0.0 (wrong)
Episode length: one step per ticket in the active task
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tasks import TaskSet, get_task, list_tasks


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_ACTIONS: List[str] = ["billing", "technical", "general", "urgent"]

# Partial-credit pairs: pairs that are "close" to each other, granting 0.5
PARTIAL_CREDIT_PAIRS: List[frozenset] = [
    frozenset({"billing", "general"}),
    frozenset({"technical", "urgent"}),
    frozenset({"billing", "urgent"}),
]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------
class SupportTicketGrader:
    """Deterministic grader for support-ticket routing actions."""

    def score(self, action: str, label: str) -> float:
        """Return a reward in {0.0, 0.5, 1.0}.

        Parameters
        ----------
        action : str
            The category chosen by the agent.
        label : str
            The ground-truth category for the ticket.

        Returns
        -------
        float
            1.0  — exact match
            0.5  — partial credit (plausible near-miss)
            0.0  — wrong
        """
        action = action.strip().lower()
        label  = label.strip().lower()

        if action == label:
            return 1.0

        pair = frozenset({action, label})
        if pair in PARTIAL_CREDIT_PAIRS:
            return 0.5

        return 0.0

    def episode_score(self, rewards: List[float]) -> float:
        """Normalise total reward to [0, 1].

        Uses the maximum possible reward (1.0 per step) as the denominator.
        """
        if not rewards:
            return 0.0
        max_possible = len(rewards) * 1.0
        raw_total    = sum(rewards)
        # Clip to [0, max_possible] then normalise
        clipped      = max(0.0, min(raw_total, max_possible))
        return clipped / max_possible


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class SupportTicketEnv:
    """OpenEnv-compliant reinforcement-learning environment.

    Usage
    -----
    >>> env = SupportTicketEnv()
    >>> obs = env.reset(task_name="easy")
    >>> while True:
    ...     action = agent.act(obs)
    ...     obs, reward, done, info = env.step(action)
    ...     if done:
    ...         break
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.grader: SupportTicketGrader = SupportTicketGrader()

        # Episode state (initialised in reset)
        self._task: Optional[TaskSet] = None
        self._tickets: List[Dict[str, str]] = []
        self._cursor: int = 0
        self._rewards: List[float] = []
        self._done: bool = True

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------
    def reset(self, task_name: str = "easy") -> str:
        """Start a new episode for the given task.

        Parameters
        ----------
        task_name : str
            One of ``list_tasks()`` — ``"easy"``, ``"medium"``, or ``"hard"``.

        Returns
        -------
        str
            The first ticket text (the initial observation).
        """
        self._task    = get_task(task_name)
        self._tickets = list(self._task["tickets"])   # shallow copy — deterministic order
        self._cursor  = 0
        self._rewards = []
        self._done    = False

        return self._current_observation()

    def step(self, action: str) -> Tuple[Optional[str], float, bool, Dict[str, Any]]:
        """Process one agent action and advance the environment.

        Parameters
        ----------
        action : str
            The routing category chosen by the agent.

        Returns
        -------
        observation : str | None
            Next ticket text, or ``None`` when the episode is finished.
        reward : float
            Reward for this action.
        done : bool
            ``True`` when all tickets have been processed.
        info : dict
            Auxiliary data: ``{"correct_label": ..., "step": ...,
                               "episode_score": ..., "error": ...}``
        """
        if self._done:
            raise RuntimeError(
                "Episode is already finished.  Call reset() to start a new one."
            )

        action = action.strip().lower()

        error: Optional[str] = None
        if action not in VALID_ACTIONS:
            error  = f"Invalid action '{action}'. Must be one of {VALID_ACTIONS}."
            action = ""     # Force a wrong answer
            reward = 0.0
        else:
            current_ticket = self._tickets[self._cursor]
            reward = self.grader.score(action, current_ticket["label"])

        self._rewards.append(reward)
        self._cursor += 1

        if self._cursor >= len(self._tickets):
            self._done = True

        obs = self._current_observation() if not self._done else None

        info: Dict[str, Any] = {
            "correct_label":  self._tickets[self._cursor - 1]["label"],
            "step":           self._cursor,
            "episode_score":  self.grader.episode_score(self._rewards),
            "error":          error,
        }

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return a snapshot of the current environment state.

        Conforms to the OpenEnv ``state()`` contract.
        """
        task_name = self._task["name"] if self._task else None
        return {
            "task":          task_name,
            "total_tickets": len(self._tickets),
            "current_step":  self._cursor,
            "done":          self._done,
            "rewards":       list(self._rewards),
            "episode_score": self.grader.episode_score(self._rewards),
            "valid_actions": VALID_ACTIONS,
        }

    # ------------------------------------------------------------------
    # Properties & helpers
    # ------------------------------------------------------------------
    @property
    def action_space(self) -> List[str]:
        """List of valid action strings."""
        return list(VALID_ACTIONS)

    @property
    def task_name(self) -> Optional[str]:
        """Name of the currently active task, or None before reset()."""
        return self._task["name"] if self._task else None

    def _current_observation(self) -> Optional[str]:
        """Return the text of the ticket at the current cursor position."""
        if self._cursor < len(self._tickets):
            return self._tickets[self._cursor]["text"]
        return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SupportTicketEnv(task={self.task_name!r}, "
            f"step={self._cursor}/{len(self._tickets)}, "
            f"done={self._done})"
        )
