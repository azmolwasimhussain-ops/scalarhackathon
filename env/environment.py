import random
from .models import Observation, Action
from .grader import grade
from .tasks import TASKS

class TicketEnv:
    def __init__(self):
        self.tasks = TASKS
        self.index = 0
        self.current = None
        self.done = False
        self._graders = {task["name"]: True for task in TASKS}

    @property
    def graders(self):
        """Return grader status for each task."""
        return self._graders

    def reset(self):
        """Start a new one-step episode with the current task."""
        self.current = self.tasks[self.index]
        self.done = False
        return Observation(ticket_text=self.current["ticket"], history=[])

    def step(self, action: Action):
        """Score the submitted category and end the episode."""
        if self.current is None:
            self.reset()

        pred = action.category
        truth = self.current["label"]
        reward = grade(pred, truth)
        self.done = True

        return (
            Observation(ticket_text=self.current["ticket"], history=[]),
            reward,
            self.done,
            {"truth": truth}
        )

    def next_task(self):
        """Move to the next task."""
        self.index = (self.index + 1) % len(self.tasks)

    def state(self):
        """Return raw state for debugging and API introspection."""
        return self.current

    @staticmethod
    def _normalize_category(category: str) -> str:
        """Map common variants to canonical class labels."""
        value = (category or "").strip().lower()

        aliases = {
            "billing": "billing",
            "bill": "billing",
            "payment": "billing",
            "refund": "billing",
            "tech": "tech",
            "technical": "tech",
            "bug": "tech",
            "issue": "tech",
            "general": "general",
            "question": "general",
            "info": "general",
        }

        return aliases.get(value, value)