import random
from .models import Observation, Action
from .grader import grade
from .tasks import TASKS

class TicketEnv:
    def __init__(self):
        self.named_tasks = TASKS
        self.tasks = list(TASKS.values())
        self.current = None
        self.done = False

    def reset(self, task_name: str | None = None):
        """Start a new one-step episode with a random or named ticket."""
        if task_name:
            selected = self.named_tasks.get(task_name.lower())
            if selected is None:
                available = ", ".join(sorted(self.named_tasks.keys()))
                raise ValueError(f"Unknown task '{task_name}'. Available: {available}")
            self.current = selected
        else:
            self.current = random.choice(self.tasks)
        self.done = False
        return Observation(ticket_text=self.current["text"], history=[])

    def step(self, action: Action):
        """Score the submitted category and end the episode."""
        if self.current is None:
            self.reset()

        correct = self.current["label"]
        prediction = self._normalize_category(action.category)
        reward = grade(prediction, correct)
        self.done = True

        return (
            Observation(ticket_text=self.current["text"], history=[]),
            reward,
            self.done,
            {"correct": correct}
        )

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