import os
from openai import OpenAI
from env.models import Action
from env.environment import TicketEnv
from env.tasks import TASKS

MODEL_NAME = os.getenv("MODEL_NAME")


def build_client():
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url=os.getenv("API_BASE_URL"), api_key=api_key)


def heuristic_classifier(ticket_text: str) -> str:
    text = ticket_text.lower()
    if "refund" in text or "charged" in text or "payment" in text:
        return "billing"
    if "error" in text or "crash" in text or "not working" in text:
        return "tech"
    return "general"


def normalize_action(raw_text: str) -> str:
    cleaned = (raw_text or "").strip().lower()
    if "billing" in cleaned:
        return "billing"
    if "tech" in cleaned:
        return "tech"
    if "general" in cleaned:
        return "general"
    return "general"


def model_classifier(client: OpenAI, ticket_text: str) -> str:
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    prompt = (
        "Classify this support ticket into one label: billing, tech, or general. "
        f"Ticket: {ticket_text}"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return normalize_action(response.choices[0].message.content)


def choose_action(client: OpenAI, ticket_text: str, model_name: str) -> str:
    if client and model_name:
        try:
            return model_classifier(client, ticket_text)
        except Exception as e:
            print(f"[WARNING] Model API call failed: {e}, falling back to heuristic")
    return heuristic_classifier(ticket_text)


def main():
    client = build_client()
    env = TicketEnv()
    task_name = os.getenv("TASK_NAME", "hard").lower()
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    if task_name not in TASKS:
        task_name = "hard"

    print(f"[START] task={task_name} env=ticket-routing model={model_name}")

    observation = env.reset(task_name=task_name)
    done = False
    step_count = 0
    rewards = []

    while not done:
        step_count += 1
        action_label = choose_action(client, observation.ticket_text, model_name)
        observation, reward, done, _info = env.step(Action(category=action_label))
        rewards.append(reward)
        print(
            f"[STEP] step={step_count} action={action_label} "
            f"reward={reward:.2f} done={str(done).lower()} error=null"
        )

    score = sum(rewards) / len(rewards)
    print(
        f"[END] success=true steps={step_count} score={score:.2f} "
        f"rewards={','.join([f'{r:.2f}' for r in rewards])}"
    )


if __name__ == "__main__":
    main()
