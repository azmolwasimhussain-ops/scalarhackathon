#!/usr/bin/env python3
"""
inference.py — Agent loop for the Customer Support Ticket Routing Environment.

Environment variables
---------------------
API_BASE_URL   Base URL of an OpenAI-compatible endpoint
               default: https://router.huggingface.co/v1
MODEL_NAME     Model identifier to use
               default: Qwen/Qwen2.5-72B-Instruct
HF_TOKEN       HuggingFace API token (used as the Bearer token)
TASK_NAME      Task difficulty to run  [easy | medium | hard]
               default: easy

Log format (EXACT — do not deviate)
------------------------------------
[START] task=<task_name> env=support-ticket model=<model_name>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=null
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from env import SupportTicketEnv, VALID_ACTIONS

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "")
TASK_NAME: str    = os.getenv("TASK_NAME", "easy")


SYSTEM_PROMPT: str = textwrap.dedent(
    """
    You are an expert customer support ticket classifier.

    Your job is to read a customer support ticket and classify it into
    EXACTLY ONE of the following categories:

      - billing   : payment issues, invoices, refunds, charges, subscriptions
      - technical : software bugs, crashes, connectivity, API errors, login issues
      - general   : general questions, product info, policies, how-to queries
      - urgent    : life-safety, security breach, account locked, production down,
                    time-critical emergencies

    Rules:
    1. Reply with ONLY the single category word — no punctuation, no explanation.
    2. Your reply must be one of: billing, technical, general, urgent
    3. Choose the PRIMARY intent when a ticket has multiple themes.
    """
).strip()


def build_client() -> OpenAI:
    api_key = HF_TOKEN if HF_TOKEN else "hf-no-key"
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def rule_based_fallback(ticket_text: str) -> str:
    """Keyword-based classifier used when the LLM API is unavailable.

    Priority order:
      1. urgent   — safety / security / time-critical signals
      2. billing  — payment / invoice / refund signals  (checked BEFORE technical
                    so "payment error" routes to billing, not technical)
      3. technical — crash / login / error / API signals
      4. general  — catch-all default
    """
    text_lower = ticket_text.lower()

    urgent_keywords    = {"urgent", "asap", "emergency", "immediately", "stranded",
                          "hack", "hacked", "breach", "outage", "down", "safety"}
    billing_keywords   = {"payment", "invoice", "refund", "charge", "charged",
                          "billing", "subscription", "bill", "fee", "overcharged"}
    technical_keywords = {"crash", "login", "error", "bug", "api", "connectivity",
                          "driver", "install", "update", "503", "latency", "webhook"}

    if any(kw in text_lower for kw in urgent_keywords):
        return "urgent"
    if any(kw in text_lower for kw in billing_keywords):
        return "billing"
    if any(kw in text_lower for kw in technical_keywords):
        return "technical"
    return "general"


def classify_ticket(client: OpenAI, ticket_text: str) -> Tuple[str, Optional[str]]:
    """Call the LLM and return a normalised action string.

    Falls back to rule_based_fallback() on any API failure or invalid response.
    Returns (action, error_message).
    """
    action: Optional[str] = None
    error_msg: Optional[str] = None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ticket_text},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        raw: str = response.choices[0].message.content.strip().lower()
        action = raw.strip(".,;:!?\"'").split()[0] if raw else ""

        if action not in VALID_ACTIONS:
            error_msg = f"model returned unexpected value '{raw}', fallback applied"
            action = rule_based_fallback(ticket_text)

    except Exception as exc:
        error_msg = str(exc)
        action = rule_based_fallback(ticket_text)

    return action, error_msg


# ---------------------------------------------------------------------------
# Logging helpers (EXACT format required)
# ---------------------------------------------------------------------------
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=support-ticket model={model}")


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_str = "true" if done else "false"
    if error is None:
        error_str = "null"
    else:
        clean_error = str(error).replace("\n", " ").replace("\r", " ")
        error_str = (
            f"\"{clean_error}\""
            if (" " in clean_error or "=" in clean_error)
            else clean_error
        )
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_str}"
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_str   = f"{score:.4f}"
    print(
        f"[END] success={success_str} steps={steps} score={score_str} "
        f"rewards={rewards_str}"
    )


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------
def run_agent(task_name: str = TASK_NAME) -> None:
    """Execute the full agent episode for *task_name* and print structured logs."""
    client = build_client()
    env    = SupportTicketEnv()

    try:
        obs = env.reset(task_name=task_name)
    except KeyError:
        obs = env.reset(task_name="easy")
        task_name = "easy"

    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    step_count: int = 0

    while obs is not None:
        step_count += 1
        action, llm_error = classify_ticket(client, obs)

        try:
            obs, reward, done, info = env.step(action)
            env_error = info.get("error")

            final_error = None
            if llm_error and env_error:
                final_error = f"LLM error: {llm_error} | Env error: {env_error}"
            elif llm_error:
                final_error = llm_error
            elif env_error:
                final_error = env_error

            rewards.append(reward)
            log_step(step=step_count, action=action, reward=reward,
                     done=done, error=final_error)

            if done:
                break

        except Exception as step_exc:
            log_step(step=step_count, action=action, reward=0.0,
                     done=True, error=f"fatal environment error: {step_exc}")
            rewards.append(0.0)
            break

    final_state = env.state()
    score   = final_state["episode_score"]
    success = score >= 0.5

    log_end(success=success, steps=step_count, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # FIX: removed unused `import builtins`
    sys.stdout.reconfigure(line_buffering=True)
    task = sys.argv[1] if len(sys.argv) > 1 else TASK_NAME
    run_agent(task_name=task)
