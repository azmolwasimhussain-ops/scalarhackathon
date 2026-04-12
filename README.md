---
title: Scalar Open Env Ticket Routing
emoji: 📊
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Ticket Routing OpenEnv

This project provides a small support-ticket classification environment for OpenEnv / Scalar validation.

## What it does

- `app.py` exposes a FastAPI service for the environment.
- `env/environment.py` contains the ticket environment logic.
- `interference.py` runs a simple inference loop against the environment.
- `openenv.yaml` describes the environment for the validator.

## Environment Behavior

The environment is one-step and returns a ticket with a label:

- `billing`
- `tech`
- `general`

Use `reset()` to start a new ticket, `step()` to submit a category, and `state()` to inspect the current ticket.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Run the inference script

```bash
python interference.py
```

## Validation

The project is set up to pass the OpenEnv validator.

```bash
openenv validate .
```

## Project Files

- `app.py` - API server
- `server/app.py` - validator entrypoint
- `env/environment.py` - environment implementation
- `env/models.py` - Pydantic models
- `interference.py` - inference runner
- `openenv.yaml` - OpenEnv metadata
- `pyproject.toml` - packaging and validator metadata
