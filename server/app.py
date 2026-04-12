import argparse
import json

from fastapi import FastAPI, HTTPException

from env.environment import TicketEnv
from env.models import Action

app = FastAPI(title="Ticket Routing Environment API")
ticket_env = TicketEnv()


@app.get("/")
def health():
    return {"status": "ok", "message": "Ticket Routing OpenEnv API is running"}


@app.post("/reset")
def reset(task: str | None = None):
    try:
        return ticket_env.reset(task_name=task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/reset")
def reset_get(task: str | None = None):
    try:
        return ticket_env.reset(task_name=task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = ticket_env.step(action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return ticket_env.state()


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Ticket Routing environment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=7860)

    reset_parser = subparsers.add_parser("reset", help="Reset environment")
    reset_parser.add_argument("--task", choices=["easy", "medium", "hard"], default=None)

    step_parser = subparsers.add_parser("step", help="Submit action category")
    step_parser.add_argument("category", choices=["billing", "tech", "general"])
    step_parser.add_argument("--task", choices=["easy", "medium", "hard"], default=None)

    subparsers.add_parser("state", help="Show current raw state")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.command == "reset":
        observation = ticket_env.reset(task_name=args.task)
        print(observation.model_dump_json())
        return

    if args.command == "step":
        if ticket_env.state() is None:
            ticket_env.reset(task_name=args.task)
        observation, reward, done, info = ticket_env.step(Action(category=args.category))
        print(
            json.dumps(
                {
                    "observation": observation.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )
        )
        return

    if args.command == "state":
        print(json.dumps(ticket_env.state()))
        return


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
