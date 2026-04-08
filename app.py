import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env import SupportTicketEnv
import inference as inference_module

app = FastAPI(title="Ticket Routing Dashboard")

# FIX: Serve static assets from the project root (not a nonexistent ./static/ subfolder)
app.mount("/static", StaticFiles(directory="."), name="static")

class RunRequest(BaseModel):
    task: str
    hf_token: str

@app.get("/")
def get_root():
    # FIX: index.html lives in root, not static/index.html
    return FileResponse("index.html")

@app.post("/api/run")
def run_simulation(req: RunRequest):
    token = req.hf_token or ""
    os.environ["HF_TOKEN"] = token
    # FIX: also patch the module-level variable so build_client() sees the
    # request-time token instead of the empty string captured at import time
    inference_module.HF_TOKEN = token

    client = inference_module.build_client()
    env = SupportTicketEnv()

    try:
        obs = env.reset(task_name=req.task)
    except KeyError:
        obs = env.reset(task_name="easy")

    results = []
    step_count = 0

    while obs is not None:
        step_count += 1
        action, llm_error = inference_module.classify_ticket(client, obs)
        try:
            next_obs, reward, done, info = env.step(action)
            error_str = None
            if llm_error and info.get("error"):
                error_str = f"LLM error: {llm_error} | Env error: {info.get('error')}"
            elif llm_error:
                error_str = llm_error
            elif info.get("error"):
                error_str = info.get("error")

            results.append({
                "step": step_count,
                "ticket": obs,
                "action": action,
                "reward": reward,
                "error": error_str,
                "done": done
            })
            obs = next_obs
            if done:
                break
        except Exception as e:
            results.append({
                "step": step_count,
                "ticket": obs,
                "action": action,
                "reward": 0.0,
                "error": str(e),
                "done": True
            })
            break

    final_state = env.state()
    return {
        "score": final_state["episode_score"],
        "success": final_state["episode_score"] >= 0.5,
        "results": results,
        "steps": step_count
    }
