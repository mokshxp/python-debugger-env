from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from env import PythonDebuggerEnv
from models import Action, Observation, Reward
from tasks import TASKS

app = FastAPI(
    title="Python Debugger -- OpenEnv",
    description="A real-world OpenEnv environment where AI agents debug Python code.",
    version="1.0.0",
)

_envs: Dict[str, PythonDebuggerEnv] = {}

def _get_env(task_id: str) -> PythonDebuggerEnv:
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'. Use: {list(TASKS.keys())}")
    if task_id not in _envs:
        _envs[task_id] = PythonDebuggerEnv(task_id=task_id)
    return _envs[task_id]

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    model_config = {"extra": "ignore"}
    @classmethod
    def __get_validators__(cls):
        yield cls._validate
    @classmethod
    def _validate(cls, v):
        if v is None: return cls()
        return cls(**v) if isinstance(v, dict) else v

class StepRequest(BaseModel):
    task_id: str = "task_easy"
    code: str

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "environment": "python-debugger", "version": "1.0.0"}

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": getattr(t, 'id', None),
                "name": getattr(t, 'name', None),
                "difficulty": getattr(t, 'difficulty', None),
                "max_steps": getattr(t, 'max_steps', None),
            }
            for t in TASKS.values()
        ]
    }

@app.post("/reset", response_model=Dict[str, Any])
def reset(req: Optional[ResetRequest] = None):
    if req is None: req = ResetRequest()
    env = _get_env(req.task_id)
    obs = env.reset()
    return obs.model_dump()

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task_id)
    action = Action(code=req.code)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )

@app.get("/state")
def state(task_id: str = Query("task_easy")):
    env = _get_env(task_id)
    return env.state()

@app.post("/validate")
def validate():
    from models import Action
    results = {}
    for task_id in TASKS:
        env = PythonDebuggerEnv(task_id=task_id)
        obs = env.reset()
        assert obs.task_id == task_id
        assert obs.step == 0
        action = Action(code="print('test')")
        obs2, rew, done, info = env.step(action)
        assert isinstance(rew.value, float)
        state = env.state()
        assert "task_id" in state
        results[task_id] = "PASS"
    return {"validation": results, "status": "ALL_PASS"}
