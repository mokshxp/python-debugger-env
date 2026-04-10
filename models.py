from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class Observation(BaseModel):
    task_id: str
    current_code: str
    test_results: Dict[str, Any]
    feedback: str
    step: int
    max_steps: int
    score: float

class Action(BaseModel):
    code: str

class Reward(BaseModel):
    value: float
    correctness: float
    efficiency: float
