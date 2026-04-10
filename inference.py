import os
import requests
import json
import time
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")
BENCHMARK    = "python-debugger"

TASK_IDS    = ["task_easy", "task_medium", "task_hard"]
TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = """You are an expert Python engineer.
Debug and fix the given Python code so it passes all test cases.
Output ONLY the corrected Python code -- no explanations, no markdown, no backticks. Do not include ```python or ```."""

# ---------------------------------------------------------------------------
# Structured logging STRICT FORMAT
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_str = repr(action[:100]) + ("..." if len(action)>100 else "") # truncated for logging safely
    action_str = "code(...)"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Env Client 
# ---------------------------------------------------------------------------
def env_reset(base_url, task_id):
    r = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(base_url, task_id, code):
    r = requests.post(f"{base_url}/step", json={"task_id": task_id, "code": code}, timeout=30)
    r.raise_for_status()
    return r.json()

def build_prompt(obs):
    return f"""CURRENT CODE:
{obs.get('current_code', '')}

FEEDBACK:
{obs.get('feedback', '')}

Write the corrected Python code:"""

def run_task(client, base_url, task_id):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    obs = env_reset(base_url, task_id)
    max_steps = obs.get("max_steps", 10)
    
    done = False
    step = 0
    best_score = 0.0
    rewards = []
    
    try:
        while not done and step < max_steps:
            step += 1
            user_prompt = build_prompt(obs)
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                code_output = completion.choices[0].message.content or ""
            except Exception as e:
                code_output = obs.get('current_code', '') # fallback
            
            code_output = code_output.replace("```python", "").replace("```", "").strip()
            
            result = env_step(base_url, task_id, code_output)
            obs = result["observation"]
            reward_val = result["reward"]["value"]
            done = result["done"]
            
            score = obs.get("score", 0.0)
            best_score = max(best_score, score)
            rewards.append(reward_val)
            
            error_val = "" if result["observation"]["test_results"]["success"] else result["observation"]["test_results"]["error"]
            log_step(step=step, action=code_output, reward=reward_val, done=done, error=error_val if error_val else None)
            
    except Exception as e:
        log_end(success=False, steps=step, score=best_score, rewards=rewards)
        return
        
    success = best_score >= 0.8
    log_end(success=success, steps=step, score=best_score, rewards=rewards)

def main():
    base_url = ENV_URL
    import time
    time.sleep(2) # ensure env is up
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_id in TASK_IDS:
        run_task(client, base_url, task_id)

if __name__ == "__main__":
    main()
