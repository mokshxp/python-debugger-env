from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
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

def run_agent_on_task(task_id: str):
    from openai import OpenAI
    import os
    
    # Get token from environment (HF Secrets)
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        return {"error": "HF_TOKEN secret not found in Space settings."}
        
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=api_key)
    env = _get_env(task_id)
    obs = env.reset()
    
    # Run simple 1-step test for demo
    user_prompt = f"CURRENT CODE:\n{obs.current_code}\n\nFEEDBACK: {obs.feedback}\n\nWrite the corrected Python code:"
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "system", "content": "You are an expert Python engineer. Output ONLY raw code."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    code = completion.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    obs, reward, done, _ = env.step(Action(code=code))
    
    return {
        "task_id": task_id,
        "agent_code": code,
        "success": obs.test_results["success"],
        "reward": reward.value,
        "error": obs.test_results["error"]
    }

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

@app.get("/", response_class=HTMLResponse)
def home():
    tasks_html = "".join([
        f"""
        <div class="task-card">
            <div class="task-header">
                <span class="difficulty {t.difficulty}">{t.difficulty.upper()}</span>
                <h3>{t.name}</h3>
            </div>
            <p>ID: <code>{t.id}</code> | Max Steps: {t.max_steps}</p>
            <button class="run-btn" onclick="runAgent('{t.id}')">Run AI Agent Demo</button>
            <div id="result-{t.id}" class="result-box" style="display:none;"></div>
        </div>
        """ for t in TASKS.values()
    ])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenEnv | Python Debugger</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0a0a0c;
                --card-bg: rgba(255, 255, 255, 0.03);
                --primary: #4f46e5;
                --accent: #d946ef;
                --text: #f8fafc;
            }}
            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                background-image: radial-gradient(circle at 50% -20%, #1e1b4b 0%, var(--bg) 80%);
            }}
            .container {{ width: 100%; max-width: 800px; padding: 40px 20px; }}
            header {{ text-align: center; margin-bottom: 60px; }}
            h1 {{ font-weight: 800; font-size: 3rem; margin: 0; background: linear-gradient(135deg, #fff 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            .status-badge {{ display: inline-block; padding: 6px 12px; border-radius: 20px; background: rgba(34, 197, 94, 0.1); color: #4ade80; font-size: 0.8rem; border: 1px solid rgba(34, 197, 94, 0.2); margin-top: 10px; }}
            
            .dashboard {{ display: grid; gap: 20px; }}
            .task-card {{ background: var(--card-bg); border: 1px solid rgba(255,255,255,0.05); padding: 24px; border-radius: 16px; backdrop-filter: blur(10px); transition: transform 0.2s; }}
            .task-card:hover {{ transform: translateY(-4px); border-color: rgba(255,255,255,0.1); }}
            .task-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
            .difficulty {{ font-size: 0.7rem; font-weight: 800; padding: 2px 8px; border-radius: 4px; }}
            .easy {{ background: rgba(34, 197, 94, 0.1); color: #4ade80; }}
            .medium {{ background: rgba(234, 179, 8, 0.1); color: #facc15; }}
            .hard {{ background: rgba(239, 68, 68, 0.1); color: #f87171; }}
            
            .api-info {{ margin-top: 60px; padding: 30px; background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(217, 70, 239, 0.1)); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); }}
            code {{ background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #818cf8; }}
            .links {{ margin-top: 20px; }}
            a {{ color: #818cf8; text-decoration: none; font-weight: 600; margin-right: 20px; }}
            a:hover {{ text-decoration: underline; }}

            .run-btn {{ margin-top: 15px; background: var(--primary); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }}
            .run-btn:hover {{ opacity: 0.9; }}
            .result-box {{ margin-top: 15px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; font-size: 0.85rem; border-left: 4px solid var(--primary); white-space: pre-wrap; }}
        </style>
        <script>
            async function runAgent(taskId) {{
                const btn = event.target;
                const resBox = document.getElementById('result-' + taskId);
                btn.disabled = true;
                btn.innerText = 'Agent Thinking...';
                resBox.style.display = 'block';
                resBox.innerText = 'Sending task to AI...';
                
                try {{
                    const response = await fetch('/demo/' + taskId, {{ method: 'POST' }});
                    const data = await response.json();
                    if (data.error) {{
                        resBox.innerHTML = '<span style="color:#f87171">Error: ' + data.error + '</span>';
                    }} else {{
                        resBox.innerHTML = '<b>Result:</b> ' + (data.success ? '✅ Success' : '❌ Failed') + 
                                         '\\n<b>Reward:</b> ' + data.reward + 
                                         '\\n<b>Code:</b>\\n<code>' + data.agent_code + '</code>';
                    }}
                }} catch (e) {{
                    resBox.innerText = 'Error connection to API.';
                }} finally {{
                    btn.disabled = false;
                    btn.innerText = 'Run AI Agent Demo';
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>OpenEnv Debugger</h1>
                <div class="status-badge">● Environment Active</div>
            </header>
            
            <div class="dashboard">
                {tasks_html}
            </div>
            
            <div class="api-info">
                <h3>Developer API</h3>
                <p>This space is an OpenEnv-compliant environment. Connect your agent using the <code>/reset</code> and <code>/step</code> endpoints.</p>
                <div class="links">
                    <a href="/docs">📚 API Documentation</a>
                    <a href="/health">✅ Health Check</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok", "environment": "python-debugger", "version": "1.0.0"}

@app.post("/demo/{task_id}")
def run_demo(task_id: str):
    return run_agent_on_task(task_id)

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
