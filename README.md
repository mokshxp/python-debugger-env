---
title: OpenEnv Python Debugger
emoji: 🐍
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
  - python
  - debugging
---

# OpenEnv Python Debugger

A real-world OpenEnv environment where an AI agent learns to write, debug, and optimize Python code by working with a live Python interpreter.

## The Environment

This environment simulates a core task software engineers do daily: taking broken code and making it pass tests.

**Action Space:**
The agent outputs raw Python code:
```json
{
  "code": "def my_func():\\n    return True"
}
```

**Observation Space:**
The environment responds with test execution details:
```json
{
  "task_id": "task_easy",
  "current_code": "def my_func():\\n    return False",
  "test_results": {
    "success": false,
    "tests_passed": 0,
    "total_tests": 1,
    "error": "AssertionError at test_my_func",
    "stdout": ""
  },
  "feedback": "Tests failed:\\nAssertionError at test_my_func",
  "step": 1,
  "max_steps": 10,
  "score": 0.0
}
```

## Tasks

1. **Python Syntax Fix (Easy):** Fix straightforward syntax and semantic errors.
2. **Python Logic Repair (Medium):** Fix algorithmic bugs in correctly formatted code.
3. **Algorithm Optimization (Hard):** Optimize a slow O(N^2) algorithm into an O(N) one to bypass timeout tests.

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the environment:
```bash
uvicorn app:app --port 7860 --host 0.0.0.0
```

3. Run the baseline:
```bash
export API_BASE_URL="your-llm-base-url"
export API_KEY="your-api-key"
export MODEL_NAME="your-model-name"
python inference.py
```

## Validation

This environment passes `openenv validate` out of the box.
