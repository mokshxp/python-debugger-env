import subprocess
import tempfile
import os
import json
from typing import Dict, Any, Tuple

from models import Observation, Action, Reward
from tasks import TASKS

class PythonDebuggerEnv:
    def __init__(self, task_id: str = "task_easy"):
        if task_id not in TASKS:
            task_id = "task_easy"
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.step_count = 0
        self.best_score = 0.0
        self.current_code = self.task.initial_code

    def reset(self) -> Observation:
        self.step_count = 0
        self.best_score = 0.0
        self.current_code = self.task.initial_code
        
        return Observation(
            task_id=self.task_id,
            current_code=self.current_code,
            test_results={"success": False, "tests_passed": 0, "total_tests": 1, "error": "", "stdout": ""},
            feedback="Initial broken code provided. Fix the code to pass the tests.",
            step=self.step_count,
            max_steps=self.task.max_steps,
            score=self.best_score
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        self.current_code = action.code
        
        # Indent the user's code so we can inject it inside the test function template if needed
        indented_code = "\n    ".join(self.current_code.split("\n"))
        executable_script = self.task.test_code_template.replace("{code}", indented_code)
        
        success = False
        error_msg = ""
        stdout = ""
        tests_passed = 0
        total_tests = 1
        
        timeout_seconds = 2.0
        try:
            with open("temp_test.py", "w") as f:
                f.write(executable_script)
            
            result = subprocess.run(
                ["python", "temp_test.py"], 
                capture_output=True, 
                text=True, 
                timeout=timeout_seconds
            )
            
            stdout = result.stdout
            if result.returncode == 0:
                # Execution succeeded, let's parse the JSON emitted by the test wrapper
                try:
                    data = json.loads(stdout.strip().split("\n")[-1])
                    tests_passed = data.get("passed", 0)
                    total_tests = data.get("total", 1)
                    errors = data.get("errors", [])
                    success = (tests_passed == total_tests)
                    if not success:
                        error_msg = "\n".join(errors)
                except Exception as parse_e:
                    error_msg = "Parse error in test output: " + str(parse_e) + "\nRaw output: " + stdout
            else:
                error_msg = result.stderr or result.stdout
        except subprocess.TimeoutExpired:
            error_msg = f"TimeoutError: Execution exceeded {timeout_seconds}s limit. Your algorithm is too slow or has an infinite loop."
        except Exception as e:
            error_msg = str(e)
        finally:
            if os.path.exists("temp_test.py"):
                os.remove("temp_test.py")

        # Meaningful reward logic based on partial progress
        if total_tests > 0:
            score = tests_passed / total_tests
        else:
            score = 0.0
            
        reward_value = max(0.0, score - self.best_score)
        self.best_score = max(self.best_score, score)
        
        done = success or (self.step_count >= self.task.max_steps)
        
        if success:
            feedback = "All tests passed! Great job."
        else:
            feedback = f"Passed {tests_passed}/{total_tests} tests.\nErrors:\n{error_msg[-500:]}"
            
        obs = Observation(
            task_id=self.task_id,
            current_code=self.current_code,
            test_results={
                "success": success,
                "tests_passed": tests_passed,
                "total_tests": total_tests,
                "error": error_msg,
                "stdout": stdout,
            },
            feedback=feedback,
            step=self.step_count,
            max_steps=self.task.max_steps,
            score=self.best_score
        )
        
        reward = Reward(
            value=reward_value,
            correctness=score,
            efficiency=1.0 if success else 0.0
        )

        return obs, reward, done, {"info": "executed"}

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": self.step_count,
            "best_score": self.best_score,
            "current_code": self.current_code
        }
