from dataclasses import dataclass

@dataclass
class TaskDef:
    id: str
    name: str
    difficulty: str
    max_steps: int
    initial_code: str
    test_code_template: str

TASKS = {}

EASY_INIT = '''def calculate_total(prices, tax_rate):
    total = 0
    for p in prices:
        total += p
    return total * (1 + tax_rate)
'''

# The test script uses a custom runner to output JSON results 
# so we can parse exactly how many passed vs failed.
EASY_TEST = '''
import json
import traceback

def run_tests():
    {code}
    
    tests = [
        lambda: calculate_total([100, 200], 0.1) == 330.0,
        lambda: calculate_total([], 0.2) == 0.0,
        lambda: calculate_total([50], 0) == 50.0,
        lambda: calculate_total([10, 20, 30], 0.05) == 63.0
    ]
    
    passed = 0
    errors = []
    for i, t in enumerate(tests):
        try:
            if t():
                passed += 1
            else:
                errors.append(f"Test {i+1} failed: returned False")
        except Exception as e:
            errors.append(f"Test {i+1} raised {type(e).__name__}: {str(e)}")
            
    print(json.dumps({{"passed": passed, "total": len(tests), "errors": errors}}))

if __name__ == "__main__":
    run_tests()
'''

TASKS['task_easy'] = TaskDef(
    id='task_easy',
    name='Python Syntax Fix',
    difficulty='easy',
    max_steps=10,
    initial_code=EASY_INIT,
    test_code_template=EASY_TEST
)


MEDIUM_INIT = '''def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] < merged[-1][1]: 
            merged[-1][1] = intervals[i][1]
        else:
            merged.append(intervals[i])
    return merged
'''

MEDIUM_TEST = '''
import json
import traceback

def run_tests():
    {code}
    
    tests = [
        lambda: merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]],
        lambda: merge_intervals([[1,4],[4,5]]) == [[1,5]],
        lambda: merge_intervals([[1,4],[2,3]]) == [[1,4]],
        lambda: merge_intervals([[1,4],[0,4]]) == [[0,4]],
        lambda: merge_intervals([[1,4],[2,5],[7,9],[14,15]]) == [[1,5],[7,9],[14,15]]
    ]
    
    passed = 0
    errors = []
    for i, t in enumerate(tests):
        try:
            if t():
                passed += 1
            else:
                errors.append(f"Test {i+1} failed")
        except Exception as e:
            errors.append(f"Test {i+1} failed: {type(e).__name__}")
            
    print(json.dumps({{"passed": passed, "total": len(tests), "errors": errors}}))

if __name__ == "__main__":
    run_tests()
'''

TASKS['task_medium'] = TaskDef(
    id='task_medium',
    name='Python Logic Repair',
    difficulty='medium',
    max_steps=15,
    initial_code=MEDIUM_INIT,
    test_code_template=MEDIUM_TEST
)

HARD_INIT = '''def find_longest_sequence(nums):
    max_len = 0
    for num in nums:
        current_num = num
        current_len = 1
        while current_num + 1 in nums:
            current_num += 1
            current_len += 1
        max_len = max(max_len, current_len)
    return max_len
'''

HARD_TEST = '''
import json
import time

def run_tests():
    {code}
    
    passed = 0
    errors = []
    
    # Correctness tests
    def t1(): return find_longest_sequence([100, 4, 200, 1, 3, 2]) == 4
    def t2(): return find_longest_sequence([0,3,7,2,5,8,4,6,0,1]) == 9
    def t3(): return find_longest_sequence([]) == 0
    def t4(): return find_longest_sequence([1,2,0,1]) == 3
    
    tests = [t1, t2, t3, t4]
    for i, t in enumerate(tests):
        try:
            if t(): passed += 1
            else: errors.append(f"Basic test {i+1} failed")
        except Exception as e:
            errors.append(f"Basic test {i+1} error: {e}")
            
    # Performance test
    try:
        start = time.time()
        large_input = list(range(10000)) + [1000000]
        res = find_longest_sequence(large_input)
        if res != 10000:
            errors.append(f"Performance test logic failed: got {res}")
        else:
            if time.time() - start > 0.5:
                # Timed out but answer was correct -- give partial progress
                errors.append("Algorithm is too slow (O(N^2) or O(N^3)). Need O(N) using sets.")
            else:
                passed += 1
    except Exception as e:
        errors.append(f"Performance test err: {e}")
        
    print(json.dumps({{"passed": passed, "total": 5, "errors": errors}}))

if __name__ == "__main__":
    run_tests()
'''

TASKS['task_hard'] = TaskDef(
    id='task_hard',
    name='Algorithm Optimization & Correctness',
    difficulty='hard',
    max_steps=20,
    initial_code=HARD_INIT,
    test_code_template=HARD_TEST
)
