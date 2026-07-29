from analyst_agent.core import get_configs
from analyst_agent import Agent, tasks, run_task, Solved, ExecutionFail, ExtractionFail, GenuineMiss
import pandas as pd
import logging

logger = logging.getLogger(__name__)


agent = Agent("Qwen/Qwen3-1.7B")
temp = 0.7
k = 8
no_think = True
buckets = {
    'solved': [],
    'extraction_fail': [],
    'execution_fail': [],
    'genuine_miss': []
}
results = {}
df = pd.read_parquet(get_configs()['carbon_data'])
for task in tasks:
    logger.info(f"Testing task{task.id}")
    results[task.id] = []
    
    for _ in range(k):
        output = agent.answer(task.question, no_think=no_think, temp=temp)
        results[task.id].append(run_task(task, output, no_think, df))

solve_rates = []
for task in tasks:
    outcomes = results[task.id]
    counts = {'solved': 0, 'execution_fail': 0, 'extraction_fail': 0, 'genuine_miss': 0}
    exec_fails = []
    extract_fails = []
    misses = []
    for o in outcomes:
        match o:
            case Solved():
                counts['solved'] += 1
            case ExecutionFail(): 
                counts['execution_fail'] += 1
                exec_fails.append(o.result.code)
            case ExtractionFail(): 
                counts['extraction_fail'] += 1
                extract_fails.append(o.raw)
            case GenuineMiss():   
                counts['genuine_miss'] += 1
                misses.append(o.grader.obtained_answer)

    solve_rates.append(counts['solved']/len(results[task.id]))
    print(f'Solve Rate: {solve_rates[-1]}')
    print(f'Exec fail Rate: {counts["execution_fail"]/len(results[task.id])}')
    if len(exec_fails) > 0:
        for fails in exec_fails:
            print(fails)
    print(f'Extract fail Rate: {counts["extraction_fail"]/len(results[task.id])}')
    if len(extract_fails) > 0:
        for fails in extract_fails:
            print(fails)
    print(f'Miss Rate: {counts["genuine_miss"]/len(results[task.id])}')
    if len(misses) > 0:
        for miss in misses:
            print(miss)

    print('-'*9)

print(f'Total solve rate: {sum(solve_rates)/len(solve_rates)}')