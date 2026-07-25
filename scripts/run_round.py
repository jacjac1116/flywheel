from analyst_agent.core import get_configs
from analyst_agent import Agent, tasks, run_task, Solved, ExecutionFail, ExtractionFail, GenuineMiss
import pandas as pd
import logging

logger = logging.getLogger(__name__)


agent = Agent("Qwen/Qwen3-1.7B")
no_think = True
buckets = {
    'solved': [],
    'extraction_fail': [],
    'execution_fail': [],
    'genuine_miss': []
}
df = pd.read_parquet(get_configs()['carbon_data'])
for task in tasks:
    logger.info(f"Testing task{task.id}")
    output = agent.answer(task.question, no_think)

    outcome = run_task(task, output, no_think, df)

    match outcome:
        case Solved():
            buckets['solved'].append(outcome)
        case ExtractionFail():
            buckets['extraction_fail'].append(outcome)
        case ExecutionFail():
            buckets['execution_fail'].append(outcome)
        case GenuineMiss():
            buckets['genuine_miss'].append(outcome)

for k, v in buckets.items():
    if len(buckets[k]) > 0:
        if k == 'solved':
            print(f"{len(buckets[k])} solved tasks\n")
            for i in v:
                print(f"Question: {i.task.question}")
                print(f"Raw answer: {i.raw}")
                print('#'*9)

        elif k == 'extraction_fail':
            print(f"{len(buckets[k])} tasks failed at extraction\n")
            for i in v:
                print(f"Question: {i.task.question}")
                print(f"Expected answer: {i.task.answer(df)}")
                print(f"Raw answer: {i.raw}")
                print('#'*9)

        elif k == 'execution_fail':
            print(f"{len(buckets[k])} tasks failed at execution\n")
            for i in v:
                print(f"Question: {i.task.question}")
                print(f"Expected answer: {i.task.answer(df)}")
                print(f"Generated code: {i.result.code}")
                print(f"Error: {i.result.stderr}")
                print(f"Raw answer: {i.raw}")
                print('#'*9)
        else:
            print(f"{len(buckets[k])} tasks failed at due to wrong answer\n")
            for i in v:
                print(f"Question: {i.task.question}")
                print(f"Expected answer: {i.task.expected_answer}")
                print(f"Obtained answer: {i.grader.obtained_answer}")
                print(f"Generated code: {i.grader.generated_script}")
                print(f"Raw answer: {i.raw}")
                print('#'*9)
