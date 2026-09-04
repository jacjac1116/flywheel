from analyst_agent.core import get_configs
from analyst_agent import Agent, tasks, Solved, ExecutionFail, ExtractionFail, GenuineMiss, solve_task, train, run_task, Task, test
import pandas as pd
import logging
import json
from datetime import datetime


logger = logging.getLogger(__name__)


def print_bar(rate: float, width: int = 10) -> str:
    """Returns a unicode bar for a percentage."""

    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled)


def get_counts(outcomes):

    counts = {
        "solved": 0,
        "execution_fail": 0,
        "extraction_fail": 0,
        "genuine_miss": 0
    }

    for outcome in outcomes:

        match outcome:

            case Solved():
                counts["solved"] += 1

            case ExecutionFail():
                counts["execution_fail"] += 1

            case ExtractionFail():
                counts["extraction_fail"] += 1

            case GenuineMiss():
                counts["genuine_miss"] += 1

    return counts


def save_report(results: dict, path: str):

    report = {}

    for task_id, outcomes in results.items():

        report[task_id] = []

        for outcome in outcomes:

            if isinstance(outcome, Solved):

                report[task_id].append({
                    "status": "solved",
                    "question": outcome.task.question,
                    "tier": outcome.task.tier,
                    "expected": outcome.grader.expected_answer,
                    "obtained": outcome.grader.obtained_answer,
                    "code": outcome.grader.generated_script,
                    "raw": outcome.raw,
                    "no_think": outcome.no_think
                })


            elif isinstance(outcome, ExecutionFail):

                report[task_id].append({
                    "status": "execution_fail",
                    "question": outcome.task.question,
                    "tier": outcome.task.tier,
                    "stderr": outcome.result.stderr,
                    "code": outcome.result.code,
                    "raw": outcome.raw,
                    "no_think": outcome.no_think
                })


            elif isinstance(outcome, ExtractionFail):

                report[task_id].append({
                    "status": "extraction_fail",
                    "question": outcome.task.question,
                    "tier": outcome.task.tier,
                    "raw": outcome.raw,
                    "no_think": outcome.no_think
                })


            elif isinstance(outcome, GenuineMiss):

                report[task_id].append({
                    "status": "genuine_miss",
                    "question": outcome.task.question,
                    "tier": outcome.task.tier,
                    "expected": outcome.grader.expected_answer,
                    "obtained": outcome.grader.obtained_answer,
                    "code": outcome.grader.generated_script,
                    "raw": outcome.raw,
                    "no_think": outcome.no_think
                })


    with open(path, "w") as f:
        json.dump(report, f, indent=4, default=str)



def run_round(
        k: int,
        temp: float,
        task_list: list[Task],
        tool_use: bool,
        no_think: bool = True,
        model: str = "Qwen/Qwen3-1.7B",
        adapter_path : str| None = None,
        save: bool = True
):

    """
    Run one complete evaluation round.

    A round consists of:
    - Loading the model agent
    - Running every benchmark task k times
    - Collecting outcomes
    - Calculating aggregate metrics
    - Optionally saving a detailed JSON report

    Returns:
        results: Raw task outcomes
        overall: Aggregate counts across all evaluations
        solve_rate: Overall success percentage
    """

    agent = Agent(model, adapter_path)

    df = pd.read_parquet(get_configs()['carbon_data'] )

    path = (
            "reports/"
            f"evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )


    results = {}


    # Run every task multiple times to estimate reliability
    for task in task_list:

        logger.info(f"Testing task {task.id}")

        results[task.id] = []

        for _ in range(k):

            if tool_use == True:
                outcome = solve_task(agent,task,df,temp,no_think)
            else:
                output = agent.answer(task.question, no_think, temp=temp)
                outcome = run_task(task, output, no_think, df)

            results[task.id].append(outcome)

            save_report(results, path)



    # Aggregate results across all tasks
    overall = {
        "solved": 0,
        "execution_fail": 0,
        "extraction_fail": 0,
        "genuine_miss": 0
    }


    total_runs = sum(len(v) for v in results.values())


    for outcomes in results.values():

        counts = get_counts(outcomes)

        for key in overall:

            overall[key] += counts[key]



    solve_rate = overall["solved"] / total_runs



    # Print concise summary
    print("\n")
    print("=" * 80)
    print("OVERALL")
    print("=" * 80)

    print(f"{'Tasks':25}{len(task_list)}")
    print(f"{'Runs per task':25}{k}")
    print(f"{'Total evaluations':25}{total_runs}")

    print()

    for name, key in [
        ("Solve rate", "solved"),
        ("Execution failure rate", "execution_fail"),
        ("Extraction failure rate", "extraction_fail"),
        ("Genuine miss rate", "genuine_miss")
    ]:

        print(
            f"{name:25}{overall[key] / total_runs:.1%}"
        )


    print("\n")
    print("=" * 80)
    print("TASK RESULTS")
    print("=" * 80)


    for task in task_list:

        counts = get_counts(results[task.id])

        rate = counts["solved"] / len(results[task.id])

        print()
        print(f"Task {task.id} (Tier {task.tier})")

        print(f"Solve rate: {print_bar(rate)} {rate:.1%}")

        print()
    print(f"Report saved: {path}")


    return results, overall, solve_rate



if __name__ == "__main__":

    for i in range(1):

        results, overall, rate = run_round(k=32, temp=0.7, task_list=test, tool_use=False, adapter_path='adapters/round1')

        print(f"Run {i}: {rate:.1%}")