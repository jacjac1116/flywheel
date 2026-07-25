from analyst_agent import run_code, grade, Output, Task, Grader, ExecutionResult, Agent, tasks
from dataclasses import dataclass
from typing import Any
import logging
import pandas as pd
from analyst_agent.core import get_configs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Solved:
    task: Task
    grader: Grader
    no_think: bool
    raw: str

@dataclass
class ExtractionFail:
    no_think: bool
    raw: str
    task: Task

@dataclass
class ExecutionFail:
    task: Task
    no_think: bool
    result: ExecutionResult
    raw: str

@dataclass
class GenuineMiss:
    task: Task
    grader: Grader
    no_think: bool
    raw: str

Outcome = Solved | ExtractionFail | ExecutionFail | GenuineMiss


def run_task(task: Task, output: Output, no_think: bool, df: pd.DataFrame) -> Outcome:
    if output.code is None:
        logger.info('No script found, model attempt will be assigned to ExtractionFail...')
        return ExtractionFail(
            no_think=no_think,
            raw=output.raw,
            task=task
        )
    else:
        exec_result = run_code(output.code, timeout=20)
        if exec_result.success == False:
            logger.info('Script did not execute, model attempt will be assigned to ExecutionFail...')
            return ExecutionFail(
                task=task,
                no_think=no_think,
                result=exec_result,
                raw=output.raw
            )
        else:
            score = grade(exec_result, task, df)
            if score.correct == True:
                logger.info('Model attempt was a success!')
                return Solved(
                    task=task,
                    grader=score,
                    no_think=no_think,
                    raw=output.raw
                )
            else:
                logger.info(f'Script answered {score.obtained_answer} but was expecting {score.expected_answer}')
                return GenuineMiss(
                    task=task,
                    grader=score,
                    no_think=no_think,
                    raw=output.raw
                )