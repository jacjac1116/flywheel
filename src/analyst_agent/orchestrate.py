from analyst_agent.sandbox import run_code, ExecutionResult
from analyst_agent.agent import Agent, Output
from analyst_agent.tasks import grade, Task, Grader
from analyst_agent.tools_schema import TOOLS
from dataclasses import dataclass
from typing import Any
import logging
import pandas as pd
from analyst_agent.core import get_configs
import json

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
    """
    Execute and grade a script extracted from a model response.

    This path is used when the model returns Python code directly without
    invoking the execution tool.

    Outcome flow:
        1. No code extracted           -> ExtractionFail
        2. Script fails to execute     -> ExecutionFail
        3. Script executes correctly:
            - Correct answer           -> Solved
            - Incorrect answer         -> GenuineMiss
    """

    # No executable Python script could be extracted from the model output
    if output.code is None:
        logger.info('No script found, model attempt will be assigned to ExtractionFail...')
        return ExtractionFail(
            no_think=no_think,
            raw=output.raw,
            task=task
        )

    else:

        # Execute the extracted script inside the sandbox
        exec_result = run_code(output.code, timeout=20)

        # Runtime or syntax error
        if exec_result.success == False:
            logger.info('Script did not execute, model attempt will be assigned to ExecutionFail...')
            return ExecutionFail(
                task=task,
                no_think=no_think,
                result=exec_result,
                raw=output.raw
            )

        else:

            # Compare the script's output against the ground truth
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


def classify(result: ExecutionResult, raw: str, task: Task, df: pd.DataFrame, no_think: bool) -> Outcome:
    """
    Classify the outcome of an already executed script.

    Unlike `run_task`, this function assumes the script has already been
    executed (typically via the execution tool) and therefore only needs to
    determine whether execution failed, the answer was correct, or the answer
    was incorrect.
    """

    # Script failed to execute
    if result.success == False:
        logger.info('Script did not execute, model attempt will be assigned to ExecutionFail...')
        return ExecutionFail(
            task=task,
            no_think=no_think,
            result=result,
            raw=raw
        )

    else:

        # Grade the successfully executed script
        score = grade(result, task, df)

        if score.correct == True:
            logger.info('Model attempt was a success!')
            return Solved(
                task=task,
                grader=score,
                no_think=no_think,
                raw=raw
            )

        else:
            logger.info(f'Script answered {score.obtained_answer} but was expecting {score.expected_answer}')
            return GenuineMiss(
                task=task,
                grader=score,
                no_think=no_think,
                raw=raw
            )


def solve_task(agent: Agent, task: Task, df: pd.DataFrame, temp: float, no_think: bool = True, max_turns: int = 3) -> Outcome:
    """
    Attempt to solve a benchmark task using the language model.

    Logic:

        1. Build the initial conversation containing the task.
        2. Ask the model to produce either:
            - a final Python script, or
            - a tool call requesting script execution.
        3. If a tool is requested:
            - execute the requested tool,
            - append the tool result back into the conversation,
            - allow the model to revise its script.
        4. Repeat until:
            - the model returns a final script,
            - a valid script executes successfully,
            - the maximum number of tool iterations is reached.
        5. Classify the final outcome as one of:
            Solved, ExtractionFail, ExecutionFail or GenuineMiss.
    """

    # Initial conversation containing the task prompt
    messages = agent._build_messages(task.question, no_think, TOOLS)

    # Tools available to the language model
    available_functions = {
        'check_code': run_code,
    }

    # Stores the most recent execution result returned by the tool
    tool_result = None

    # Allow the model multiple attempts to repair its code
    for n in range(max_turns):

        # Generate the model's next response
        output = agent.generate(no_think, temp=temp, tools=TOOLS, messages=messages)

        # No tool requested — assume the model has produced its final script
        if '<tool_call>' not in output:
            return run_task(task, agent.extract_code(output), no_think, df)

        # Attempt to parse the tool call emitted by the model
        try:
            tool_call = output.split('<tool_call>')[1].strip().split('</tool_call>')[0].strip()

            call_schemas = json.loads(tool_call)
            function_name = call_schemas['name']
            arguments = call_schemas['arguments']

        # Invalid tool call format
        except (json.JSONDecodeError, IndexError, KeyError):
            return ExtractionFail(
                no_think=no_think,
                raw=output,
                task=task
            )

        # Execute the requested tool if it exists
        if function_name in available_functions:

            logger.info(f'Tool use {n+1}')
            try:
                tool_result = available_functions[function_name](arguments['code'])
            
            except KeyError:
                return ExtractionFail(
                    no_think=no_think,
                    raw=output,
                    task=task
                )
                    

            # Preserve the assistant response in the conversation history
            messages.append({
                'role': 'assistant',
                'content': output
            })

            # Successful execution — no further repair required
            if tool_result.success == True:
                return classify(tool_result, output, task, df, no_think)

            # Execution failed — feed stderr back to the model so it can repair the script
            else:
                messages.append({
                    'role': 'tool',
                    'content': f'Script failed stderr = {tool_result.stderr}'
                })

        # Model requested an unknown tool
        else:
            return ExtractionFail(
                no_think=no_think,
                raw=output,
                task=task
            )

    # Model never invoked a tool successfully
    if tool_result is None:
        return ExtractionFail(
            no_think=no_think,
            raw=output,
            task=task
        )

    # Maximum repair attempts reached — classify the final execution result
    return classify(tool_result, output, task, df, no_think)