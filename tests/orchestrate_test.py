import pytest
from analyst_agent.orchestrate import (
    run_task,
    Solved,
    ExtractionFail,
    ExecutionFail,
    GenuineMiss
)
from analyst_agent import Output, ExecutionResult, Grader, Task
import pandas as pd

@pytest.fixture
def df():
    return pd.DataFrame()

@pytest.fixture
def task():
    return Task(
        id=1,
        tier=1,
        question='Dummy question',
        answer=lambda df: 42
    )


def test_run_task_extraction_fail(task, df):

    output = Output(
        raw='model output',
        code=None
    )

    result = run_task(task, output, True, df)

    assert isinstance(result, ExtractionFail)
    assert result.raw == 'model output'
    assert result.task is task
    assert result.no_think is True

def test_run_task_execution_fail(task, df, monkeypatch):

    output = Output(
        raw='model output',
        code='print(hello)'
    )

    execution = ExecutionResult(
        success=False,
        stdout="",
        stderr="SyntaxError",
        timed_out=False,
        code='print(hello)'
    )

    monkeypatch.setattr('analyst_agent.orchestrate.run_code',
                        lambda code, timeout: execution)


    result = run_task(task, output, True, df)

    assert isinstance(result, ExecutionFail)
    assert result.result is execution
    assert result.raw == "model output"


def test_run_task_solved(task, df, monkeypatch):

    execution = ExecutionResult(
        success=True,
        stdout="42",
        stderr="",
        timed_out=False,
        code="print(42)"
    )

    grading = Grader(
        correct=True,
        expected_answer=42,
        obtained_answer="42",
        generated_script="print(42)"
    )

    monkeypatch.setattr('analyst_agent.orchestrate.run_code',
                        lambda *args, **kwargs: execution)
    
    monkeypatch.setattr('analyst_agent.orchestrate.grade',
                        lambda *args, **kwargs: grading)
    
    output = Output(
        raw="raw",
        code="print(42)"
    )
    
    result = run_task(task, output, True, df)

    assert isinstance(result, Solved)
    assert result.grader is grading
    assert result.task is task

def test_run_task_genuine_miss(task, df, monkeypatch):

    execution = ExecutionResult(
        success=True,
        stdout="41",
        stderr="",
        timed_out=False,
        code="print(41)"
    )

    grading = Grader(
        correct=False,
        expected_answer=42,
        obtained_answer="41",
        generated_script="print(41)"
    )

    monkeypatch.setattr('analyst_agent.orchestrate.run_code',
                        lambda *args, **kwargs: execution)
    
    monkeypatch.setattr('analyst_agent.orchestrate.grade',
                        lambda *args, **kwargs: grading)
    
    output = Output(
        raw="raw",
        code="print(41)"
    )
    
    result = run_task(task, output, True, df)

    assert isinstance(result, GenuineMiss)
    assert result.grader is grading

