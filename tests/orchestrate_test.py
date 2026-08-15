import pytest
from analyst_agent.orchestrate import (
    run_task,
    solve_task,
    Solved,
    ExtractionFail,
    ExecutionFail,
    GenuineMiss
)
from analyst_agent import Output, ExecutionResult, Grader, Task
import pandas as pd
from unittest.mock import Mock

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

def test_solve_task_returns_final_answer_without_tool(task, df, monkeypatch):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = 'print(42.0)'

    agent.extract_code.return_value = Output(
        raw='print(42.0)',
        code='print(42.0)'
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.run_task",
        lambda task, output, no_think, df: Solved(
            task=task,
            grader=Mock(correct=True),
            no_think=no_think,
            raw=output.raw
        )
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, Solved)
    agent.generate.assert_called_once()
    agent.extract_code.assert_called_once_with('print(42.0)')

def test_solve_task_executes_valid_tool_call(
    task, df, monkeypatch
):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = (
        '<tool_call>'
        '{"name": "check_code", "arguments": {"code": "print(42.0)"}}'
        '</tool_call>'
    )

    tool_result = ExecutionResult(
        success=True,
        timed_out=False,
        stderr="",
        stdout="42.0",
        code="print(42.0)"
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.run_code",
        lambda code, timeout=20: tool_result
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.classify",
        lambda result, raw, task, df, no_think:
            Solved(
                task=task,
                grader=Mock(correct=True),
                no_think=no_think,
                raw=raw
            )
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, Solved)

def test_solve_task_retries_after_tool_execution_failure(
    task, df, monkeypatch
):

    agent = Mock()

    agent._build_messages.return_value = []

    responses = [
        (
            '<tool_call>'
            '{"name": "check_code", "arguments": {"code": "print(x)"}}'
            '</tool_call>'
        ),
        'print(42.0)'
    ]

    agent.generate.side_effect = responses

    failed_result = ExecutionResult(
        success=False,
        timed_out=False,
        stderr="NameError: name 'x' is not defined",
        stdout="",
        code="print(x)"
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.run_code",
        lambda code, timeout=20: failed_result
    )

    agent.extract_code.return_value = Output(
        raw='print(42.0)',
        code='print(42.0)'
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.run_task",
        lambda task, output, no_think, df:
            Solved(
                task=task,
                grader=Mock(correct=True),
                no_think=no_think,
                raw=output.raw
            )
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, Solved)

    assert agent.generate.call_count == 2

    # The error should have been added to the conversation
    second_call_messages = agent.generate.call_args_list[1].kwargs["messages"]

    assert any(
        message["role"] == "tool"
        and "NameError" in message["content"]
        for message in second_call_messages
    )

def test_solve_task_malformed_tool_call_returns_extraction_fail(
    task, df
):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = (
        '<tool_call>'
        '{"name": "check_code", "arguments": '
        '</tool_call>'
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, ExtractionFail)
    assert result.raw == agent.generate.return_value
    assert result.task is task

def test_solve_task_missing_tool_name_returns_extraction_fail(
    task, df
):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = (
        '<tool_call>'
        '{"arguments": {"code": "print(42.0)"}}'
        '</tool_call>'
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, ExtractionFail)
    assert result.raw == agent.generate.return_value

def test_solve_task_unknown_tool_returns_extraction_fail(
    task, df
):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = (
        '<tool_call>'
        '{"name": "some_fake_tool", '
        '"arguments": {"code": "print(42.0)"}}'
        '</tool_call>'
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, ExtractionFail)
    assert result.raw == agent.generate.return_value

def test_solve_task_missing_code_returns_extraction_fail(
    task, df
):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = (
        '<tool_call>'
        '{"name": "check_code", "arguments": {}}'
        '</tool_call>'
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, ExtractionFail)
    assert result.raw == agent.generate.return_value

def test_solve_task_respects_max_turns(
    task, df, monkeypatch
):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = (
        '<tool_call>'
        '{"name": "check_code", "arguments": {"code": "print(x)"}}'
        '</tool_call>'
    )

    failed_result = ExecutionResult(
        success=False,
        timed_out=False,
        stderr="NameError: name 'x' is not defined",
        stdout="",
        code="print(x)"
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.run_code",
        lambda code, timeout=20: failed_result
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.classify",
        lambda result, raw, task, df, no_think: ExecutionFail(
            task=task,
            no_think=no_think,
            result=result,
            raw=raw
        )
    )

    result = solve_task(
        agent,
        task,
        df,
        temp=0.7,
        no_think=True,
        max_turns=3
    )

    assert agent.generate.call_count == 3
    assert isinstance(result, ExecutionFail)

def test_solve_task_successful_tool_with_wrong_answer_returns_genuine_miss(
    task, df, monkeypatch
):

    agent = Mock()

    agent._build_messages.return_value = []

    agent.generate.return_value = (
        '<tool_call>'
        '{"name": "check_code", "arguments": {"code": "print(99.0)"}}'
        '</tool_call>'
    )

    tool_result = ExecutionResult(
        success=True,
        timed_out=False,
        stderr="",
        stdout="99.0",
        code="print(99.0)"
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.run_code",
        lambda code, timeout=20: tool_result
    )

    monkeypatch.setattr(
        "analyst_agent.orchestrate.classify",
        lambda result, raw, task, df, no_think:
            GenuineMiss(
                task=task,
                grader=Mock(
                    correct=False,
                    obtained_answer="99.0",
                    expected_answer=42.0
                ),
                no_think=no_think,
                raw=raw
            )
    )

    result = solve_task(agent, task, df, temp=0.7)

    assert isinstance(result, GenuineMiss)

    # A successful execution should terminate the loop.
    assert agent.generate.call_count == 1
