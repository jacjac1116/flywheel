import pytest
import pandas as pd
import numpy as np
from analyst_agent.core import get_configs
from analyst_agent.tasks import Task, grade
from analyst_agent.sandbox import ExecutionResult

@pytest.fixture
def df():
    configs = get_configs()
    path = configs['carbon_data']
    return pd.read_parquet(path)


@pytest.fixture
def float_task():
    return Task(
        id=1,
        tier=1,
        question="Dummy float task",
        answer=lambda df: 42.5
    )


@pytest.fixture
def string_task():
    return Task(
        id=2,
        tier=1,
        question="Dummy string task",
        answer=lambda df: "moderate"
    )

@pytest.fixture
def int_task():
    return Task(
        id=3,
        tier=1,
        question="Dummy int task",
        answer=lambda df: np.int64(9)
    )

@pytest.fixture
def timestamp_task():

    return Task(
        id=1,
        tier=4,
        question="Dummy timestamp task",
        answer=lambda df: pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    )

def test_grade_correct_timestamp_aware(df, timestamp_task):

    result = ExecutionResult(
        success=True,
        stdout="2020-01-01 00:00:00+00:00",
        stderr="",
        timed_out=False,
        code='print("2020-01-01 00:00:00+00:00")'
    )

    graded = grade(result, timestamp_task, df)

    assert graded.correct

def test_grade_correct_timestamp_naive(df, timestamp_task):

    result = ExecutionResult(
        success=True,
        stdout="2020-01-01 00:00:00",
        stderr="",
        timed_out=False,
        code='print("2020-01-01 00:00:00")'
    )

    graded = grade(result, timestamp_task, df)

    assert graded.correct

def test_grade_wrong_timestamp(df, timestamp_task):

    result = ExecutionResult(
        success=True,
        stdout="2020-06-15 12:00:00+00:00",
        stderr="",
        timed_out=False,
        code='print("2020-06-15 12:00:00+00:00")'
    )

    graded = grade(result, timestamp_task, df)

    assert not graded.correct

def test_grade_invalid_timestamp(df, timestamp_task):

    result = ExecutionResult(
        success=True,
        stdout="hello",
        stderr="",
        timed_out=False,
        code='print("hello")'
    )

    graded = grade(result, timestamp_task, df)

    assert not graded.correct

def test_grade_correct_float(df, float_task):

    result = ExecutionResult(
        success=True,
        stdout="42.5",
        stderr="",
        timed_out=False,
        code="print(42.5)"
    )

    graded = grade(result, float_task, df)

    assert graded.correct

def test_grade_incorrect_float(df, float_task):

    result = ExecutionResult(
        success=True,
        stdout="123",
        stderr="",
        timed_out=False,
        code="print(123)"
    )

    graded = grade(result, float_task, df)

    assert not graded.correct

def test_grade_invalid_numeric_output(df, float_task):

    result = ExecutionResult(
        success=True,
        stdout="hello",
        stderr="",
        timed_out=False,
        code='print("hello")'
    )

    graded = grade(result, float_task, df)

    assert not graded.correct

def test_grade_correct_string(df, string_task):

    result = ExecutionResult(
        success=True,
        stdout="moderate",
        stderr="",
        timed_out=False,
        code='print("moderate")'
    )

    graded = grade(result, string_task, df)

    assert graded.correct

def test_grade_incorrect_string(df, string_task):

    result = ExecutionResult(
        success=True,
        stdout="high",
        stderr="",
        timed_out=False,
        code='print("high")'
    )

    graded = grade(result, string_task, df)

    assert not graded.correct

def test_grade_correct_int(df, int_task):

    result = ExecutionResult(
        success=True,
        stdout="9.0",
        stderr="",
        timed_out=False,
        code="print(9.0)"
    )

    graded = grade(result, int_task, df)

    assert graded.correct

def test_grade_incorrect_int(df, int_task):
    result = ExecutionResult(
        success=True,
        stdout="9.5",
        stderr="",
        timed_out=False,
        code="print(9.5)"
    )

    graded = grade(result, int_task, df)

    assert not graded.correct


def test_grade_runtime_error(df, float_task):

    result = ExecutionResult(
        success=False,
        stdout="",
        stderr="NameError",
        timed_out=False,
        code="print(x)"
    )

    graded = grade(result, float_task, df)

    assert not graded.correct
    assert graded.obtained_answer is None
    assert graded.expected_answer == 42.5

def test_grade_timeout(df, float_task):

    result = ExecutionResult(
        success=False,
        stdout="",
        stderr="Timed out",
        timed_out=True,
        code="while True: pass"
    )

    graded = grade(result, float_task, df)

    assert not graded.correct
    assert graded.obtained_answer is None
    assert graded.expected_answer == 42.5 

def test_grade_strips_whitespace(df, float_task):

    result = ExecutionResult(
        success=True,
        stdout="\n 42.5 \n",
        stderr="",
        timed_out=False,
        code="print(42.5)"
    )

    graded = grade(result, float_task, df)

    assert graded.correct