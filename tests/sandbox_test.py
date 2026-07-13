import pytest
from analyst_agent import run_code
import os
from analyst_agent.core import get_configs

@pytest.fixture
def data_filename():
    return get_configs()['carbon_data'].name


def test_crashing_code():
    result = run_code("raise ValueError('bad column')", timeout=5)
    assert result.success == False
    assert result.timed_out == False
    assert "ValueError: bad column" in result.stderr
    assert result.stdout == ""

def test_timeout_code():
    result = run_code("while True: pass", timeout=1)
    assert result.success == False
    assert result.timed_out == True
    assert "timed out" in result.stderr
    assert result.stdout == ""

def test_environment_stripped():
    os.environ['SECRET_TEST'] = 'should_not_leak'
    try:
        result = run_code("import os; print(dict(os.environ))", timeout=5)
        assert result.success == True
        assert 'SECRET_TEST' not in result.stdout
    finally:
        del os.environ['SECRET_TEST']

def test_cwd_is_isolated_scratch_dir(data_filename):
    result = run_code("import os; print(os.listdir('.'))", timeout=5)
    assert result.success == True
    assert result.stdout.strip() == f"['{data_filename}']"  #  scratch dir only contains parquet file, not my repo's files

def test_successful_code():
    result = run_code("print(2 + 2)", timeout=5)
    assert result.success == True
    assert result.timed_out == False
    assert result.stderr == ""
    assert result.stdout.strip() == "4"




