import pytest
from analyst_agent import Agent

@pytest.fixture
def agent():
    a = Agent.__new__(Agent)
    a.user_prompt = "File: {data_file}\nQ: {question}"
    a.system_prompt = "You write pandas."
    a.tools_prompt = "No tools"
    a.configs = {'carbon_data': type('P', (), {'name': 'carbon_sample.parquet'})()}
    return a


def test_no_think_suffix_appended(agent):
    msgs = agent._build_messages('What is the mean?', no_think=True)
    assert msgs[1]['content'].endswith('/no_think')

def test_no_think_false_no_suffix(agent):
    msgs = agent._build_messages('What is the mean?', no_think=False)
    assert not msgs[1]['content'].endswith('/no_think')

def test_message_structure(agent):
    msgs = agent._build_messages('What is the mean?', no_think=True)
    assert len(msgs) == 2
    assert msgs[0]['role'] == 'system'
    assert msgs[1]['role'] == 'user'

def test_extract_fenced_python():
    output = """
Some text

```python
print("hello")
```"""
    result = Agent.extract_code(output)
    assert result.code == 'print("hello")'
    assert result.raw == output

def test_extract_without_markdown():

    output = """
print("hello")
"""
    result = Agent.extract_code(output)

    assert result.code == 'print("hello")'

def test_extract_removes_thinking():

    output = """
<think>
I should use pandas.
</think>


print(42)

"""
    result = Agent.extract_code(output)

    assert result.code == "print(42)"

def test_extract_empty_string():

    result = Agent.extract_code("")

    assert result.code is None

def test_extract_empty_code_block():

    output = """
```python
"""
    result = Agent.extract_code(output)

    assert result.code is None

def test_extract_code_without_print():

    output = """
```python
x = 1 + 2
"""
    result = Agent.extract_code(output)

    assert result.code is None

def test_extract_no_code_preserves_raw():

    output = """
The answer is 42.
"""

    result = Agent.extract_code(output)

    assert result.code is None
    assert result.raw == output

def test_extract_multiline_script():

    output = """
```python
import pandas as pd

df = pd.read_parquet("carbon.parquet")

mean_forecast = df["forecast"].mean()

print(mean_forecast)
```
"""
    result = Agent.extract_code(output)

    assert result.code.endswith("print(mean_forecast)")
    assert "import pandas as pd" in result.code
    assert 'pd.read_parquet("carbon.parquet")' in result.code
    assert result.raw == output

def test_extract_stops_at_first_fenced_block():

    output = """
```python
x = 42
print(x)
``` output
42
```
"""
    result = Agent.extract_code(output)

    # The regex intentionally uses a lazy (.*?) match so extraction
    # stops at the first closing fence rather than consuming later blocks.
    assert result.code == "x = 42\nprint(x)"
    assert result.raw == output