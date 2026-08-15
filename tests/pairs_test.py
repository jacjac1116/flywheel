import pytest
from analyst_agent.sft import Pair, build_pairs, code_formatter
from analyst_agent.agent import Agent

def test_build_pairs_returns_valid_pairs():
    raw_pairs = {
        "test": [
            {
                "status": "solved",
                "question": "What is the average value?",
                "no_think": True,
                "code": "print(df.mean())",
            },
            {
                "status": "failed",
                "question": "This should not be included.",
                "no_think": True,
                "code": "print('failed')",
            },
        ]
    }

    pairs = build_pairs(raw_pairs)

    # Only solved examples should be converted into training pairs.
    assert len(pairs) == 1

    # Every generated target should contain actual Python code.
    assert all(Agent.extract_code(pair.target).code.strip() for pair in pairs)

    # Tool calls should never be present in the final training target.
    assert all("<tool_call>" not in pair.target for pair in pairs)

def test_code_formatter_formats_python_code():
    code = "print('hello')"

    result = code_formatter(code)

    assert result == (
        "<think>\n\n</think>\n\n"
        "```python\n"
        "print('hello')\n"
        "```"
    )

def test_code_formatter_preserves_multiline_code():
    code = "df = pd.read_csv('data.csv')\nprint(df.head())"

    result = code_formatter(code)

    assert result == (
        "<think>\n\n</think>\n\n"
        "```python\n"
        "df = pd.read_csv('data.csv')\n"
        "print(df.head())\n"
        "```"
    )