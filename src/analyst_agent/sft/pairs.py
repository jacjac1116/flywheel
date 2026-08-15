from dataclasses import dataclass

import yaml
from pyprojroot import here

from analyst_agent.agent import build_messages
from analyst_agent.core import get_configs


@dataclass
class Pair:
    """
    A prompt/target pair used for training or evaluation.

    Attributes:
        prompt: The chat messages passed to the language model.
        target: The expected model response.
    """

    prompt: list[dict[str, str]]
    target: str


def code_formatter(code: str) -> str:
    """
    Format generated code as a model response.

    The response is wrapped in an empty <think> block followed by a
    Python code block, matching the expected output format of the model.

    Args:
        code: Python source code to format.

    Returns:
        The code wrapped in the expected response format.
    """
    formatted_code = f"<think>\n\n</think>\n\n```python\n{code}\n```"

    return formatted_code


def build_pairs(raw_pairs: dict[str, list[dict]]) -> list[Pair]:
    """
    Convert raw question/code pairs into formatted prompt/target pairs.

    Only pairs with a status of ``"solved"`` are included. The function
    loads the prompts and project configuration required to construct
    the model input, then formats each solution as the expected target.

    Args:
        raw_pairs: Mapping of categories or groups to lists of metric
            dictionaries. Each metric dictionary is expected to contain
            ``status``, ``question``, ``no_think``, and ``code`` fields.

    Returns:
        A list of ``Pair`` objects containing the model prompts and
        corresponding expected code responses.
    """

    # Load the prompts used to construct the model's system and user messages.
    with open(here() / "configs" / "prompts.yaml") as f:
        prompts = yaml.safe_load(f)

    user_prompt = prompts["sandbox_user_prompt"]
    system_prompt = prompts["sandbox_system_prompt"]
    tools_prompt = prompts["tools_prompt"]

    # Load the project configuration to determine which data file
    # should be referenced in the generated user prompt.
    configs = get_configs()

    cleaned_pairs = []

    # Iterate through each group of generated metrics and keep only
    # solutions that were successfully solved.
    for _, metrics_list in raw_pairs.items():
        for metrics in metrics_list:
            if metrics["status"] != "solved":
                continue

            cleaned_pairs.append(
                Pair(
                    prompt=build_messages(
                        question=metrics["question"],
                        no_think=metrics["no_think"],
                        user_prompt=user_prompt,
                        data_file=configs["carbon_data"].name,
                        system_prompt=system_prompt,
                        tools_prompt=tools_prompt,
                        tools=None,
                    ),
                    target=code_formatter(metrics["code"]),
                )
            )

    return cleaned_pairs