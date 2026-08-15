from dataclasses import dataclass

@dataclass
class Pair:
    prompt: list[dict[str, str]]
    target: str

def build_pairs(raw_pairs: dict[str, list[dict]]) -> list[Pair]: