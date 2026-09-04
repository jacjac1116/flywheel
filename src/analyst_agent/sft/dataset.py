from analyst_agent.sft import Pair
import random
import json
import logging
from pyprojroot import here
from pathlib import Path

logger = logging.getLogger()

def shuffler(pairs: list[Pair]) -> list[Pair]:
    
    random.seed(4)

    return random.sample(pairs, len(pairs))

def splitter(pairs: list[Pair], train_ratio: float) -> tuple[list[Pair], list[Pair]]:
   
    cut = int(len(pairs) * train_ratio)
          
    return(pairs[:cut], pairs[cut:])

def jsonl_writer(pairs: list[Pair], path: Path) -> None:

    with open(path, "w") as f:
        for pair in pairs:
            obj = {'messages': pair.prompt + [{'role': 'assistant', 'content': pair.target}]}
            f.write(json.dumps(obj) + '\n')
    logger.info(f'{path} completed')


def write_dataset(pairs: list[Pair], train_ratio: float = 0.9) -> None:

    out_dir = here() / 'data' / 'jsonl'
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / 'train.jsonl'
    valid_path = out_dir / 'valid.jsonl'

    shuffled_pairs = shuffler(pairs)
    train, valid = splitter(shuffled_pairs, train_ratio)
    jsonl_writer(train, train_path)
    jsonl_writer(valid, valid_path)


if __name__ == '__main__':

    from analyst_agent.sft.pairs import build_pairs

    print(here())

    with open(here() / 'reports' / 'evaluation_2026-08-12_19-30-31.json', 'r') as f:
        raw_pairs = json.load(f)
    
    cleaned_pairs = build_pairs(raw_pairs)

    write_dataset(cleaned_pairs)

    with open(here() / 'data' / 'jsonl' / 'train.jsonl') as file:
        first = file.readline()

    obj = json.loads(first)
    print(len(obj['messages']), 'messages')
    print(obj)