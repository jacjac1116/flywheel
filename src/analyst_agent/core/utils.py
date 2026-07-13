from pyprojroot import here
import os
from pathlib import Path

def get_configs():

    carbon_data = os.environ.get('CARBON_DATA')
    # Python evaluates all arguments before calling the function. So here() / 'data' / ... runs on every call to get_configs(), even in CI where the result is thrown away immediately.
    # a default argument is not lazy. d.get(k, expensive()) always runs expensive(). 
    # best to do in if loop
    if carbon_data is None:
        carbon_data = here() / 'data' / 'raw' / 'carbon_2020-01-01_2025-12-31.parquet'

    return {
        'carbon_data': Path(carbon_data) # os.environ.get returns a str in CI; the fallback is already a Path locally. Wrapping unconditionally normalises both to Path, so .name is available downstream regardless of which branch fired
    }

if __name__=='__main__':
    configs = get_configs()
    print(configs['carbon_data'].name)
    