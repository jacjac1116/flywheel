from pyprojroot import here
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Any
from analyst_agent.sandbox import ExecutionResult, run_code
import numpy as np
import math


@dataclass
class Task:
    id: int
    tier: int
    question: str
    answer: Callable[[pd.DataFrame], Any]

@dataclass
class Grader:
    correct: bool
    expected_answer: Any
    obtained_answer: Any
    generated_script: str
    


task1 = Task(
    id=1,
    tier=1,
    question='What is the forecast carbon intensity for the settlement period starting at 2025-01-15 13:30:00?',
    answer= lambda df: df[df["from"] == "2025-01-15 13:30:00"]['forecast'].iloc[0]
)

task2 = Task(
    id=2,
    tier=2,
    question='What is the average forecast carbon intensity during periods classified as "high"?',
    answer= lambda df: df[df["index"] == "high"]["forecast"].mean()
    )

tasks = [task1, task2]

def grade(generated_result: ExecutionResult, task: Task, df: pd.DataFrame) -> Grader:
    
    expected_answer = task.answer(df)

    if generated_result.success == False:
        correctness = False
        obtained_answer = None
        
    else:
        obtained_answer = generated_result.stdout.strip()
        if isinstance(expected_answer, (float, np.floating)):
            try:
                correctness = math.isclose(expected_answer, float(obtained_answer))
            except ValueError:
                correctness = False
        else:
            correctness = (expected_answer == obtained_answer)
            
    return Grader(
        correct=correctness,
        expected_answer=expected_answer,
        obtained_answer=obtained_answer,
        generated_script=generated_result.code
    )


                


if __name__ == '__main__':

    path = here() / 'data' / 'raw' / 'carbon_2020-01-01_2025-12-31.parquet'
    df = pd.read_parquet(path)
    
    code="""
import pandas as pd
df=pd.read_parquet('carbon_2020-01-01_2025-12-31.parquet')
print(df[df["from"] == "2025-01-15 13:30:00"]["forecast"].iloc[0])
"""

    test = run_code(code,5)
    graded = grade(test,task1, df)
    #print(test)
    print(graded)
