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

task3 = Task(
    id=3,
    tier=1,
    question='What is the forecast carbon intensity for the settlement period ending at 2024-05-18 11:00:00?',
    answer=lambda df: df[df['to'] == '2024-05-18 11:00:00']['forecast'].iloc[0]

)

task4 = Task(
    id=4,
    tier=2,
    question='What is the maximum forecast carbon intensity during periods classified as "very high"?',
    answer=lambda df: df[df['index']=='very high']['forecast'].max()
)

task5 = Task(
    id=5,
    tier=2,
    question='How many settlement periods are classified as "low"?',
    answer=lambda df: df[df['index']=='low']['to'].count()
)

task6 = Task(
    id=6,
    tier=2,
    question='What is the minimum actual carbon intensity during periods classified as "moderate"?',
    answer=lambda df: df[df['index']=='moderate']['actual'].min()
)

task7 = Task(
    id=7,
    tier=3,
    question='What is the average forecast carbon intensity during periods where the actual carbon intensity exceeded 250?',
    answer=lambda df: df[df['actual']>250]['forecast'].mean()
)

task8 = Task(
    id=8,
    tier=3,
    question='How many settlement periods have a forecast carbon intensity greater than the actual carbon intensity?',
    answer=lambda df: (df['forecast'] > df['actual']).sum()
)

task9 = Task(
    id=9,
    tier=3,
    question='What is the average absolute difference between forecast and actual carbon intensity?',
    answer=lambda df: np.abs(df['forecast'] - df['actual']).mean()
)

task10 = Task(
    id=10,
    tier=3,
    question='Which carbon intensity category (index) occurs most frequently?',
    answer=lambda df: df['index'].mode().iloc[0]
)

task11 = Task(
    id=11,
    tier=3,
    question='For each carbon intensity category, compute the average forecast carbon intensity and return the category with the highest average.',
    answer=lambda df: df.groupby('index')['forecast'].mean().idxmax()
)

task12 = Task(
    id=12,
    tier=3,
    question='During which settlement period was the absolute forecasting error (|forecast - actual|) the largest? Return the start timestamp (from).',
    answer=lambda df: df.loc[np.abs(df['forecast'] - df['actual']).idxmax()]['from']
)

task13 = Task(
    id=13,
    tier=3,
    question='What is the average forecast carbon intensity during periods classified as "high" where the actual carbon intensity exceeded 200?',
    answer=lambda df: df[(df["index"] == "high") & (df["actual"] > 200)]["forecast"].mean()
)


task14 = Task(
    id=14,
    tier=3,
    question='What was the average forecast carbon intensity during January 2025?',
    answer=lambda df: df[(df["from"].dt.year == 2025) &(df["from"].dt.month == 1)]["forecast"].mean()
)


task15 = Task(
    id=15,
    tier=2,
    question='How many settlement periods have missing actual carbon intensity values?',
    answer=lambda df: df["actual"].isna().sum()
)


task16 = Task(
    id=16,
    tier=3,
    question='What percentage of settlement periods had a forecast carbon intensity higher than the actual carbon intensity?',
    answer=lambda df: ((df["forecast"] > df["actual"]).sum() / len(df)) * 100
)


task17 = Task(
    id=17,
    tier=4,
    question='What was the highest 24-hour rolling average forecast carbon intensity?',
    answer=lambda df: df.sort_values("from").set_index("from")["forecast"].rolling("24h").mean().max()
)

tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11, task12, task13, task14, task15, task16, task17]
train = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, task11, task12]
test = [task13, task14, task15, task16, task17]

def grade(generated_result: ExecutionResult, task: Task, df: pd.DataFrame) -> Grader:
    
    expected_answer = task.answer(df)

    if generated_result.success == False:
        correctness = False
        obtained_answer = None
        
    else:
        obtained_answer = generated_result.stdout.strip()
        if isinstance(expected_answer, (float, np.floating)) or isinstance(expected_answer, (int, np.integer)):
            try:
                correctness = math.isclose(expected_answer, float(obtained_answer))
            except ValueError:
                correctness = False
        elif isinstance(expected_answer, pd.Timestamp):
            try:
                expected_answer = expected_answer.tz_convert('UTC')
                obtained_answer = pd.Timestamp(obtained_answer)

                if obtained_answer.tzinfo is None:
                    obtained_answer = obtained_answer.tz_localize("UTC")
                else:
                    obtained_answer = obtained_answer.tz_convert("UTC")

                correctness = (expected_answer == obtained_answer)

            except Exception:
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

    from analyst_agent.core import get_configs
    df = pd.read_parquet(get_configs()['carbon_data'])
    
    code="""
import pandas as pd
df=pd.read_parquet('carbon_2020-01-01_2025-12-31.parquet')
print(df[df["from"] == "2025-01-15 13:30:00"]["forecast"].iloc[0])
"""

    test = run_code(code,5)
    graded = grade(test,task1, df)
    #print(test)
    print(graded)
