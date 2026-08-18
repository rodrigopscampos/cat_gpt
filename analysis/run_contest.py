import os
import json
import asyncio
from pydantic import BaseModel
from typing import Literal, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

### Loading Q&A from the Ollama models ###

df_list = []

for _json in os.listdir("./results"):

    df_list.append(pd.read_json(f"./results/{_json}"))

results_df = pd.concat(df_list, ignore_index=True)

answers_df = results_df.pivot_table(
    index=["_id", "question"],
    columns="model_name",
    values=["model_answer"], #["model_answer", "latency_seconds"],
    aggfunc="first"
)

answers_df.columns = [model for field, model in answers_df.columns]
answers_df = answers_df.reset_index()

### Creating Langchain classes for the contest ###

class Winner(BaseModel):

    winner: Literal[0, 1, 2]
    question: Optional[str] = None
    question_id: Optional[int] = None

answer_parser = PydanticOutputParser(pydantic_object=Winner)

answer_prompt = PromptTemplate.from_file(
    template_file="./prompts/judge_instructions.md",
    # partial_variables={"format_instructions": answer_parser.get_format_instructions()},
)

answer_models = [
    ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    ChatAnthropic(model="claude-haiku-4-5-20251001"),
    ChatAnthropic(model="claude-sonnet-5"),
    ChatAnthropic(model="claude-sonnet-4-6"),
]

answer_chain = RunnableParallel(
    {model.model: answer_prompt | model | answer_parser for model in answer_models}
)

contests = [
    ("gemma3:1b", "gemma3:4b"),
    ("llama3.2:1b", "llama3.2:3b"),
    ("gemma3:1b", "llama3.2:1b"),
    ("gemma3:4b", "llama3.2:3b"),
    ("gemma3:1b", "llama3.2:3b"),
    ("llama3.2:1b", "gemma3:4b"),
]

def get_winner(choices: list):

    if choices.count(1) > choices.count(2):
        return 1
    elif choices.count(1) < choices.count(2):
        return 2
    else:
        return 0


### Evaluating the contest between the models ###

async def main():

    for index in range(2, 101):

        models_score = {
            "_id": index,
            "gemma3:1b": 0,
            "gemma3:4b": 0,
            "llama3.2:1b": 0,
            "llama3.2:3b": 0
        }

        row = answers_df.loc[answers_df["_id"] == index].to_dict(orient="records")[0]

        question = row.get("question")

        contest_inputs = [
            {
                'answer_1': row.get(contest[0]),
                'answer_2': row.get(contest[1]),
                'question': question
            }
            for contest in contests
        ]

        all_scores = await answer_chain.abatch(
            contest_inputs,
            config={"max_concurrency": 8},
        )

        for contest, scores in zip(contests, all_scores):

            choices = [value.winner for value in scores.values()]

            result = get_winner(choices)

            if result == 1:
                print(f"For Question {index}, model {contest[0]} won against {contest[1]}.")
                models_score[contest[0]] += 1

            elif result == 2:
                print(f"For Question {index}, model {contest[1]} won against {contest[0]}.")
                models_score[contest[1]] += 1

            elif result == 0:
                print(f"For Question {index}, model {contest[0]} tied against {contest[1]}.")

        with open(f"./contest/question_{index}.json", "w", encoding="utf-8") as f:

            json.dump(models_score, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main())