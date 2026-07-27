import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Please add it to the .env file."
    )


model = ChatOpenAI(
    model=model_name,
    temperature=0,
)


class ExtractedTask(BaseModel):
    """One task extracted from the user's text."""

    summary: str = Field(
        description="A concise and clear summary of one individual task."
    )


class ExtractedTaskList(BaseModel):
    """A variable-length list of tasks."""

    tasks: list[ExtractedTask] = Field(
        description="Every individual task found in the user's text."
    )


class TaskClassification(BaseModel):
    """The category assigned to one task."""

    category: Literal["Work", "Study", "Personal"] = Field(
        description="The category that best represents the task."
    )


task_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an AI task extraction and summarization assistant.

Read the user's text and identify every individual task.

Instructions:

1. Do not omit any task.
2. Do not create tasks that are not mentioned.
3. Do not merge separate tasks.
4. Rewrite each task as one concise and clear action.
5. Preserve important deadlines and requirements.
6. The input may contain any number of tasks.

User's text:
{text}
""",
)

structured_task_model = model.with_structured_output(ExtractedTaskList)

task_extraction_chain = (
    task_extraction_prompt
    | structured_task_model
)


classification_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
Classify the following task into exactly one category.

Categories:

- Work: employment, practical training, professional responsibilities,
  meetings, clients, supervisors, or workplace projects.

- Study: university, school, assignments, exams, studying, courses,
  academic projects, or educational activities.

- Personal: household activities, shopping, appointments, family,
  friends, hobbies, health routines, or other private responsibilities.

Choose the single category that best represents the task.

Task:
{task}
""",
)

structured_classification_model = model.with_structured_output(
    TaskClassification
)

classification_chain = (
    classification_prompt
    | structured_classification_model
)


def count_words(text: str) -> int:
    """
    Count word-like elements in a string.
    """

    words = re.findall(r"\b[\w'-]+\b", text)

    return len(words)


def process_tasks(text: str) -> list: 
    """
    Extract, summarize, count, and classify every task in the text.
    """

    extracted_result = task_extraction_chain.invoke(
        {
            "text": text,
        }
    )

    final_results = []

    for task_number, extracted_task in enumerate(
        extracted_result.tasks,
        start=1,
    ):
        task_summary = extracted_task.summary.strip()

        word_count = count_words(task_summary)

        classification = classification_chain.invoke(
            {
                "task": task_summary,
            }
        )

        final_results.append(
            {
                "task_number": task_number,
                "summary": task_summary,
                "word_count": word_count,
                "category": classification.category,
            }
        )

    return final_results



sample_text = """
This week, I need to finish my database assignment and submit it before
Thursday at 11:59 PM. I also need to study chapters three and four for
the Python exam next Sunday and solve at least two previous exams.

For my practical training, I must email my supervisor with an update
about the AI Task Planner project and prepare questions about LangChain
and LangGraph for our next meeting. After class, I need to buy groceries,
clean my room, and call the dentist to arrange an appointment.
"""



results = process_tasks(sample_text)


print("\nORIGINAL TASK INPUT")
print("-------------------")
print(sample_text.strip())

print("\nPROCESSED TASKS")
print("---------------")

print(f"Number of tasks found: {len(results)}")

for result in results:
    print(f"\nTask {result['task_number']}")
    print(f"Summary: {result['summary']}")
    print(f"Word count: {result['word_count']}")
    print(f"Category: {result['category']}")