
from typing import TypedDict
from langchain.agents import create_agent
from langchain.tools import tool

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import (
    StateGraph,
    START,
    END
)

import os


class PlannerState(TypedDict):

    user_input: str

    tasks: list

    classified_tasks: list

    prioritized_tasks: list

    smart_plan: list


load_dotenv()

model = ChatOpenAI(
    model=os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    ),
    temperature=0.2
)

@tool
def evaluate_priority(task: str) -> str:
    """
    Evaluate the priority of a task and return:
    High, Medium, or Low.
    """

    prompt = f"""
    Determine the priority of this task.

    Consider:

    - urgency
    - deadlines
    - importance

    Return only one value:

    High
    Medium
    Low

    Task:
    {task}
    """

    response = model.invoke(prompt)

    return response.content.strip()


priority_agent = create_agent(
    model=model,
    tools=[evaluate_priority],
    system_prompt="""
    You are a task prioritization agent.

    Your job is to determine task priorities.

    Always use the evaluate_priority tool.
    Do not guess priorities without using the tool.

    Return only the tool result.
    """
)


def extract_tasks_node(
    state: PlannerState
):
    text = state["user_input"]

    prompt = f"""
Extract all tasks from the text.d then we'll call that agent in

Return one task per line.

Text:
{text}
"""

    response = model.invoke(
        prompt
    )

    tasks = [
        task.strip("- ")
        for task in response.content.split("\n")
        if task.strip()
    ]

    return {
        "tasks": tasks
    }



def classify_tasks_node(
    state: PlannerState
):
    tasks = state["tasks"]

    classified_tasks = []

    for task in tasks:

        prompt = f"""
Classify the task into:

- Work
- Study
- Personal

Return only the category.

Task:
{task}
"""

        response = model.invoke(
            prompt
        )

        classified_tasks.append(
            {
                "task": task,
                "category":
                response.content.strip()
            }
        )

    return {
        "classified_tasks":
        classified_tasks
    }



def prioritize_tasks_node(
    state: PlannerState
):
    classified_tasks = state["classified_tasks"]

    prioritized_tasks = []

    for item in classified_tasks:

        task = item["task"]

        result = priority_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content":
                        f"Determine the priority of: {task}"
                    }
                ]
            }
        )

        priority = (
            result["messages"][-1]
            .content
            .strip()
        )

        prioritized_tasks.append(
            {
                "task": task,
                "category": item["category"],
                "priority": priority
            }
        )

    return {
        "prioritized_tasks":
        prioritized_tasks
    }

def smart_plan_node(
    state: PlannerState
):
    tasks = (
        state["prioritized_tasks"]
    )

    high = []
    medium = []
    low = []

    for item in tasks:

        priority = (
            item["priority"]
            .lower()
        )

        if priority == "high":
            high.append(item)

        elif priority == "medium":
            medium.append(item)

        else:
            low.append(item)

    ordered_tasks = (
        high +
        medium +
        low
    )

    smart_plan = []

    for index, task in enumerate(
        ordered_tasks,
        start=1
    ):
        smart_plan.append(
            f"{index}. {task['task']}"
        )

    return {
        "smart_plan":
        smart_plan
    }


graph_builder = StateGraph(
    PlannerState
)

graph_builder.add_node(
    "extract_tasks",
    extract_tasks_node
)

graph_builder.add_node(
    "classify_tasks",
    classify_tasks_node
)

graph_builder.add_node(
    "prioritize_tasks",
    prioritize_tasks_node
)

graph_builder.add_node(
    "smart_plan",
    smart_plan_node
)


graph_builder.add_edge(
    START,
    "extract_tasks"
)

graph_builder.add_edge(
    "extract_tasks",
    "classify_tasks"
)

graph_builder.add_edge(
    "classify_tasks",
    "prioritize_tasks"
)

graph_builder.add_edge(
    "prioritize_tasks",
    "smart_plan"
)

graph_builder.add_edge(
    "smart_plan",
    END
)

graph = graph_builder.compile()


def run_task_planner(user_tasks: str):

    result = graph.invoke(
        {
            "user_input": user_tasks
        }
    )

    return result

    

if __name__ == "__main__":

    result = graph.invoke(
        {
            "user_input":
            """
            I need to finish my
            database assignment,
            study Python chapters
            three and four,
            email my supervisor,
            and buy groceries.
            """
        }
    )

    print("\ntasks: ")
    print(result["tasks"])

    print("\nclassified tasks: ")
    print(result["classified_tasks"])

    print("\nprioritized tasks: ")
    print(result["prioritized_tasks"])

    print("\nsmart plan: ")
    for task in result["smart_plan"]:
        print(task)