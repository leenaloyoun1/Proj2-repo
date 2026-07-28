import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

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



PROJECT_DIRECTORY = Path(__file__).resolve().parent # for the model to find the sampletasks file


# helper function to validate the file and return a safe path for the text file inside the proj directory
def get_safe_text_file(filename: str) -> Path:

    file_path = (PROJECT_DIRECTORY / filename).resolve()

    if file_path.parent != PROJECT_DIRECTORY:
        raise ValueError(
            "The file must be located inside the project directory."
        )

    if file_path.suffix.lower() != ".txt":
        raise ValueError(
            "Only .txt files are supported."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"The file '{filename}' was not found."
        )

    return file_path



@tool
def summarize_text_file(filename: str) -> str:
    """
    Read a text file and return a concise summary.
    """
    file_path = get_safe_text_file(filename)

    file_content = file_path.read_text(
        encoding="utf-8"
    )

    if not file_content.strip():
        return "The text file is empty."

    summarization_prompt = f"""
You are a clear and accurate text summarizer.

Summarize the following text in one concise paragraph.

Instructions:

- Preserve important tasks, deadlines, and requirements.
- Do not invent information.
- Do not omit important information.
- Clearly communicate the main responsibilities.

Text:
{file_content}

Summary:
"""

    response = model.invoke(summarization_prompt) # the part that performs the AI summarization

    return response.content # returns the generated text only


@tool
def count_word_in_file(
    filename: str,
    word: str,
) -> str:
    """
    Count how many times a word appears in a text file.
    """

    file_path = get_safe_text_file(filename)

    file_content = file_path.read_text(
        encoding="utf-8"
    )

    cleaned_word = word.strip()

    if not cleaned_word:
        return "No word was supplied for counting."

    pattern = rf"\b{re.escape(cleaned_word)}\b"

    matches = re.findall(
        pattern,
        file_content,
        flags=re.IGNORECASE,
    )

    count = len(matches)

    return (
        f"The word '{cleaned_word}' appears "
        f"{count} time(s) in '{filename}'."
    )


summary_agent = create_agent(
    model=model,
    tools=[summarize_text_file],
    system_prompt="""
You are a text-file summarization agent.

When the user asks you to summarize a text file, use the
summarize_text_file tool.

Do not invent file contents.
Do not claim that you read a file unless the tool successfully reads it.
Return the summary clearly and concisely.
""",
)


word_count_agent = create_agent(
    model=model,
    tools=[count_word_in_file],
    system_prompt="""
You are a word-counting agent.

When the user asks how many times a word appears in a text file,
use the count_word_in_file tool.

Determine the filename and the requested word from the user's message.
Always use the tool instead of estimating the count yourself.
Return the tool's result clearly.
""",
)


if __name__ == "__main__":
    print("\nfile summarization: ")

    summary_result = summary_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Summarize the file sampleTasks.txt."
                    ),
                }
            ]
        }
    )

    print(summary_result["messages"][-1].content)

    print("\nword counting: ")

    count_result = word_count_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Count how many times the word Java appears in sampleTasks.txt."
                    ),
                }
            ]
        }
    )

    print(count_result["messages"][-1].content)

