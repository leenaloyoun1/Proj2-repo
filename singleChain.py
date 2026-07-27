import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Please add it to the .env file."
    )


summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a clear and accurate text summarizer.

Summarize the text below in one concise paragraph.
Preserve important tasks, deadlines, and requirements.
Do not introduce information that does not appear in the original text.

Text:
{text}

Summary:
""",
)


model = ChatOpenAI(
    model=model_name,
    temperature=0.2,
)



output_parser = StrOutputParser()

summary_chain = summary_prompt | model | output_parser


sample_text = """
This week I have several responsibilities that I need to organize.
For university, I must complete my database assignment and submit it
before Thursday at 11:59 PM. The assignment includes creating the
database tables, writing SQL queries, testing the queries, and preparing
screenshots of the results.

I also have a Python exam next Sunday, so I need to study chapters three
and four, review the practical exercises, and solve at least two previous
exams. During my practical training, I need to email my supervisor with
an update about the AI Task Planner project and prepare questions about
LangChain and LangGraph for our next meeting.

Outside university and training, I need to buy groceries after class,
clean my room, and call the dentist to arrange an appointment. The
database assignment is currently the most urgent task because its
deadline comes first.
"""


summary = summary_chain.invoke(
    {
        "text": sample_text,
    }
)


print("\noriginal text: ")
print(sample_text.strip())

print("\nsummary: ")
print(summary)