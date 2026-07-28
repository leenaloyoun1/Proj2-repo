# AI Task Planner Agent

## Project Overview

This project was developed using LangChain, LangGraph, OpenAI, and Flask.

The application helps users organize daily tasks using artificial intelligence by:

- Extracting tasks
- Classifying tasks
- Prioritizing tasks
- Generating a smart task plan

---

## Features

### Single Chain

Summarizes user-provided text.

### Sequential Chain

Processes tasks through multiple steps:

1. Task Summarization
2. Word Counting
3. Task Classification

### Agents

#### File Summarization Agent

Reads and summarizes text files.

#### Word Count Agent

Counts occurrences of a specific word inside a text file.

### LangGraph Task Planner

Workflow:

```text
Extract Tasks
↓
Classify Tasks
↓
Prioritize Tasks
↓
Generate Smart Plan
```

### Flask Interface

Displays:

- Original Input
- Extracted Tasks
- Task Classifications
- Task Priorities
- Smart Plan

---

## Technologies Used

- Python
- LangChain
- LangGraph
- OpenAI API
- Flask
- HTML
- CSS

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

Run the application:

```bash
python app.py
```

---

## Project Structure

```text
Proj2-repo/
│
├── app.py
├── agent.py
├── graph.py
├── singleChain.py
├── sequentialChain.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── sampleTasks.txt
├── requirements.txt
├── .env.example
└── README.md
```