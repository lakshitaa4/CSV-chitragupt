# CSV Chitragupt

---

## Overview

This project implements a **Recursive Question Decomposer** capable of answering **complex, multi-layered analytical questions** over a **convenience store sales dataset**.  

Using **LangGraph for stateful orchestration** and an **LLM (Gemini or OpenAI)** for decision-making and synthesis, the system recursively decomposes complex questions into atomic sub-questions, answers each using **Pandas data queries**, and aggregates a final human-readable response.

---

## Dataset Details

The system works on a **CSV file** containing convenience store sales data.  
Each row represents a transaction with columns like:

- `store_name`
- `department`
- `revenue`
- `quantity`
- `profit`
- *(More columns may exist depending on the dataset)*

> _Note: A sample CSV or dataframe preview can be found inside the notebook._

---

## Tech Stack & Tools Used

- **Python**
- **LangChain**
- **LangGraph**
- **Gemini LLM** 
- **Pandas**
- **CLI 
- **ChromaDB

---

## System Architecture / Node Flow

```plaintext
[Start Node]
     ↓
[Complexity Decision Node] → (Is question Atomic or Complex?)
     ↓
If Complex → [Decomposition Node] → [Resolver Node] → Loop
If Atomic → [Resolver Node] → Loop
     ↓
[Aggregator Node] → Final Answer
```
--- 

## Node Responsibilities
| Node | Function |
|:-----|:---------|
| Start Node | Initializes execution and logs the starting state. |
| Complexity Decision Node | Uses LLM to classify if a question is Atomic or Complex. |
| Decomposition Node | Breaks down complex questions into sequential atomic sub-questions using LLM-driven few-shot prompting. |
| Resolver Node | Answers atomic questions by querying the Pandas dataframe (acts as a Pandas Data Agent). |
| Aggregator Node	| Synthesizes all sub-question answers into a final answer using LLM summarization. |
| Memoization Cache	| Caches already answered sub-questions to avoid redundant computation. |

--- 

## How to Run
* Set up your LLM API key:
  - Configure your .env file or environment variables for Gemini/OpenAI.

* Load your CSV Dataset
  - The notebook already reads and loads the dataset using Pandas.

* Run the Notebook
  - Open the .ipynb file and run all cells sequentially.

* Ask your Analytical Question
  - The system will prompt for user input (via CLI in notebook).

---

## Output Format
```json
{
  "original_question": "...",
  "final_answer": "...",
  "sub_questions_and_answers": {
    "Sub-question 1": "Answer 1",
    "Sub-question 2": "Answer 2"
  },
  "logs": [
    {"step": "Start", "input": "...", "output": "..."},
    {"step": "Complexity Decision", "input": "...", "output": "..."},
    ...
  ]
}
```

---

## Example Questions Supported
- Which product category had the highest revenue in Q1 2023 across all stores?
- Which store had the highest average basket size in March 2023?
- What are the top-selling products by revenue in each store?
- Did sales increase or decrease for beverages from January to June 2023?
- Which store had the highest profit, and what was the revenue for the 'CO : HOT FOOD' department in that store?

---

## Features Implemented
* Recursive, LLM-driven question decomposition
* Atomic sub-question resolution via Pandas
* Aggregated, user-friendly final answer generation
* JSON-structured output with reasoning logs (forming the tree like structure)
* Memoization / Caching for repeated sub-questions
* CLI-based user interaction 

---
