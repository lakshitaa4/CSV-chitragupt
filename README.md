<h1 align='center'> Avatar of Chitragupt </h1>

<h2 align='center'>Your Personal AI Scribe for Data Analysis.</h2>

<h2 align='center'><b>Chitragupt</b> is an intelligent, conversational web application that lets you perform complex data analysis on any CSV file simply by asking questions in plain English. </h2>

* * * * *

🚀 Live Demo
------------

* * * * *

✨ Features
----------

-   **Sophisticated Question Decomposition:** Powered by **LangGraph**, Chitragupt can break down complex multi-part questions into simpler atomic sub-questions. For example: *"Which store was most profitable, and what was its best-selling product?"*

-   **Interactive Chat UI:** An intuitive, chat-based interface built with **Streamlit** lets you engage with your data effortlessly.

-   **Analyze Any CSV:** Upload any tabular dataset and start querying instantly. The app is completely data-agnostic.

-   **Natural Language Answers:** Chitragupt queries your data using **Pandas** in the background and synthesizes the findings into clear, human-friendly responses.

-   **Transparent Reasoning:** See every sub-question and corresponding answer to understand the AI's thought process step-by-step.

-   **Dynamic Model Selection:** Choose between different **Google Gemini** models (`gemini-1.5-pro`, `gemini-1.5-flash`, etc.) for faster responses or deeper analysis.

-   **Stateful Sessions:** The app remembers your conversation until you upload a new file, at which point it resets for a fresh analysis session.

* * * * *

🛠️ Tech Stack & Tools
----------------------

-   **Python**

-   **Streamlit** 

-   **LangChain & LangGraph** 

-   **Google Gemini API** 

-   **Pandas** 

* * * * *

🧠 How It Works: The Recursive Engine
-------------------------------------

```
[Start: User asks a question]
              ↓
[Complexity Node: Atomic or Complex?]
       ↙            ↘
 [Decomposition]    [Resolver Node]
       ↓                   ↑
[Resolver Node] ←-----------|
       ↓
   [Aggregator Node]
       ↓
      [End]

```

### Node Functions:

| Node | Function |
| --- | --- |
| **Complexity Node** | Uses Gemini to classify if the question is simple (Atomic) or complex. |
| **Decomposition Node** | Breaks complex queries into a sequence of atomic sub-questions. |
| **Resolver Node** | Uses **Pandas** to answer atomic questions by executing code. |
| **Aggregator Node** | Synthesizes multiple answers into a coherent final answer for the user. |
| **Memoization Cache** | Caches answers to speed up repeated queries. |

* * * * *

⚙️ How to Run Locally
---------------------

### Prerequisites

-   Python 3.9+

-   Google Gemini API Key

### 1\. Clone the Repository

```
git clone [your-repo-url]
cd [your-repo-folder]

```

### 2\. Set Up a Virtual Environment

#### Windows

```
python -m venv .venv
.\.venv\Scripts\activate

```

#### macOS / Linux

```
python3 -m venv .venv
source .venv/bin/activate

```

### 3\. Install Dependencies

```
pip install -r requirements.txt

```

### 4\. Run the Streamlit App

```
streamlit run app.py

```

Your browser will open automatically with the app running locally.

* * * * *

❓ Example Questions You Can Ask:
--------------------------------

-   Which department had the highest total revenue, and what was its average profit margin?

-   Compare the total quantity sold for 'Product A' vs 'Product B'.

-   What are the top 3 stores by total profit?

-   For the store with the highest revenue, what were its top 5 selling items by quantity?

-   Which store had the highest profit, and what was the revenue for the 'CO : HOT FOOD' department in that store?

* * * * *

📌 Notes:
---------

-   You can select the model (`pro` or `flash`) based on your need for speed vs. depth.

-   The app auto-resets when you upload a new CSV to ensure accuracy.

* * * * *

Made with ❤️ for data storytellers.

* * * * *
