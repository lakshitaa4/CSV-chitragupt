import streamlit as st
import pandas as pd
import os
import re
import json
import logging
from typing import TypedDict, List, Dict, Optional

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LangGraph State and Node Functions (No changes here) ---

class GraphState(TypedDict):
    """Represents the state of our graph."""
    original_question: str
    questions_to_process: List[str]
    question_answer_pairs: Dict[str, str]
    final_answer: str
    logs: List[Dict]
    decision: Optional[str]
    pandas_agent: any
    memoization_cache: Dict[str, str]

def start_node(state: GraphState) -> Dict:
    state["logs"].append({"step": "Start", "output": "Graph execution started."})
    return {}

def complexity_node(state: GraphState) -> Dict:
    llm = st.session_state.llm
    question = state["questions_to_process"][0]
    prompt = f"You are a query classifier. Determine if a question is 'Atomic' or 'Complex'. Question: \"{question}\". Respond with only 'Atomic' or 'Complex'."
    try:
        decision = llm.invoke(prompt).content.strip()
        state["logs"].append({"step": "Complexity Decision", "input": question, "output": decision})
        return {"decision": decision}
    except Exception as e:
        logger.error(f"LLM call failed in complexity_node: {e}")
        return {"decision": "Atomic", "logs": state["logs"] + [{"step": "Error", "message": "LLM call failed, defaulting to Atomic."}]}

def decomposition_node(state: GraphState) -> Dict:
    llm = st.session_state.llm
    question = state["questions_to_process"].pop(0)
    prompt = f"""You are an expert query decomposer. Break down a complex question into a series of simpler, atomic sub-questions. Output ONLY a valid JSON list of strings. 
---
Example 1:
Original Question: "Which store is most profitable, and what is its top selling item by quantity?"
Output:
["What is the total profit for each store?", "Which store has the highest total profit?", "For the most profitable store, what is its item description with the highest quantity sold?"]
---
Original Question: {question}
Output:
"""
    try:
        response = llm.invoke(prompt).content.strip()
        cleaned_response = re.sub(r"```json\n?|```", "", response)
        sub_questions = json.loads(cleaned_response)
    except Exception as e:
        logger.warning(f"Failed to parse JSON for decomposition. Fallback to original. Error: {e}, Response: {response}")
        sub_questions = [question]
    new_questions_to_add = [q for q in sub_questions if q not in state["questions_to_process"] and q not in state["question_answer_pairs"]]
    state["logs"].append({"step": "Decomposition", "input": question, "output": new_questions_to_add})
    state["questions_to_process"].extend(new_questions_to_add)
    return {}

def resolver_node(state: GraphState) -> Dict:
    pandas_agent = state["pandas_agent"]
    memoization_cache = state["memoization_cache"]
    question = state["questions_to_process"].pop(0)
    if question in memoization_cache:
        answer = memoization_cache[question]
        log_step = "Data Resolver (from cache)"
    else:
        try:
            response = pandas_agent.invoke({"input": question})
            answer = response.get('output', str(response))
        except Exception as e:
            answer = f"Error answering question: {e}"
        memoization_cache[question] = answer
        log_step = "Data Resolver"
    state["question_answer_pairs"][question] = answer
    state["logs"].append({"step": log_step, "input": question, "output": answer})
    return {"memoization_cache": memoization_cache}

def aggregator_node(state: GraphState) -> Dict:
    llm = st.session_state.llm
    original_question = state["original_question"]
    sub_answers = json.dumps(state["question_answer_pairs"], indent=2)
    prompt = f"""You are an expert data analyst. Synthesize a final, user-friendly, and comprehensive answer from the following data.
Address the user's original question directly. If some parts of the question could not be answered, explain why based on the context provided.
Original Question: {original_question}
Sub-Questions and their Answers (in JSON format):
{sub_answers}
Final, Synthesized Answer:"""
    try:
        final_answer = llm.invoke(prompt).content.strip()
    except Exception as e:
        logger.error(f"LLM call failed in aggregator_node: {e}")
        final_answer = "I have gathered the following details, but encountered an error while summarizing the final answer."
    state["logs"].append({"step": "Aggregation", "input": "All collected answers", "output": final_answer})
    return {"final_answer": final_answer}

def decide_path(state: GraphState) -> str:
    return "decompose" if state['decision'] == "Complex" else "resolve"

def should_continue(state: GraphState) -> str:
    return "continue_processing" if state["questions_to_process"] else "aggregate"

# --- Streamlit UI and Application Logic ---

st.set_page_config(page_title="Chitragupt’s Great-Grandkid", layout="wide")

st.markdown(
    """<div style="text-align: center;">
    <h1>Chitragupt’s Great-Grandkid</h1>
    <h3>You might be Yamraj in this case😅<br>
    Well, I am your personal AI Scribe for Data Analysis</h3>
    <p>Upload a CSV file, and I will keep a complete record of its contents, ready to answer any complex question you have.</p>
    </div>""", unsafe_allow_html=True
)
st.divider()

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "pandas_agent" not in st.session_state:
    st.session_state.pandas_agent = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- UI: Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Google API Key", type="password", placeholder="Enter your Gemini API Key")
    
    # --- ADDED: Model Selector ---
    model_choice = st.selectbox(
        "Choose Gemini Model",
        options=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"],
        index=1,
        help="Pro is more powerful but slower. Flash is faster and more cost-effective."
    )
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Core Application Logic ---

if api_key_input:
    os.environ['GOOGLE_API_KEY'] = api_key_input

# --- ADDED: Session Reset on New File Upload ---
if uploaded_file is not None and uploaded_file.name != st.session_state.get("current_file"):
    st.info(f"New file '{uploaded_file.name}' uploaded. Resetting session.")
    st.session_state.messages = []
    st.session_state.llm = None
    st.session_state.graph = None
    st.session_state.pandas_agent = None
    st.session_state.current_file = uploaded_file.name
    # Force a rerun to clear the UI and re-initialize with the new file
    st.rerun()

# --- MODIFIED: Function to initialize components now takes model_choice ---
@st.cache_resource
def initialize_components(_df, _api_key_present, model_name):
    if not _api_key_present:
        return None, None, None
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        pandas_agent = create_pandas_dataframe_agent(llm, _df, verbose=False, agent_executor_kwargs={"handle_parsing_errors": True}, allow_dangerous_code=True)
        workflow = StateGraph(GraphState)
        workflow.add_node("start_node", start_node)
        workflow.add_node("complexity_node", complexity_node)
        workflow.add_node("decomposition_node", decomposition_node)
        workflow.add_node("resolver_node", resolver_node)
        workflow.add_node("aggregator_node", aggregator_node)
        workflow.set_entry_point("start_node")
        workflow.add_edge("start_node", "complexity_node")
        workflow.add_conditional_edges("complexity_node", decide_path, {"decompose": "decomposition_node", "resolve": "resolver_node"})
        workflow.add_edge("decomposition_node", "resolver_node")
        workflow.add_conditional_edges("resolver_node", should_continue, {"continue_processing": "complexity_node", "aggregate": "aggregator_node"})
        workflow.add_edge("aggregator_node", END)
        app = workflow.compile()
        return llm, app, pandas_agent
    except Exception as e:
        st.error(f"Failed to initialize AI components: {e}")
        return None, None, None

# 3. Main app flow
if uploaded_file and api_key_input:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or len(df.columns) < 2:
            st.error("The uploaded CSV is empty or not formatted correctly. Please upload a valid file.")
            st.stop()
            
        def sanitize_column_name(col_name):
            return re.sub(r'[^a-zA-Z0-9_]+', '_', str(col_name).lower()).strip('_')
        df.columns = [sanitize_column_name(col) for col in df.columns]
        
        with st.expander("View a sample of your uploaded data (first 5 rows)"):
            st.dataframe(df.head())
        
        # --- MODIFIED: Pass model_choice to the initializer ---
        st.session_state.llm, st.session_state.graph, st.session_state.pandas_agent = initialize_components(df, bool(api_key_input), model_choice)

        if st.session_state.graph and st.session_state.llm and st.session_state.pandas_agent:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "details" in message:
                        st.markdown("---")                    
                        with st.expander("Show Full Execution Logs"):
                            st.json(message["details"]["execution_logs"])

            if prompt := st.chat_input("Ask a question about your data..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing the records..."):
                        initial_state = {
                            "original_question": prompt,
                            "questions_to_process": [prompt],
                            "question_answer_pairs": {},
                            "final_answer": "",
                            "logs": [],
                            "decision": None,
                            "pandas_agent": st.session_state.pandas_agent,
                            "memoization_cache": {}
                        }
                        try:
                            final_state = st.session_state.graph.invoke(initial_state)
                            final_answer = final_state.get("final_answer", "Sorry, I couldn't generate an answer.")
                            details = {"sub_questions_and_answers": final_state.get("question_answer_pairs", {}), "execution_logs": final_state.get("logs", [])}
                            st.markdown(final_answer)
                            st.markdown("---")                            
                            with st.expander("Show Full Execution Logs"):
                                st.json(details["execution_logs"])
                            st.session_state.messages.append({"role": "assistant", "content": final_answer, "details": details})
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {e}")
                            logger.error(f"Graph invocation failed: {e}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        logger.error(f"File processing failed: {e}")

elif not api_key_input:
    st.info("Please provide your Google API Key in the sidebar to summon the scribe.")
elif not uploaded_file:
    st.info("Please upload a CSV file so I can begin my records.")
