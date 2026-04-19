import os
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

# LangGraph import (handles version differences)
try:
    from langgraph.prebuilt import create_react_agent as create_agent
except ImportError:
    from langchain.agents import create_agent

from langgraph.checkpoint.mongodb import MongoDBSaver

# ----------------------
# 1. Load environment variables
# ----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# ----------------------
# 2. Define Search Tool (FIXED)
# ----------------------
def searchoninternet(query: str):
    """Search the internet and return formatted results"""
    results = DDGS().text(query, max_results=5)

    output = ""
    for r in results:
        output += f"Title: {r.get('title')}\n"
        output += f"Snippet: {r.get('body')}\n"
        output += f"Link: {r.get('href')}\n\n"

    return output

# ----------------------
# 3. MongoDB Checkpointer
# ----------------------
client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(client, db_name="rafay")

# ----------------------
# 4. Initialize Model (UPDATED)
# ----------------------
model = ChatGroq(
    model="llama-3.1-8b-instant",   # You can change to 70b if needed
    temperature=0.1,
    max_retries=2,
    api_key=GROQ_API_KEY
)

# ----------------------
# 5. Create Agent
# ----------------------
agent = create_agent(
    model,
    tools=[searchoninternet],
    checkpointer=checkpointer
)

# ----------------------
# 6. Streamlit UI
# ----------------------
st.set_page_config(page_title="Search Agent", page_icon="🔍")
st.title("DuckDuckGo Search Agent")

# Session thread
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "new_session_1"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ----------------------
# 7. Display Chat History
# ----------------------
def display_chat_history():
    st.subheader("Chat History")
    try:
        state = agent.get_state(config)
        messages = state.values.get("messages", [])

        for msg in messages:
            if msg.type in ["human", "ai"]:
                role = "user" if msg.type == "human" else "assistant"
                with st.chat_message(role):
                    st.write(msg.content)
    except Exception:
        st.info("No chat history yet.")

display_chat_history()

# ----------------------
# 8. User Input + Agent Call (FIXED)
# ----------------------
user_query = st.chat_input("Enter your search query...")

if user_query:
    # Show user message
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Searching..."):
        try:
            response = agent.invoke(
                {
                    "messages": [
                        ("system", """You are a search assistant.

You MUST use the search tool for:
- places (like buildings, universities, companies)
- current information
- weather
- anything unknown

Do NOT answer from your own knowledge.
Always call the search tool first, then answer based on the results.

If you do not use the tool, your answer is wrong.
"""),
                        ("user", user_query)
                    ]
                },
                config=config
            )

            ai_response = response["messages"][-1].content

        except Exception as e:
            ai_response = f"Error: {str(e)}"

    # Show AI response
    with st.chat_message("assistant"):
        st.write(ai_response)