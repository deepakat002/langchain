# Import necessary modules from the installed libraries.
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os

# -----------------------------------------------------------------------------
# Step 1: Environment Setup
# -----------------------------------------------------------------------------
# Load environment variables from a .env file. This is where you should store
# your OpenAI API key. The key should be saved as OPENAI_API_KEY.
# Example .env file content:
# OPENAI_API_KEY="YOUR_API_KEY_HERE"
load_dotenv()

# Get the API key from the environment.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it.")

# -----------------------------------------------------------------------------
# Step 2: Define LLM and Tools
# -----------------------------------------------------------------------------
# Initialize the Large Language Model (LLM) as per the user's request.
# The 'temperature' parameter is set to 0.0 for deterministic and factual responses,
# which is ideal for a task like finding a population.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Initialize the search tool using DuckDuckGoSearchRun.
# This tool allows the agent to perform web searches.
search_tool = DuckDuckGoSearchRun()

# The `create_react_agent` function expects a list of tools.
tools = [search_tool]

# -----------------------------------------------------------------------------
# Step 3: Pull the ReAct Prompt from LangChain Hub
# -----------------------------------------------------------------------------
# The LangChain Hub is a repository of pre-built prompts. We are using the
# standard 'ReAct' prompt which is designed to enable the Thought/Action/Observation
# loop that defines the ReAct pattern.
prompt = hub.pull("hwchase17/react")

# -----------------------------------------------------------------------------
# Step 4: Create the Agent and Executor
# -----------------------------------------------------------------------------
# Create the ReAct agent. This function binds the LLM, tools, and prompt
# together to form the core of the agent.
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor. This is the runtime that executes the agent's
# thought-action loop. The 'verbose=True' flag is very useful for debugging
# as it prints the entire Thought/Action/Observation process.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -----------------------------------------------------------------------------
# Step 5: Run the Agent
# -----------------------------------------------------------------------------
# Define the task for the agent.
task = "Find the population of the capital of India."

# Invoke the agent executor with the task. This starts the ReAct loop.
result = agent_executor.invoke({"input": task})

# Print the final answer provided by the agent.
print("\n" + "="*50)
print(f"Agent's Final Answer: {result['output']}")
print("="*50)
