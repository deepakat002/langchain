# runnable_branch.py
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define different LLM chains and simple responders for each branch

# 1. Math Problem Solver Chain: Expects a single string input 'query_text'
math_solver_chain = (
    PromptTemplate.from_template("Solve the following math problem. Provide only the numerical answer: {query_text}")
    | llm
    | StrOutputParser()
)

# 2. General Question Answering Chain: Expects a single string input 'query_text'
general_qa_chain = (
    PromptTemplate.from_template("Answer the following question concisely: {query_text}")
    | llm
    | StrOutputParser()
)

# 3. Simple Greeting Responder (no LLM needed)
greeting_response_runnable = RunnableLambda(lambda x: "Hello there! How can I help you today?")

# 4. Simple Farewell Responder (no LLM needed)
farewell_response_runnable = RunnableLambda(lambda x: "Goodbye! Have a great day!")

# Define a routing function to determine which branch to take
# This function takes the raw query string and returns a category string.
def route_query_category(query: str) -> str:
    """Determines the category of the query for routing."""
    lower_query = query.lower()
    if "hello" in lower_query or "hi" in lower_query:
        return "greeting"
    elif "math" in lower_query or "calculate" in lower_query or "sum" in lower_query:
        return "math"
    elif "bye" in lower_query or "goodbye" in lower_query:
        return "farewell"
    else:
        return "general"

# Create the RunnableBranch
# It takes a list of (condition, runnable) tuples and a default runnable.
# The `condition` must be a Runnable that returns a boolean.
# We wrap the comparison `route_query_category(x) == "category"` inside a RunnableLambda.
branched_router = RunnableBranch(
    (RunnableLambda(lambda q: route_query_category(q) == "greeting"), greeting_response_runnable),
    (RunnableLambda(lambda q: route_query_category(q) == "math"), math_solver_chain),
    (RunnableLambda(lambda q: route_query_category(q) == "farewell"), farewell_response_runnable),
    general_qa_chain # Default branch if no other condition matches
)

print("--- RunnableBranch Examples with OpenAI Integration (Simplified) ---")

# Test queries
queries = [
    "Hello!",
    "What is 123 + 456?",
    "What is the capital of France?",
    "Goodbye for now!",
    "Tell me a fun fact about dogs."
]

for i, q in enumerate(queries):
    print(f"\nQuery {i+1}: '{q}'")
    # Invoke the branched_router directly with the query string.
    # The RunnableBranch will pass this string to the chosen branch.
    result = branched_router.invoke(q)
    print(f"Result: {result}")

# Explanation:
# RunnableBranch allows you to create conditional workflows, routing the
# input to different Runnables based on specific conditions. It acts like
# an if-elif-else statement for your LangChain chains, enabling dynamic
# behavior based on the input data. This simplified example demonstrates
# routing different types of queries to specialized LLM chains or simple
# text responders.
