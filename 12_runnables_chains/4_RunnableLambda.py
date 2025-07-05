# runnable_lambda.py
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# Initialize LLM (for a simple chain example)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define a simple Python function that converts text to uppercase
def to_uppercase(text: str) -> str:
    """Converts the input text to uppercase."""
    print(f"  [RunnableLambda] Converting '{text}' to uppercase...")
    return text.upper()

# Define a simple Python function that adds a greeting
def add_greeting(name: str) -> str:
    """Adds a greeting to a given name."""
    print(f"  [RunnableLambda] Adding greeting to '{name}'...")
    return f"Hello, {name}!"

# Create RunnableLambda instances from the functions
uppercase_runnable = RunnableLambda(to_uppercase)
greeting_runnable = RunnableLambda(add_greeting)

print("--- RunnableLambda Examples ---")

# Example 1: Using RunnableLambda for text transformation
input_text = "Holla! runnables"
print(f"Input for uppercase: '{input_text}'")
result_uppercase = uppercase_runnable.invoke(input_text)
print(f"Output (uppercase): '{result_uppercase}'\n")
print("---------------------xxx--------------------")

# Example 2: Using RunnableLambda in a simple chain with LLM
# This demonstrates how a custom Python function (wrapped in RunnableLambda)
# can fit into a LangChain Expression Language (LCEL) chain.
# The `greeting_runnable` will produce the input for the prompt.
prompt_for_llm = PromptTemplate(
    template="Write a short, friendly message to {person_greeting}.",
    input_variables=["person_greeting"]
)
llm_chain = prompt_for_llm | llm | StrOutputParser()

# Chain: RunnableLambda -> Prompt -> LLM -> Parser
full_greeting_chain = greeting_runnable | llm_chain

input_name_for_llm = "Bob"
print("greeting_runnable.invoke: ", greeting_runnable.invoke(input_name_for_llm))
print(f"Input name for LLM chain: '{input_name_for_llm}'")
result_llm_chain = full_greeting_chain.invoke(input_name_for_llm)
print(f"LLM generated message:\n{result_llm_chain}")

# Explanation:
# RunnableLambda allows you to wrap any standard Python function or lambda
# expression, making it a compatible component within LangChain's Runnable
# ecosystem. This is incredibly useful for injecting custom logic, data
# preprocessing, or transformation steps directly into your LLM-powered chains.
