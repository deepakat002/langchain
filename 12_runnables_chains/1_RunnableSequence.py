# runnable_sequence.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 1️⃣ PromptTemplate
prompt = PromptTemplate(
    template="Generate a short, creative name for a {product_type} called {product_name}.",
    input_variables=["product_type", "product_name"]
)

# 2️⃣ OutputParser to extract text
parser = StrOutputParser()

# 3️⃣ Create the RunnableSequence
# Here using the convenient pipe syntax:
name_generation_chain = prompt | llm | parser

# If you prefer, you can also explicitly write it with RunnableSequence:
# name_generation_chain = RunnableSequence(prompt, llm, parser)

print("--- RunnableSequence Example with OpenAI Integration ---\n")

# Example input 1
input_data = {"product_type": "coffee shop", "product_name": "Morning Brew"}
print(f"Input: {input_data}")
result = name_generation_chain.invoke(input_data)
print(f"Generated Name:\n{result}\n")

# -------------------------------------------------------------
# Explanation:
# - RunnableSequence allows you to chain multiple steps.
# - prompt → llm → parser
# - each step’s output automatically becomes the next step’s input.
# - you can write it explicitly as:
#
#   name_generation_chain = RunnableSequence(prompt, llm, parser)
#
#   or use the | pipe syntax:
#
#   name_generation_chain = prompt | llm | parser
#
#   both are equivalent in functionality.
