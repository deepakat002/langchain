# runnable_parallel.py
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI # Import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define a prompt for summarization
summary_prompt = PromptTemplate(
    template="Summarize the following text concisely:\n\n{text}",
    input_variables=["text"]
)

# Define a prompt for keyword extraction
keywords_prompt = PromptTemplate(
    template="Extract 3-5 main keywords from the following text, comma-separated:\n\n{text}",
    input_variables=["text"]
)

# Create individual LLM chains for each task
summary_chain = summary_prompt | llm | StrOutputParser()
keywords_chain = keywords_prompt | llm | StrOutputParser()

# Use RunnableParallel to run these chains concurrently.
# The input to parallel_processor will be passed to each sub-chain.
# The output will be a dictionary where keys are 'summary' and 'keywords'.
parallel_processor = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain
)

# Example text to process
input_text = (
    "The new AI model achieved groundbreaking results in natural language understanding. "
    "It demonstrated superior performance on various benchmarks, particularly in text "
    "generation and sentiment analysis. Researchers are excited about its potential "
    "applications in customer service and content creation."
)

print(f"Processing text in parallel:\n'{input_text}'\n")

# Invoke the parallel processor
# The input dictionary for invoke must contain the 'text' key expected by the prompts.
results = parallel_processor.invoke({"text": input_text})

print("--- RunnableParallel Results ---")
print(results,"\n")
print(f"Summary:\n{results['summary']}\n")
print(f"Keywords:\n{results['keywords']}")

# Explanation:
# RunnableParallel is used to execute multiple Runnables simultaneously.
# It takes a dictionary where keys define the output structure and values
# are the Runnables to run. All Runnables within RunnableParallel receive
# the same input, and their outputs are collected into a single dictionary.
# This is highly efficient for independent operations on the same data.
