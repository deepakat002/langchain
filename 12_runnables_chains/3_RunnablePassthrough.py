from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough,RunnableLambda

from dotenv import load_dotenv

load_dotenv()

# Prompt 1: generate a short summary
prompt1 = PromptTemplate(
    template="Write a short summary about {topic}.",
    input_variables=["topic"]
)

# Prompt 2: generate a catchy caption from the summary
prompt2 = PromptTemplate(
    template="Write a catchy caption for this summary: {text} in 5 words",
    input_variables=["text"]
)

# Initialize LLM and parser
llm = ChatOpenAI()
parser = StrOutputParser()

# Chain to generate the summary
summary_gen_chain = RunnableSequence(prompt1, llm, parser)

# Parallel:
# - keep the summary as 'summary'
# - generate a caption as 'caption'
parallel_chain = RunnableParallel({
    "summary": RunnablePassthrough(),  # keep the summary
    "caption": RunnableSequence(prompt2, llm, parser)
})

# Combine them
final_chain = RunnableSequence(summary_gen_chain, parallel_chain)

# Run
result = final_chain.invoke({"topic": "pitbull"})

print("--- Final Result ---")
print(result)


# -------------------------------------------------------------
# Explanation:
# 1. RunnableSequence:
#    Chains steps in a strict order, passing output of each step to the next.
#    summary_gen_chain = prompt1 → LLM → parser
#
# 2. RunnableParallel:
#    Executes multiple runnables *in parallel* with the same input:
#    - one path (RunnablePassthrough) keeps the generated summary
#    - another path generates a caption for the same summary
#    It collects both outputs into a dictionary:
#    {
#        "summary": <original summary>,
#        "caption": <catchy caption>
#    }
#
# 3. final_chain:
#    - First generates a summary about 'pitbull'
#    - Then sends that summary into the parallel branch:
#         - preserving the summary
#         - creating a caption for it
#
# In short:
# RunnableParallel allows you to fork the same input to multiple chains and gather
# their results, which is highly efficient for independent tasks on the same data.


##### with assign 

print(f"==================================================\n")

# Define a prompt that expects 'topic' and 'adjective'
prompt = PromptTemplate(
    template="Write a {adjective} short story about {topic} in 50 words.",
    input_variables=["adjective", "topic"]
)
# Create a simple chain: Prompt -> LLM -> Parser
llm_chain = prompt | llm | StrOutputParser()
print("--- RunnablePassthrough.assign with LLM Integration ---")

dynamic_story_chain = (
    RunnablePassthrough.assign(
        adjective=lambda x: "hilarious" if "cat" in x["topic"].lower() else "thrilling"
    )
    | llm_chain # Receives {"topic": ..., "adjective": ...}
)

### 
"""
If you pass "cats"

topic → "cats"

adjective → "hilarious"
→ output is {"topic": "cats", "adjective": "hilarious"}

If you pass "space exploration"

topic → "space exploration"

adjective → "thrilling"
→ output is {"topic": "space exploration", "adjective": "thrilling"}
"""
# Invoke the chain with a dictionary containing the topic
story_topic_1 = "cats"
print(f"Input topic: '{story_topic_1}' (passed as dictionary)")
story_result_1 = dynamic_story_chain.invoke({"topic": story_topic_1}) # Pass dictionary
print(f"Generated story (for '{story_topic_1}'):\n{story_result_1}\n")

story_topic_2 = "space exploration"
print(f"Input topic: '{story_topic_2}' (passed as dictionary)")
story_result_2 = dynamic_story_chain.invoke({"topic": story_topic_2}) # Pass dictionary
print(f"Generated story (for '{story_topic_2}'):\n{story_result_2}")