from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

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
model = ChatOpenAI()
parser = StrOutputParser()

# Chain to generate the summary
summary_gen_chain = RunnableSequence(prompt1, model, parser)

# Parallel:
# - keep the summary as 'summary'
# - generate a caption as 'caption'
parallel_chain = RunnableParallel({
    "summary": RunnablePassthrough(),  # keep the summary
    "caption": RunnableSequence(prompt2, model, parser)
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
