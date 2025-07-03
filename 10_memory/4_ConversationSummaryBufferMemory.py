import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

# Load the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
print(f"---------gpt-3.5-turbo loaded -------------- ")

parser = StrOutputParser()

# === Prompt Template with MessagesPlaceholder ===
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert. strictly limited to 50 words."),
    MessagesPlaceholder(variable_name="chat_history"), # Expects 'chat_history' from memory
    ("human", "{user_input}") # Expects 'user_input' for current human message
])

# Initialize ConversationSummaryBufferMemory globally
# It requires an LLM for summarization and a max_token_limit.
# We explicitly set memory_key="chat_history" to match the MessagesPlaceholder.
# return_messages=True ensures the history is returned as a list of BaseMessage objects.
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=150, memory_key="chat_history", return_messages=True)

# Initialize ConversationChain directly
# We explicitly set input_key="user_input" to match the human input variable in the prompt.
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=chat_prompt,
    input_key="user_input",
    verbose=True # Set to True to see the chain's internal workings
)

conv = 0

# === Main Chainlit app ===
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    global conv
    conv += 1
    print(f"---------------- conv:{conv} -------------------")

    # Directly invoke the ConversationChain
    response_dict = conversation_chain.invoke(
        {"user_input": user_input}
    )
    
    response = response_dict["response"]

    # For demonstration, print the current state of the memory
    print(f"----------------- current memory state \n {memory.load_memory_variables({})}")

    # Send response to user
    await cl.Message(content=response).send()

