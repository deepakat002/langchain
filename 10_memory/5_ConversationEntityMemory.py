import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationEntityMemory
from langchain.chains import ConversationChain

# Load the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
print(f"---------gpt-3.5-turbo loaded -------------- ")

parser = StrOutputParser()

# === Prompt Template with MessagesPlaceholder ===
# ConversationEntityMemory provides both 'history' (messages) and 'entities' (facts).
# We need to include both in the prompt for the LLM to leverage entity knowledge.
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert. strictly limited to 50 words. "
               "Here are some facts about entities you know: {entities}"), # Add entities to system prompt
    MessagesPlaceholder(variable_name="history"), # Changed to "history" to match ConversationEntityMemory's output
    ("human", "{user_input}") # Expects 'user_input' for current human message
])

# Initialize ConversationEntityMemory globally
# It requires an LLM to extract entities and their facts.
# Note: memory_key for ConversationEntityMemory refers to the key for the *entity summary*,
# not the chat history messages themselves. The chat history messages are always under 'history'.
# return_messages=True ensures the chat history is returned as a list of BaseMessage objects.
memory = ConversationEntityMemory(llm=llm, return_messages=True)

# Initialize ConversationChain directly
# We explicitly set input_key="user_input" to match the human input variable in the prompt.
# Note: ConversationChain will automatically pass 'entities' if it's in the prompt
# and provided by the memory.
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

    # For demonstration, print the current state of the memory (including entities)
    # IMPORTANT: For ConversationEntityMemory, load_memory_variables needs the current
    # input to correctly retrieve/update entity facts.
    print(f"----------------- current memory state \n {memory.load_memory_variables({'user_input': user_input})}")

    # Send response to user
    await cl.Message(content=response).send()
