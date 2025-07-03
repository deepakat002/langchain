import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.memory.kg import ConversationKGMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ChatMessageHistory # Import ChatMessageHistory

# Load the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
print(f"---------gpt-3.5-turbo loaded -------------- ")

parser = StrOutputParser()

# === Prompt Template with MessagesPlaceholder ===
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert. strictly limited to 50 words. "
               "Here are some facts from our conversation: {kg_facts}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# Initialize ConversationKGMemory globally
kg_memory = ConversationKGMemory(llm=llm)

# In-memory session store for RunnableWithMessageHistory
# Now, we will store ChatMessageHistory objects instead of plain lists.
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Function to retrieve chat history for a given session ID.
    This is required by RunnableWithMessageHistory.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory() # Initialize ChatMessageHistory
    return store[session_id]


# Define the core runnable chain: Prompt -> LLM -> Parser
core_chain = chat_prompt | llm | parser

# Wrap the core chain with RunnableWithMessageHistory
with_message_history = RunnableWithMessageHistory(
    core_chain,
    get_session_history=get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)

conv = 0

# === Main Chainlit app ===
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    global conv
    conv += 1
    print(f"---------------- conv:{conv} -------------------")

    # Get the current session ID (Chainlit provides this)
    session_id = cl.user_session.get("id")

    # Manually load KG triplets from ConversationKGMemory
    kg_memory_state = kg_memory.load_memory_variables({"input": user_input})
    kg_facts = kg_memory_state.get("kg", "No specific facts yet.")

    # Invoke the chain.
    response = await with_message_history.ainvoke(
        {"user_input": user_input, "kg_facts": kg_facts},
        config={"configurable": {"session_id": session_id}}
    )

    # After the response, update the KG memory with the latest interaction.
    kg_memory.save_context({"input": user_input}, {"output": response})

    # For demonstration, print the current state of the KG memory
    print(f"----------------- current KG memory state \n {kg_memory.load_memory_variables({'input': user_input})}")

    # Send response to user
    await cl.Message(content=response).send()
