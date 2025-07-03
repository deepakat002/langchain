import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import ConversationChain

# Load the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
print(f"---------gpt-3.5-turbo loaded -------------- ")

parser = StrOutputParser()

# === Prompt Template for VectorStoreRetrieverMemory ===
# VectorStoreRetrieverMemory typically returns a string of retrieved documents.
# We'll use a standard {retrieved_history} placeholder for the retrieved context.
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert. strictly limited to 50 words. "
               "Here is some relevant past conversation context: {retrieved_history}"),
    ("human", "{user_input}") # Expects 'user_input' for current human message
])

# === Setup for VectorStoreRetrieverMemory ===
# 1. Initialize Embeddings
embeddings = OpenAIEmbeddings()

# 2. Create a dummy VectorStore (Chroma in-memory for this example)
# Add some initial documents to the vector store to simulate existing knowledge
docs = [
    Document(page_content="My dog's name is Max and he is a Golden Retriever. He loves to play fetch."),
    Document(page_content="I have another dog, Bella, who is a small poodle. She is very intelligent."),
    Document(page_content="Golden Retrievers are known for their friendly nature and loyalty."),
    Document(page_content="Poodles are very intelligent and can be easily trained, often used in shows."),
    Document(page_content="My cat, Whiskers, is very independent."),
]
vectorstore = Chroma.from_documents(docs, embeddings)

# 3. Create a Retriever from the VectorStore
# k=3 means it will retrieve the top 3 most relevant documents.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Initialize VectorStoreRetrieverMemory globally
# The memory_key 'retrieved_history' will be used to inject the retrieved documents into the prompt.
# Note: VectorStoreRetrieverMemory does not use return_messages=True as it returns a string.
memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="retrieved_history")

# Initialize ConversationChain directly
# We explicitly set input_key="user_input" to match the human input variable in the prompt.
# The chain will automatically pass the retrieved_history from memory to the prompt.
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
    # The chain handles loading history (retrieving documents), invoking the LLM, and saving context.
    response_dict = conversation_chain.invoke(
        {"user_input": user_input}
    )
    
    response = response_dict["response"]

    # For demonstration, print the current state of the memory (retrieved docs based on last input)
    # Note: load_memory_variables will perform a retrieval based on a dummy input here
    # to show what it would retrieve, as it needs an input to query the vector store.
    print(f"----------------- current memory state (retrieved based on last input) \n {memory.load_memory_variables({'user_input': user_input})}")

    # Send response to user
    await cl.Message(content=response).send()

