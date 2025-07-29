import os
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Read the token from your specific environment variable
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Check if the API token is available
if not api_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file. Please add it.")

# === 1. Initialize the raw HuggingFace Endpoint ===
# NOTE: Switched to a more stable model for testing purposes.
endpoint = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',  # <-- Changed model
    task='text-generation',
    temperature=0.7,
    max_new_tokens=150, # Increased slightly for the larger model
    huggingfacehub_api_token=api_token,
)

# === 2. Wrap the endpoint with ChatHuggingFace ===
llm = ChatHuggingFace(llm=endpoint)

print("✅ Using ChatHuggingFace wrapper for mistralai/Mistral-7B-Instruct-v0.2")


# === Define the prompt template ===
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert. Provide informative and concise answers, limited to 50 words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# === Define the output parser ===
parser = StrOutputParser()

# === Create the full chain using LangChain Expression Language (LCEL) ===
chain = chat_prompt | llm | parser


# === Functions to save and load chat history ===
def save_history_to_txt(history: list, filepath: str = "4_chat_history_api.txt"):
    """Saves the chat history to a text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in history:
            role = msg.type.upper()
            f.write(f"{role}: {msg.content}\n\n")

def load_history_from_txt(filepath: str = "4_chat_history_api.txt"):
    """Loads the chat history from a text file."""
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n\n")
        messages = []
        for block in lines:
            if block.startswith("HUMAN:"):
                messages.append(HumanMessage(content=block.replace("HUMAN: ", "").strip()))
            elif block.startswith("AI:"):
                messages.append(AIMessage(content=block.replace("AI: ", "").strip()))
        return messages


# === Chainlit Handler for incoming messages ===
@cl.on_chat_start
async def on_chat_start():
    """Initial setup when a new chat starts."""
    cl.user_session.set("history", load_history_from_txt())
    await cl.Message(content="Hello! I am a dog expert. How can I help you today? 🐾").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handles new messages from the user."""
    user_input = message.content
    chat_history = cl.user_session.get("history", [])

    print(f"User input: {user_input}")

    try:
        # Asynchronously invoke the chain
        response = await chain.ainvoke({
            "chat_history": chat_history,
            "user_input": user_input
        })

        # Validate the response from the model
        if not response or not isinstance(response, str) or not response.strip():
            await cl.Message(content="Sorry, I could not generate a valid response. The model might be temporarily unavailable. Please try again.").send()
            return

        # Update and save the history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        cl.user_session.set("history", chat_history)
        save_history_to_txt(chat_history)

        print(f"Final response: {response}")

        # Send the response back to the user
        await cl.Message(content=response).send()

    except Exception as e:
        error_msg = f"Sorry, an unexpected error occurred: {str(e)}"
        await cl.Message(content=error_msg).send()