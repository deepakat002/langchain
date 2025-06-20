# chat_news_summary.py

import os
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # To load HUGGINGFACEHUB_API_TOKEN from .env

# === Load HuggingFace API-based model ===

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512
)

parser = StrOutputParser()

# === Prompt Template with MessagesPlaceholder ===
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert. Provide informative and concise answers, limited to 50 words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# === Save chat history ===
def save_history_to_txt(history: list, filepath: str = "chat_history.txt"):
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in history:
            role = msg.type.upper()
            f.write(f"{role}: {msg.content}\n\n")

def load_history_from_txt(filepath: str = "chat_history.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n\n")
            messages = []
            for block in lines:
                if block.startswith("HUMAN:"):
                    messages.append(HumanMessage(content=block.replace("HUMAN: ", "").strip()))
                elif block.startswith("AI:"):
                    messages.append(AIMessage(content=block.replace("AI: ", "").strip()))
            return messages
    except FileNotFoundError:
        return []

# === Chainlit Handler ===
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content

    # Load previous history
    chat_history = load_history_from_txt()

    # Format the prompt
    filled_prompt = chat_prompt.format(chat_history=chat_history, user_input=user_input)

    # Run model & parse output
    response = parser.invoke(llm.invoke(filled_prompt))

    # Update & save history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    save_history_to_txt(chat_history)

    # Display response
    await cl.Message(content=response).send()
