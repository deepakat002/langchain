# chat_news_summary.py

import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Load the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
print(f"---------gpt-3.5-turbo loaded -------------- ")

parser = StrOutputParser()

# === Prompt Template with MessagesPlaceholder ===
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert.strictly limited to 50 words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# === Helper to load/save chat history ===
def save_history_to_txt(history: list, filepath: str = "1_chat_history.txt"):
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in history:
            role = msg.type.upper()
            f.write(f"{role}: {msg.content}\n\n")

def load_history_from_txt(filepath: str = "1_chat_history.txt"):
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

conv = 0
# === Main Chainlit app ===
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    global conv
    conv +=1
    print(f"---------------- conv:{conv} -------------------\n")

    # Load prior history from file
    chat_history = load_history_from_txt()

    # Fill the prompt with chat history and new user input
    filled_prompt = chat_prompt.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    print(f"------------------- filled prompt --------------\n{filled_prompt}\n\n")

    # Run the model
    response = parser.invoke(llm.invoke(filled_prompt))

    # Append the new interaction to history and save
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    save_history_to_txt(chat_history)
    print(f"----------------- chat history \n {chat_history}")

    # Send response to user
    await cl.Message(content=response).send()
