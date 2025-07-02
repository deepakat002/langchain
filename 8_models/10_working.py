# chat_news_summary.py

import os
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
import torch

load_dotenv()  # To load HUGGINGFACEHUB_API_TOKEN from .env

# === Set HuggingFace cache directory ===
os.environ['HF_HOME'] = 'E:/YOUTUBE_CODES/40_LLMs/hugging_face'
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not available. Using CPU.")

os.environ['HF_HOME'] = 'E:/YOUTUBE_CODES/40_LLMs/hugging_face'

# === Load Local HuggingFace Pipeline-based model ===
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm, model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0')

print(f"----------TinyLlama/TinyLlama-1.1B-Chat-v1.0 loaded locally -------------- ")

parser = StrOutputParser()

# === Prompt Template with MessagesPlaceholder ===
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dog expert. Provide informative and concise answers, limited to 50 words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# === Save chat history ===
def save_history_to_txt(history: list, filepath: str = "4_chat_history.txt"):
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in history:
            role = msg.type.upper()
            f.write(f"{role}: {msg.content}\n\n")

def load_history_from_txt(filepath: str = "4_chat_history.txt"):
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

# === Chainlit Handler ===
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    global conv
    conv +=1
    print(f"---------------- conv:{conv} -------------------\n")

    # Load previous history
    chat_history = load_history_from_txt()

    # Format the prompt with proper structure
    formatted_messages = []
    for msg in chat_history:
        formatted_messages.append(msg)
   
    # Create the full prompt
    prompt = chat_prompt.format_messages(chat_history=chat_history, user_input=user_input)
   
    print(f"------------------- filled prompt --------------\n{prompt}\n\n")

    try:
        # Run model & parse output (using the local pipeline directly)
        response = await llm.ainvoke(prompt)
        parsed_response = parser.invoke(response)
        
        # Extract only the assistant's response (remove the prompt echo)
        # Split by "Human:" and take the last part, then split by "Assistant:" if present
        if "Human:" in parsed_response:
            # Find the last occurrence of the user input and extract response after it
            parts = parsed_response.split(user_input)
            if len(parts) > 1:
                # Get everything after the last occurrence of user input
                assistant_response = parts[-1].strip()
                # Remove any "Assistant:" prefix if present
                if assistant_response.startswith("Assistant:"):
                    assistant_response = assistant_response.replace("Assistant:", "").strip()
                elif assistant_response.startswith("AI:"):
                    assistant_response = assistant_response.replace("AI:", "").strip()
                parsed_response = assistant_response
        
        # Fallback: if the response still contains the full prompt, try to extract after "Assistant:" or similar
        if "System:" in parsed_response and "Human:" in parsed_response:
            # Look for common assistant indicators
            for indicator in ["Assistant:", "AI:", "\n\n"]:
                if indicator in parsed_response:
                    parts = parsed_response.split(indicator)
                    if len(parts) > 1:
                        parsed_response = parts[-1].strip()
                        break
        
        # Ensure we have a meaningful response
        if not parsed_response or len(parsed_response.strip()) == 0:
            parsed_response = "I apologize, but I couldn't generate a proper response. Please try again."

        # Update & save history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=parsed_response))
        save_history_to_txt(chat_history)
        print(f"----------------- chat history \n {chat_history}")
        print(f"----------------- final response: {parsed_response}")

        # Display response
        await cl.Message(content=parsed_response).send()
       
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        await cl.Message(content=error_msg).send()

if __name__ == "__main__":
    # Run the Chainlit app
    pass