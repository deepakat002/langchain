from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# === INIT LLM & TOOL ===
print("[INIT] Initializing LLM and DuckDuckGo tool...")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
search_tool = DuckDuckGoSearchRun()

print(f"------- tool info --------\n")
print(f"name: {search_tool.name} \n description:{search_tool.description}\n arguments:{search_tool.args}")

### tool binding
bound_llm = llm.bind_tools([search_tool])

# === TOOL CALLING EXAMPLE ===
def run_tool_call_example(question: str):
    print(f"\n[USER INPUT] {question}")
    messages = [HumanMessage(content=question)]

    print("[LLM] Sending message to LLM to decide tool call...")
    response = bound_llm.invoke(messages)
    messages.append(response)

    print(f"\n-----------------------------\n response:{response} \n -----------------------\n ")

    if response.tool_calls:
        print("[LLM] Tool call detected.")
        tool_call = response.tool_calls[0]

        print(f"[EXECUTION] Invoking tool with tool call: {tool_call}")
        tool_result = search_tool.invoke(tool_call)

        print("[LLM] Feeding tool result back into LLM...")
        messages.append(tool_result)
        print(f"[LLM]LLM call on final message \n ------------- \n {messages}\n ------------ after tool calling...")

        final = bound_llm.invoke(messages)
        return final.content
    else:
        print("[LLM] No tool needed. Returning direct answer.")
        return response.content

# === MAIN RUN ===
if __name__ == "__main__":
    question = "What is the latest news on India?"
    answer = run_tool_call_example(question)
    print("\n🧠 Final Answer:\n", answer)
