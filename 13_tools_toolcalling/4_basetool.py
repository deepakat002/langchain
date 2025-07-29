# custom_add_numbers_tool.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from dotenv import load_dotenv

load_dotenv()

# === INIT LLM & CUSTOM TOOL ===
print("[INIT] Initializing LLM and custom 'add_numbers' BaseTool...")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Pydantic schema for the tool's input
class AddNumbersInput(BaseModel):
    a: float = Field(description="The first number to add.")
    b: float = Field(description="The second number to add.")

# Define the tool using BaseTool
class AddNumbersTool(BaseTool):
    name: str = "add_numbers"
    description: str = "Adds two floating-point numbers together and returns the sum."

    args_schema: Type[BaseModel] = AddNumbersInput

    def _run(self, a: float, b: float) -> float:
        print(f"  [TOOL EXECUTION] Performing addition: {a} + {b}")
        return a + b

# Instantiate the tool
add_numbers_tool = AddNumbersTool()

print(f"------- Custom Tool Info --------\n")
print(f"name: {add_numbers_tool.name} \n description:{add_numbers_tool.description}\n arguments:{add_numbers_tool.args_schema.schema()}")

# === TOOL BINDING ===
bound_llm = llm.bind_tools([add_numbers_tool])

# === TOOL CALLING EXAMPLE ===
def run_tool_call_example(question: str):
    print(f"\n[USER INPUT] {question}")
    messages = [HumanMessage(content=question)]

    print("[LLM] Sending message to LLM to decide tool call...")
    response = bound_llm.invoke(messages)
    messages.append(response)

    print(f"\n-----------------------------\n LLM Response: {response} \n -----------------------\n ")

    final_answer_content = ""

    if response.tool_calls:
        print("[LLM] Tool call detected.")
        tool_call = response.tool_calls[0]

        print(f"[EXECUTION] Invoking tool '{tool_call['name']}' with args: {tool_call['args']}")
        if tool_call['name'] == add_numbers_tool.name:
            tool_result = add_numbers_tool.invoke(tool_call['args'])
            print(f"  Tool '{tool_call['name']}' executed. Output: {tool_result}")

            messages.append(ToolMessage(str(tool_result), tool_call_id=tool_call['id']))
            print(f"[LLM] Feeding tool result back into LLM for final answer...")

            final_response = bound_llm.invoke(messages)
            final_answer_content = final_response.content
        else:
            print(f"[ERROR] Unknown tool requested: {tool_call['name']}")
            final_answer_content = "Tool call failed. Unknown tool."
    else:
        print("[LLM] No tool needed. Returning direct answer.")
        final_answer_content = response.content

    return final_answer_content

# === MAIN RUN ===
if __name__ == "__main__":
    question = "What is 123 plus 456?"
    answer = run_tool_call_example(question)
    print("\n🧠 Final Answer:\n", answer)
