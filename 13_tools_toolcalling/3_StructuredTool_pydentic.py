# custom_add_numbers_tool.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool # Import StructuredTool
from pydantic import BaseModel, Field # CORRECTED: Import BaseModel and Field directly from pydantic
from dotenv import load_dotenv

load_dotenv()

# === INIT LLM & CUSTOM TOOL ===
print("[INIT] Initializing LLM and custom 'add_numbers' tool...")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) # Lower temperature for more deterministic tool calls

# Define the Pydantic schema for the tool's input arguments
class AddNumbersInput(BaseModel):
    """Input schema for the add_numbers tool."""
    a: float = Field(description="The first floating-point number to add.")
    b: float = Field(description="The second floating-point number to add.")

# Define your custom tool using StructuredTool
# StructuredTool requires an explicit name, description, and the args_schema
def _add_numbers_func(a: float, b: float) -> float:
    """
    Adds two floating-point numbers together and returns their sum.
    Useful for basic arithmetic calculations.
    """
    print(f"  [TOOL EXECUTION] Performing addition: {a} + {b}")
    return a + b

add_numbers_tool = StructuredTool.from_function(
    func=_add_numbers_func,
    name="add_numbers",
    description="Adds two floating-point numbers together and returns their sum. Useful for basic arithmetic calculations.",
    args_schema=AddNumbersInput # Link the Pydantic schema here
)

print(f"------- Custom Tool Info --------\n")
print(f"name: {add_numbers_tool.name} \n description:{add_numbers_tool.description}\n arguments:{add_numbers_tool.args}")

### Tool Binding
# Bind the custom 'add_numbers' tool to the LLM.
# This makes the LLM aware of this tool and its capabilities.
bound_llm = llm.bind_tools([add_numbers_tool])

# === TOOL CALLING EXAMPLE ===
def run_tool_call_example(question: str):
    print(f"\n[USER INPUT] {question}")
    messages = [HumanMessage(content=question)]

    # Step 1: LLM Tool Calling Decision
    # The LLM processes the question and decides if it needs to call a tool.
    print("[LLM] Sending message to LLM to decide tool call...")
    response = bound_llm.invoke(messages)
    messages.append(response) # Add LLM's response to messages for context

    print(f"\n-----------------------------\n LLM Response: {response} \n -----------------------\n ")

    final_answer_content = ""

    if response.tool_calls:
        print("[LLM] Tool call detected.")
        # Access the tool_call dictionary from the list
        tool_call = response.tool_calls[0] # Assuming one tool call for simplicity

        # Step 2: Tool Execution
        # Manually invoke the custom tool with the arguments provided by the LLM.
        # Access name, args, and id using dictionary keys
        print(f"[EXECUTION] Invoking tool '{tool_call['name']}' with args: {tool_call['args']}")
        if tool_call['name'] == add_numbers_tool.name: # Compare with the StructuredTool's name
            # The tool_call['args'] is already a dictionary matching the function signature
            # Invoke the StructuredTool instance directly
            tool_result = add_numbers_tool.invoke(tool_call['args']) # Pass args dict
            print(f"  Tool '{tool_call['name']}' executed. Output: {tool_result}")

            # Step 3: Feed Tool Result back into LLM
            # Add the tool's output as a ToolMessage to the conversation history.
            messages.append(ToolMessage(str(tool_result), tool_call_id=tool_call['id']))
            print(f"[LLM] Feeding tool result back into LLM for final answer...")
            print(f"[LLM] Current messages for final call:\n ------------- \n {messages}\n ------------ \n")

            # Step 4: LLM Generates Final Answer
            # The LLM now uses the tool's output to formulate a human-readable answer.
            final_response = bound_llm.invoke(messages)
            final_answer_content = final_response.content
        else:
            print(f"[ERROR] LLM requested an unknown tool: {tool_call['name']}")
            final_answer_content = "I'm sorry, I tried to use a tool but encountered an issue."
    else:
        print("[LLM] No tool needed. Returning direct answer.")
        final_answer_content = response.content # LLM answered directly

    return final_answer_content

# === MAIN RUN ===
if __name__ == "__main__":
    question = "What is 123 plus 456?"
    answer = run_tool_call_example(question)
    print("\n🧠 Final Answer:\n", answer)