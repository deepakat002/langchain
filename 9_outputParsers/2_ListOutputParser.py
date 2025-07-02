"""
ListOutputParser - Parses comma-separated or newline-separated items into a Python list
Shows the difference between regular prompt and prompt with format instructions.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()

def demo_list_output_parser():
    model = ChatOpenAI()
    
    print("=" * 50)
    print("LIST OUTPUT PARSER DEMO")
    print("=" * 50)
    
    # WITHOUT PARSER - Regular prompt, returns string
    prompt_without_parser = PromptTemplate(
        template="List 5 popular {category} items.",
        input_variables=["category"]
    )
    
    chain_without_parser = prompt_without_parser | model
    result_without = chain_without_parser.invoke({"category": "programming languages"})
    
    print("BEFORE PARSER:")
    print("Prompt used:", prompt_without_parser.invoke({"category": "programming languages"}))
    print(f"Type: {type(result_without)}")
    print(f"result_without: {result_without}")
    print()
    
    # WITH PARSER - Modified prompt with format instructions
    parser = CommaSeparatedListOutputParser()
    format_instructions = parser.get_format_instructions()
    
    prompt_with_parser = PromptTemplate(
        template="List 5 popular {category} items.\n{format_instructions}",
        input_variables=["category"],
        partial_variables={"format_instructions": format_instructions}
    )

    print("=" * 50)

    chain_with_parser = prompt_with_parser | model | parser
    result_with = chain_with_parser.invoke({"category": "programming languages"})
    
    print("AFTER PARSER:")
    print("Modified prompt:", prompt_with_parser.invoke({"category": "programming languages"}))
    print(f"Type: {type(result_with)}")
    print(f"result_with: {result_with}")
    print(f"First item: '{result_with[0] if result_with else 'None'}'")
    print(f"Number of items: {len(result_with)}")
    print()
    

if __name__ == "__main__":
    demo_list_output_parser()