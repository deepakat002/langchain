"""
StrOutputParser - Returns the raw string content from LLM response
This parser extracts just the text content, removing any metadata or wrapper objects.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def demo_str_output_parser():
    model = ChatOpenAI()
    
    prompt = PromptTemplate(
        template="Write a brief greeting for a {animal} in a friendly tone.",
        input_variables=["animal"]
    )
    
    print("=" * 50)
    print("STR OUTPUT PARSER DEMO")
    print("=" * 50)
    
    # WITHOUT PARSER - Returns AIMessage object
    chain_without_parser = prompt | model
    result_without = chain_without_parser.invoke({"animal": "dog"})
    
    print("BEFORE PARSER (AIMessage object):")
    print(f"Type: {type(result_without)}")
    print(f"Full object: {result_without}")
    print()
    
    # WITH PARSER - Returns clean string
    parser = StrOutputParser()
    chain_with_parser = prompt | model | parser
    result_with = chain_with_parser.invoke({"animal": "dog"})
    
    print("AFTER PARSER (Clean string):")
    print(f"Type: {type(result_with)}")
    print(f"Direct string: '{result_with}'")
    print()
    


if __name__ == "__main__":
    demo_str_output_parser()