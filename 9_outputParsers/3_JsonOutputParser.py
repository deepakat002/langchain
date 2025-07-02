"""
JsonOutputParser - Parses LLM output into JSON/dictionary format
Shows the difference between regular text output and structured JSON parsing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import json

load_dotenv()

def demo_json_output_parser():
    model = ChatOpenAI()
    
    print("=" * 50)
    print("JSON OUTPUT PARSER DEMO")
    print("=" * 50)
    
    # WITHOUT PARSER - Regular prompt, returns unstructured string
    prompt_without_parser = PromptTemplate(
        template="Analyze the movie {movie_title} and provide details about genre, rating, and key themes.",
        input_variables=["movie_title"]
    )
    
    chain_without_parser = prompt_without_parser | model
    result_without = chain_without_parser.invoke({"movie_title": "The Matrix"})
    
    print("BEFORE PARSER:")
    print("Prompt used:", prompt_without_parser.invoke({"movie_title": "The Matrix"}))
    print(f"Type: {type(result_without.content)}")
    print(f"result: {result_without}")
    print()
    
    # WITH PARSER - Modified prompt with format instructions
    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    print("=" * 50)

    
    prompt_with_parser = PromptTemplate(
        template="""Analyze the movie {movie_title} and provide details about genre, rating, and key themes.
        {format_instructions}""",
        input_variables=["movie_title"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    chain_with_parser = prompt_with_parser | model | parser
    result_with = chain_with_parser.invoke({"movie_title": "The Matrix"})
    
    print("AFTER PARSER:")
    print("Modified prompt",prompt_with_parser.invoke({"movie_title": "The Matrix"}))
    print(f"Type: {type(result_with)}")
    print(f"result: {result_with}")
    print(f"Title: {result_with.get('title', 'N/A')}")
    print(f"Genre: {result_with.get('genre', 'N/A')}")
    print(f"Rating: {result_with.get('rating', 'N/A')}")
    print(f"Themes: {result_with.get('themes', 'N/A')}")
    print(f"Year: {result_with.get('year', 'N/A')}")
    print()
    

if __name__ == "__main__":
    demo_json_output_parser()