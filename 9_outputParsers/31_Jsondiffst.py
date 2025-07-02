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

   
    
    # WITH PARSER - Modified prompt with format instructions
    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()


    
    prompt_with_parser = PromptTemplate(
        template="""Analyze the movie {movie_title} and provide details about key themes.
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

    

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define the specific schema for the "flat" themes
# You might want to define a maximum number of themes, e.g., 5
response_schemas_themes = [
    ResponseSchema(name="movie_title", description="The title of the movie"),
    ResponseSchema(name="theme_1", description="Description of the first key theme of the movie"),
    ResponseSchema(name="theme_2", description="Description of the second key theme of the movie"),
    ResponseSchema(name="theme_3", description="Description of the third key theme of the movie"),
    ResponseSchema(name="theme_4", description="Description of the fourth key theme of the movie"),
    ResponseSchema(name="theme_5", description="Description of the fifth key theme of the movie")
]

def demo_structured_output_parser_themes():
    model = ChatOpenAI(temperature=0) # Use temperature=0 for more consistent results

    print("\n" + "=" * 50)
    print("STRUCTUREDOUTPUTPARSER FOR FLAT THEMES DEMO")
    print("=" * 50)

    parser_structured = StructuredOutputParser.from_response_schemas(response_schemas_themes)
    format_instructions_structured = parser_structured.get_format_instructions()

    prompt_structured = PromptTemplate(
        template="""Analyze the movie {movie_title} and provide its 5 most important key themes.
        {format_instructions}""",
        input_variables=["movie_title"],
        partial_variables={"format_instructions": format_instructions_structured}
    )

    chain_structured = prompt_structured | model | parser_structured
    result_structured = chain_structured.invoke({"movie_title": "The Matrix"})

    print("\nAFTER STRUCTUREDOUTPUTPARSER (for flat themes):")
    print("Modified prompt:", prompt_structured.invoke({"movie_title": "The Matrix"}))
    print(f"Type: {type(result_structured)}")
    print(f"Result:\n{result_structured}")
    print("\nAccessing specific themes:")
    print(f"Movie Title: {result_structured.get('movie_title', 'N/A')}")
    print(f"Theme 1: {result_structured.get('theme_1', 'N/A')}")
    print(f"Theme 2: {result_structured.get('theme_2', 'N/A')}")
    # You can now reliably access theme_3, theme_4, theme_5 directly
    print("-" * 50)

if __name__ == "__main__":
    # Call the new demo function
    demo_json_output_parser() # Keep the original demo for comparison
    demo_structured_output_parser_themes()