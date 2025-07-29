# movie_metadata_parsers_v2.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# Updated imports for StructuredOutputParser and ResponseSchema
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
# Updated import for PydanticOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def demo_movie_metadata_parsers():
    # It's good practice to set temperature to 0 for consistent structured output
    model = ChatOpenAI(temperature=0) 

    print("=" * 50)
    print("MOVIE METADATA PARSERS DEMO (Latest LangChain Imports)")
    print("StructuredOutputParser vs PydanticOutputParser")
    print("=" * 50)

    # --- Desired Schema ---
    print("Desired Output Schema:")
    print("{ 'movie': string, 'release_year': integer, 'rating': float }")
    print("-" * 50)

    # --- Scenario 1: Raw LLM Output (No Parser) ---
    print("\n--- BEFORE ANY PARSER (Raw LLM Output) ---")
    prompt_raw = PromptTemplate(
        template="Provide the movie title, release year, and rating for 'Inception'.",
        input_variables=[]
    )
    chain_raw = prompt_raw | model
    result_raw = chain_raw.invoke({})
    print(f"Type: {type(result_raw)}")
    print(f"Content:\n{result_raw.content}")
    print("Issue: Unstructured text. Cannot directly access fields programmatically.")
    print("-" * 50)


    # --- Scenario 2: StructuredOutputParser (Schema Enforcement, Limited Type Validation) ---
    print("\n--- WITH STRUCTUREDOUTPUTPARSER ---")
    response_schemas_structured = [
        ResponseSchema(name="movie", description="The title of the movie"),
        ResponseSchema(name="release_year", description="The year the movie was released (integer)"),
        ResponseSchema(name="rating", description="The movie's rating out of 10 (decimal number)"),
    ]
    parser_structured = StructuredOutputParser.from_response_schemas(response_schemas_structured)
    format_instructions_structured = parser_structured.get_format_instructions()

    prompt_structured = PromptTemplate(
        template="""Provide the movie title, release year, and rating for 'Inception'.
        {format_instructions}""",
        input_variables=[],
        partial_variables={"format_instructions": format_instructions_structured}
    )

    chain_structured = prompt_structured | model | parser_structured
    try:
        result_structured = chain_structured.invoke({})
        print(f"Type: {type(result_structured)}")
        print(f"Parsed Content: {result_structured}")
        print(f"Movie: {result_structured.get('movie', 'N/A')}")
        print(f"Release Year (Type: {type(result_structured.get('release_year'))}): {result_structured.get('release_year', 'N/A')}")
        print(f"Rating (Type: {type(result_structured.get('rating'))}): {result_structured.get('rating', 'N/A')}")
        print("Benefits: Guides LLM to produce JSON with specific keys. Easier access to fields.")
        print("Limitation: Does NOT strictly validate Python data types (e.g., if LLM puts 'two thousand ten' for year, it's passed as string if parsable as JSON string).")
    except Exception as e:
        print(f"An error occurred with StructuredOutputParser: {e}")
    print("-" * 50)


    # --- Scenario 3: PydanticOutputParser (Schema Enforcement AND Strict Type Validation) ---
    print("\n--- WITH PYDANTICOUTPUTPARSER ---")

    # Define Pydantic model for strict validation
    class MovieDetails(BaseModel):
        movie: str = Field(description="The title of the movie")
        release_year: int = Field(description="The year the movie was released")
        rating: float = Field(description="The movie's rating out of 10")

    parser_pydantic = PydanticOutputParser(pydantic_object=MovieDetails)
    format_instructions_pydantic = parser_pydantic.get_format_instructions()

    prompt_pydantic = PromptTemplate(
        template="""Provide the movie title, release year, and rating for 'Inception'.
        {format_instructions}""",
        input_variables=[],
        partial_variables={"format_instructions": format_instructions_pydantic}
    )
    

    chain_pydantic = prompt_pydantic | model | parser_pydantic
    try:
        result_pydantic = chain_pydantic.invoke({})
        print(f"Type: {type(result_pydantic)}")
        print(f"Parsed Content: {result_pydantic}")
        print(f"Movie: {result_pydantic.movie}")
        print(f"Release Year (Type: {type(result_pydantic.release_year)}): {result_pydantic.release_year}")
        print(f"Rating (Type: {type(result_pydantic.rating)}): {result_pydantic.rating}")
        print("Benefits: Enforces JSON schema AND strictly validates/coerces Python data types. Object-oriented access.")
    except ValidationError as e:
        print(f"Validation Error with PydanticOutputParser:")
        print(f"  {e.errors()}")
        print("This error occurs if the LLM's output does not strictly match the Pydantic model's types/schema (even after coercion attempts).")
    except Exception as e:
        print(f"An unexpected error occurred with PydanticOutputParser: {e}")

    print("-" * 50)

if __name__ == "__main__":
    demo_movie_metadata_parsers()