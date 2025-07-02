"""
PydanticOutputParser - Parses LLM output into a Pydantic model for strict validation
Shows how format instructions are crucial for proper parsing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Define Pydantic model
class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years", ge=0, le=150)
    occupation: str = Field(description="Person's job or profession")
    skills: List[str] = Field(description="List of person's key skills")

def demo_pydantic_output_parser():
    model = ChatOpenAI()
    
    print("=" * 50)
    print("PYDANTIC OUTPUT PARSER DEMO")
    print("=" * 50)
    
    # WITHOUT PARSER - Regular prompt, returns unstructured string
    prompt_without_parser = PromptTemplate(
        template="Create a fictional person profile for a {profession}.",
        input_variables=["profession"]
    )
    
    chain_without_parser = prompt_without_parser | model
    result_without = chain_without_parser.invoke({"profession": "data scientist"})
    
    print("BEFORE PARSER:")
    print("Prompt used:", prompt_without_parser.invoke({"profession": "data scientist"}))
    print(f"Type: {type(result_without)}")
    print(f"result_without: {result_without}")
    print()
    
    # WITH PARSER - Modified prompt with format instructions
    parser = PydanticOutputParser(pydantic_object=Person)
    format_instructions = parser.get_format_instructions()

    
    prompt_with_parser = PromptTemplate(
        template="Create a fictional person profile for a {profession}.\n{format_instructions}",
        input_variables=["profession"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    chain_with_parser = prompt_with_parser | model | parser
    result_with = chain_with_parser.invoke({"profession": "data scientist"})
    print("=" * 50)
    
    print("AFTER PARSER:")
    print("Modified prompt", prompt_with_parser.invoke({"profession": "data scientist"}))
    print(f"Type: {type(result_with)}")
    print(f"result_with: {result_with}")
    print(f"Name: {result_with.name}")
    print(f"Age: {result_with.age}")
    print(f"Occupation: {result_with.occupation}")
    print(f"Skills: {result_with.skills}")
    print()

if __name__ == "__main__":
    demo_pydantic_output_parser()