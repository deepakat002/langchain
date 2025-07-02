# custom_output_parser.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatOpenAI(temperature=0)

# Define your custom output parser
class UpperCaseOutputParser(BaseOutputParser[str]):
    """A custom output parser that converts the LLM output to uppercase."""

    def parse(self, text: str) -> str:
        """Parses the string output of an LLM into uppercase.

        Args:
            text: The string output of the LLM.

        Returns:
            The uppercase version of the input text.
        """
        return text.upper()

    @property
    def _type(self) -> str:
        return "uppercase_parser"

print("\n--- CustomOutputParser Example ---")

prompt = PromptTemplate(
    template="Say hello to {name}.",
    input_variables=["name"]
)

# BEFORE PARSER: Raw LLM output
chain1 = prompt | model
raw_output = chain1.invoke({"name": "Alice"})
print("\nBEFORE PARSER (Type:", type(raw_output), "):")
print(raw_output.content)

# AFTER PARSER: Custom UpperCaseOutputParser
parser = UpperCaseOutputParser()
chain2 = prompt | model | parser
parsed_output = chain2.invoke({"name": "Alice"})
print("\nAFTER PARSER (Type:", type(parsed_output), "):")
print(parsed_output)