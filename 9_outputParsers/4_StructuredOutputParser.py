"""
StructuredOutputParser - Expects JSON format and parses it into a dictionary
Shows how response schemas define the expected structure and format instructions.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

def demo_structured_output_parser():
    model = ChatOpenAI()
    
    print("=" * 50)
    print("STRUCTURED OUTPUT PARSER DEMO")
    print("=" * 50)
    
    # WITHOUT PARSER - Regular prompt, returns unstructured string
    prompt_without_parser = PromptTemplate(
        template="Create a product description for a {product_type}.",
        input_variables=["product_type"]
    )
    
    chain_without_parser = prompt_without_parser | model
    result_without = chain_without_parser.invoke({"product_type": "smartphone"})
    
    print("BEFORE PARSER:")
    print("Original prompt:", prompt_without_parser.invoke({"product_type": "smartphone"}))
    print(f"Type: {type(result_without.content)}")
    print(f"Content: {result_without.content}")
    print("To get structured data, need manual parsing and field extraction")
    print()
    
    # WITH PARSER - Define response schema and get format instructions
    response_schemas = [
        ResponseSchema(name="product_name", description="Name of the product"),
        ResponseSchema(name="price", description="Price of the product in USD"),
        ResponseSchema(name="category", description="Product category"),
        ResponseSchema(name="rating", description="Product rating out of 5 stars"),
        ResponseSchema(name="features", description="List of key product features")
    ]
    
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    print("Response schemas defined:")
    for schema in response_schemas:
        print(f"  - {schema.name}: {schema.description}")
    print()


    
    prompt_with_parser = PromptTemplate(
        template="Create a product description for a {product_type}.\n{format_instructions}",
        input_variables=["product_type"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    chain_with_parser = prompt_with_parser | model | parser
    result_with = chain_with_parser.invoke({"product_type": "smartphone"})
    
    print("=" * 50)

    print("AFTER PARSER:")
    print("Modified prompt",prompt_with_parser.invoke({"product_type": "smartphone"}))
    print(f"Type: {type(result_with)}")
    print(f"result: {result_with}")
    print(f"Product Name: {result_with.get('product_name', 'N/A')}")
    print(f"Price: {result_with.get('price', 'N/A')}")
    print(f"Category: {result_with.get('category', 'N/A')}")
    print(f"Rating: {result_with.get('rating', 'N/A')}")
    print(f"Features: {result_with.get('features', 'N/A')}")
    print()


if __name__ == "__main__":
    demo_structured_output_parser()