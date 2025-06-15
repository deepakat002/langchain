from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = '''
Dogs are loyal and friendly animals. They love to play and interact with humans. Cats, on the other hand, are more independent and curious.


Tigers are powerful wild cats known for their strength and striped fur. Elephants are the largest land animals and are famous for their intelligence and long trunks. 

'''

# Perform the split

chunks = splitter.create_documents([text])

print(chunks)

print("\n\n -----------------  1  ---------------------  \n\n")
print(len(chunks))
print(chunks[0])

