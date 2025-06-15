
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Cats are independent animals. 
They like fish.

Dogs are loyal companions.
They love playing fetch.

Birds can fly.
They love water"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---\n{chunk}\n")



