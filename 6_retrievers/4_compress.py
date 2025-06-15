from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ---------------------- Step 1: Sample Mixed-Content Chunks ----------------------
texts = [
    "Beagles are small hounds with tricolored coats. They were bred in England for rabbit hunting.",
    "These dogs have floppy ears and expressive eyes. Their excellent nose helped hunters track prey.",
    "The Beagle's compact build and short legs make it agile. It’s also known for being playful and vocal.",
    "Known for their loyalty, Beagles are also easily recognizable by their white-tipped tails and sturdy bodies.",
    "Though they need regular exercise, their coat pattern and droopy ears make them stand out visually."
]

documents = [Document(page_content=txt) for txt in texts]

# ---------------------- Step 2: Create Embeddings and VectorStore ----------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# ---------------------- Step 3: Create a Query ----------------------
query = "What does a Beagle look like?"

# ---------------------- Step 4: Search Without Compression ----------------------
print("\n❌ Without Compression (Raw Retrieved Chunks)\n")
raw_results = vectorstore.similarity_search(query, k=4)
for i, doc in enumerate(raw_results, 1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)
    print()

# ---------------------- Step 5: Set Up Compression Retriever ----------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
compressor = LLMChainExtractor.from_llm(llm) ### it takes an llm as compressor
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

# ---------------------- Step 6: Search With Compression ----------------------
print("\n✅ With Compression (Compressed to Appearance Info)\n")
compressed_docs = retriever.get_relevant_documents(query)
for i, doc in enumerate(compressed_docs, 1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)
    print()
