# faiss_pipeline.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Step 1: Load PDF
loader = PyPDFLoader('../data_files/beagle.pdf')
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 4: Query and retrieve relevant docs
query = "what is the appearance of beagle?"
results = vectorstore.similarity_search(query, k=3)

print("\nTop 3 Relevant Documents:")
for doc in results:
    print(doc.page_content)
    print(f"----------- ----------------------")
