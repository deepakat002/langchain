from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ---------------------- Load PDF ----------------------
print("\n# ---------------------- Load PDF ----------------------")
loader = PyPDFLoader('../data_files/beagle.pdf')
documents = loader.load()

# ---------------------- Split into Chunks ----------------------
print("\n# ---------------------- Split into Chunks ----------------------")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ---------------------- Create Vector Store (Chroma) ----------------------
print("\n# ---------------------- Create Vector Store (Chroma) ----------------------")
embeddings = OpenAIEmbeddings()
persist_directory = './chroma_db'

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Persist to disk
vectorstore.persist()

# ---------------------- Query and Retrieve Documents ----------------------
print("\n# ---------------------- Query and Retrieve Documents ----------------------")
query = "what is the appearance of beagle?"
# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "lambda_mult": 1}
)
results = retriever.invoke(query)

print("\n🔍 Top 3 Relevant Documents:")
for doc in results:
    print(doc.page_content)
    print("----------- ----------------------")

print("\n# ----------------------  lambda = 0.1----------------------")

# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "lambda_mult": 0.1}
)
results = retriever.invoke(query)

print("\n🔍 Top 3 Relevant Documents:")
for doc in results:
    print(doc.page_content)
    print("----------- ----------------------")
