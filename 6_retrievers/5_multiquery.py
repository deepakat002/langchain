from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# ---------------------- Load Environment Variables ----------------------
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
query = "How does a Beagle look?"

# ---------------------- Step 4: Normal Retriever ----------------------
print("\n❌ Normal Retriever Results\n")
normal_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
normal_results = normal_retriever.get_relevant_documents(query)

for i, doc in enumerate(normal_results, 1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)
    print()

# ---------------------- Step 5: MultiQueryRetriever ----------------------
print("\n✅ MultiQueryRetriever Results\n")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

multi_results = multi_retriever.get_relevant_documents(query)

for i, doc in enumerate(multi_results, 1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)
    print()
