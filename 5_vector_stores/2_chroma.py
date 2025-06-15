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
results = vectorstore.similarity_search(query, k=3)

print("\n🔍 Top 3 Relevant Documents:")
for doc in results:
    print(doc.page_content)
    print("----------- ----------------------")

# ---------------------- View Stored Data ----------------------
print("\n# ---------------------- View Stored Data ----------------------")
db_data = vectorstore.get(include=['documents', 'embeddings', 'metadatas'])

print(f"📚 Number of stored chunks: {len(db_data['documents'])}")

# Print 3 example documents (content + metadata)
print("\n📄 First 3 Documents:\n")
for i, (doc, meta, doc_id) in enumerate(zip(db_data['documents'][:3], db_data['metadatas'][:3], db_data['ids'][:3])):
    print(f"--- Document {i+1} ---")
    print(f"🆔 ID: {doc_id}")
    print("📄 Content:")
    print(doc)
    print("📝 Metadata:")
    print(meta)
    print()

# Optional: list IDs
ids = db_data['ids']
print("\n🆔 Stored Document IDs:")
print(ids[:3])

# ---------------------- Update an Existing Document ----------------------
print("\n# ---------------------- Update an Existing Document ----------------------")
# ---------------------- View Document Before Update ----------------------
print("\n# ---------------------- View Document Before Update ----------------------")
print(f"🆔 Document ID to update: {ids[0]}")
before_update = vectorstore.get(ids=[ids[0]], include=["documents", "metadatas"])
print("📄 Content Before Update:")
print(before_update['documents'][0])
print("📝 Metadata Before Update:")
print(before_update['metadatas'][0])

# ---------------------- Perform Update ----------------------
updated_doc = Document(
    page_content="XXXXX Beagles are small hound dogs with short legs, floppy ears, and a keen sense of smell.",
    metadata={"source": "manual_update"}
)
vectorstore.update_document(document_id=ids[0], document=updated_doc)
print(f"\n✅ Document {ids[0]} updated.")

# ---------------------- View Document After Update ----------------------
print("\n# ---------------------- View Document After Update ----------------------")
after_update = vectorstore.get(ids=[ids[0]], include=["documents", "metadatas"])
print("📄 Content After Update:")
print(after_update['documents'][0])
print("📝 Metadata After Update:")
print(after_update['metadatas'][0])

# ---------------------- Add a New Document ----------------------
print("\n# ---------------------- Add a New Document ----------------------")
new_doc = Document(
    page_content="Beagles are often used as detection dogs in airports due to their strong scenting ability.",
    metadata={"source": "new_doc"}
)
vectorstore.add_documents([new_doc])
print("✅ New document added.")

# ---------------------- Delete a Document ----------------------
print("\n# ---------------------- Delete a Document ----------------------")
vectorstore.delete([ids[1]])
print(f"🗑️ Document {ids[1]} deleted.")
