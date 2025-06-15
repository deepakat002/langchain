from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter



loader = TextLoader('../data_files/dogs.txt', encoding='utf-8')

docs = loader.load()

print(docs, type(docs))
print("\n\n ---------------- 1 with 0 overlap ----------------------  \n\n")

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result)

print("\n\n ---------------- 2 with overlap ----------------------  \n\n")
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separator=''
)

result = splitter.split_documents(docs)

print(result)



