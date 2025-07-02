from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedder = OpenAIEmbeddings(model="text-embedding-3-large")
vector = embedder.embed_query("What is LangChain?")
print(vector[:5])  # print first 5 dimensions
