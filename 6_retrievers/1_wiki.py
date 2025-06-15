from langchain.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")
docs = retriever.get_relevant_documents("What is a Beagle?")

for i, doc in enumerate(docs):
    print(f"\n-------------- Result {i+1} -------------------\n")
    print(doc.page_content)
