from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='multi_pdf',
    glob='*.pdf',
    loader_cls=PyPDFLoader,
    show_progress=True
)

docs = loader.load()

print(docs) 
print(f"\ntype:{type(docs)} len:{len(docs)}")

print("\n\n -----------------  1  ---------------------  \n\n")
print(docs[0])
print("\n\n -------------------  2  ------------------  \n\n")
print(docs[0].page_content)
print(docs[0].metadata)

print(docs[15].page_content)
print(docs[15].metadata)

#### ------------- lazy load ------------
'''
In LangChain, lazy_load=True in DirectoryLoader is used to load documents lazily — 
meaning one at a time, only when needed, instead of loading all files into memory at once.
'''

# ### with load()
# for document in docs:
#     print(document.metadata)

# ### with lazyload()
# docs = loader.lazy_load()

# for document in docs:
#     print(document.metadata)





############ model interaction
# parser = StrOutputParser()

# model = ChatOpenAI()

# prompt = PromptTemplate(
#     template='create a 50 words summary on- \n {dog_essay}',
#     input_variables=['dog_essay']
# )


# chain = prompt | model | parser

# print(chain.invoke({'dog_essay':docs[0].page_content}))