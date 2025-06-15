from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

### create open


load_dotenv()





loader = TextLoader('dogs.txt', encoding='utf-8')

docs = loader.load()

print(docs, type(docs))
print("\n\n --------------------------------------  \n\n")
print(docs[0])
print("\n\n --------------------------------------  \n\n")
print(docs[0].page_content)
print(docs[0].metadata)



########## model interaction
parser = StrOutputParser()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Find the height of each dog from - \n {dog_essay}',
    input_variables=['dog_essay']
)


chain = prompt | model | parser

print(chain.invoke({'dog_essay':docs[0].page_content}))