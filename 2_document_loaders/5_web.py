from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.flipkart.com/nothing-phone-2-dark-grey-256-gb/p/itmc1490711c3eb9?pid=MOBGZSDKU5CGE8FX&lid=LSTMOBGZSDKU5CGE8FXFDQPQN&marketplace=FLIPKART&q=nothing+phone2&store=tyy%2F4io&srno=s_1_2&otracker=search&otracker1=search&fm=organic&iid=363edc3e-5239-4d97-b199-a909f8658711.MOBGZSDKU5CGE8FX.SEARCH&ppt=hp&ppn=homepage&ssid=rhfzx0o5u80000001749313491660&qH=ed2dd322081898e8")

docs = loader.load()

print(docs) 
print(f"\ntype:{type(docs)} len:{len(docs)}")

print("\n\n -----------------  1  ---------------------  \n\n")
print(docs[0])
print("\n\n -------------------  2  ------------------  \n\n")
print(docs[0].page_content)
print(docs[0].metadata)



############ model interaction
# parser = StrOutputParser()

# model = ChatOpenAI()

# prompt = PromptTemplate(
#     template='Create a summary on - \n {texts}',
#     input_variables=['texts']
# )

# chain = prompt | model | parser

# print(chain.invoke({'texts':docs[0].page_content}))