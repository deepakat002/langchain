from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()
model = ChatOpenAI()


template = load_prompt('template.json')

chain = template | model
result = chain.invoke({
    'dog':"pitbull",
    'nwords':50,
    'attribute':"color"
})

print(result.content)