from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="""
You are a dog expert. Please provide information about {dog} in {nwords}. 
Also, give information about their {attribute}
""",
input_variables=["dog","nwords","attribute"],
validate_template=True
)

template.save('template.json')