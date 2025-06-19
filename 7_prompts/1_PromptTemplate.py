from langchain_core.prompts import PromptTemplate

# Our news article content
news_article_content = """
Scientists at the Mars Rover mission announced today a significant discovery of ancient microbial life evidence. 
The latest geological survey of the Jezero Crater revealed unique mineral formations and isotopic signatures 
that strongly suggest the presence of water and biological activity billions of years ago. This finding marks 
a monumental step in the search for extraterrestrial life and could reshape our understanding of the universe.
"""

# Define the PromptTemplate
summary_prompt_template = PromptTemplate(
    template="""Please provide a concise, 50-word summary of the following news article:

Article:
{article_text}

Summary:""",
    input_variables=["article_text"]
)

# How it's used to create the final prompt string:
final_prompt_string = summary_prompt_template.invoke({"article_text": news_article_content})

print("--- Prompt created with PromptTemplate ---")
print(final_prompt_string) # .text to get the string representation
print("\nType of output:", type(final_prompt_string.text))