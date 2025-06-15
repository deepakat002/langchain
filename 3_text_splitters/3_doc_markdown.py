from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
# Golden Retriever Guide

A well-documented guide dedicated to one of the most beloved dog breeds — the Golden Retriever. Learn about their background, appearance, and origin in a structured format.

## History

Golden Retrievers were first bred in the mid-1800s in Scotland by Lord Tweedmouth. He aimed to create the perfect hunting companion — one that was obedient, gentle, and excellent at retrieving game from both land and water. Over time, the breed gained popularity not just for hunting, but also for their affectionate nature and intelligence.

## Appearance

Golden Retrievers are medium-to-large dogs with a muscular, athletic build. They have a dense, water-resistant double coat that ranges in color from light cream to deep golden. Their expressive brown eyes, floppy ears, and signature wagging tails make them instantly lovable. Adult males typically weigh between 65–75 pounds, while females range from 55–65 pounds.

## Origin

The breed originated in the Scottish Highlands. Golden Retrievers were developed by crossing a Yellow Retriever with the now-extinct Tweed Water Spaniel, and later incorporating Bloodhound and Irish Setter traits. Their versatility made them ideal for retrieving birds during hunts, and their temperament later made them ideal family pets and service dogs worldwide.

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=491,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)
print(chunks)

print("\n\n -----------------  1  ---------------------  \n\n")
print(len(chunks))
print(chunks[0])