import os
import json
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("import done successfully")

#step 1: Load the text file
file_path = "ddd/input/ddd_requirement.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()
print("DDD Requirements Loaded Successfully")

# step 2: Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")

# Step 3: Define the prompt template
prompt = PromptTemplate(
    input_variables=["requirement_text"],
    template="""
From the following business requirement, extract the Domain-Driven Design (DDD) structure in **valid JSON format only**.
- Do NOT include explanations, markdown, or natural language descriptions outside of the JSON.
- Aggregate is non mandatory, but if it is present, it should be included.
- For each aggregate, include **at least 2 relevant entities** if available in the requirement text or can be reasonably inferred.
- For each entity, include **at least 8 relevant attributes** if available in the requirement text or can be reasonably inferred.
- Use the following JSON structure:
- The JSON should be **valid** and **well-structured**.

Business Requirement:
{requirement_text}

Return a JSON object in this format:

{{
  "Domain": "...",
  "Subdomains": [
    {{
      "name": "...",
      "BoundedContexts": [
        {{
          "name": "...",
          "Aggregates": [
            {{
              "name": "...",
              "Entities": [
                {{
                  "name": "...",
                  "Attributes": [
                    {{
                      "name": "...",
                      "data_type": "...",
                      "description": "..."
                    }}
                  ]
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}
"""
)


# Step 4: Initialize the LLM
llm = ChatOllama(model="llama3.2", temperature=0, max_tokens=2000)

# Step 5: Create the LLM chain
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False
)

# Step 6: Process each chunk and extract the DDD structure
extracted_data = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}")
    requirement_text = chunk.page_content
    response = llm_chain.run(requirement_text=requirement_text)
    extracted_data.append(response)
    print(f"Chunk {i + 1} processed successfully")
    print(f"Response: {response}")
    print("-" * 80)

# Step 7: Save the extracted data to a file
output_file_path = "ddd/output/ddd_extracted_structure.json"

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Validate each response as JSON before writing
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for data in extracted_data:
        try:
            parsed = json.loads(data)  # Validate it's JSON
            json.dump(parsed, output_file, indent=2)
        except json.JSONDecodeError as e:
            print("Invalid JSON. Skipping this chunk.")
            print(e)
print(f"Extracted data saved to {output_file_path}")