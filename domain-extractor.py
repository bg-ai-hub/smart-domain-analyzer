import os
import json
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_requirements(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print("DDD Requirements Loaded Successfully")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")
    return chunks

def get_prompt_template():
    return PromptTemplate(
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

def initialize_llm(model_name="llama3.2", temperature=0, max_tokens=2000):
    return ChatOllama(model=model_name, temperature=temperature, max_tokens=max_tokens)

def create_llm_chain(llm, prompt):
    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )

def extract_ddd_structure(chunks, llm_chain):
    extracted_data = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}")
        requirement_text = chunk.page_content
        response = llm_chain.run(requirement_text=requirement_text)
        extracted_data.append(response)
        print(f"Chunk {i + 1} processed successfully")
        print(f"Response: {response}")
        print("-" * 80)
    return extracted_data

def save_extracted_data(extracted_data, output_file_path):
    # Ensure the directory exists before writing the file
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for data in extracted_data:
            try:
                parsed = json.loads(data)  # Validate it's JSON
                json.dump(parsed, output_file, indent=2)
            except json.JSONDecodeError as e:
                print("Invalid JSON. Skipping this chunk.")
                print(e)
    print(f"Extracted data saved to {output_file_path}")

def fix_and_merge_json_file(json_file_path, output_json_path, llm=None):
    """
    Reads a possibly invalid JSON file (with multiple root objects), 
    uses LLM to fix and merge into a single valid JSON, and writes the result to output_json_path.
    Ensures all information from the original file is retained in the output.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    prompt = PromptTemplate(
        input_variables=["raw_json"],
        template="""
The following text contains multiple JSON objects concatenated together, which is not valid JSON.
Your task is to merge ALL of them into a single valid JSON object, 
ensuring the result is valid JSON and **retains ALL information from the original content**.

Do NOT omit any data or fields. Combine everything into a single JSON object following the structure below.
Return only the corrected JSON, nothing else.

JSON structure:
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

Original invalid JSON:
{raw_json}
"""
    )

    if llm is None:
        llm = initialize_llm()

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    fixed_json = llm_chain.run(raw_json=raw_content)

    # Use save_extracted_data to save the corrected JSON
    save_extracted_data([fixed_json], output_json_path)

def main():
    print("import done successfully")
    input_file_path = "ddd/input/ddd_requirement.txt"
    output_file_path = "ddd/output/ddd_extracted_structure.json"
    formatted_output_file_path = "ddd/output/ddd_extracted_formatted_structure.json"

    documents = load_requirements(input_file_path)
    chunks = split_documents(documents)
    prompt = get_prompt_template()
    llm = initialize_llm()
    llm_chain = create_llm_chain(llm, prompt)
    extracted_data = extract_ddd_structure(chunks, llm_chain)
    save_extracted_data(extracted_data, output_file_path)
    fix_and_merge_json_file(output_file_path,formatted_output_file_path, llm)
if __name__ == "__main__":
    main()