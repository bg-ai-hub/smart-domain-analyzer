import os
import json
import warnings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Define the global JSON structure template
DDD_JSON_STRUCTURE = """
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

def load_requirements(file_path):
    print("[INFO] Calling load_requirements")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print("DDD Requirements Loaded Successfully")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    print("[INFO] Calling split_documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")
    return chunks
'''
def get_prompt_template():
    print("[INFO] Calling get_prompt_template")
    return PromptTemplate(
        input_variables=["requirement_text"],
        template=f"""
From the following business requirement, extract the Domain-Driven Design (DDD) structure in **valid JSON format only**.

**DDD Rules:**
- Every Domain can have multiple Subdomains.
- A Subdomain can have further nested Subdomains (i.e., subdomains can be hierarchical).
- Only the leaf-level Subdomains (those without further subdomains) will have Bounded Contexts.
- A leaf-level Subdomain can have multiple Bounded Contexts.
- Each Bounded Context represents the boundary of a microservice.
- For every Entity, include the following attributes (in addition to any domain-specific attributes):
    - First, generate at least 8 business-relevant attributes based on the business context.
    - After the business attributes, always append these default attributes at the end of the attribute list:
        - id (should be created by appending the entity name with 'id', e.g., 'Orderid')
        - created_by
        - creation_date
        - last_modified_by
        - last_modification_date
        - is_active
        - tenant_id
        - status

- Do NOT include explanations, markdown, or natural language descriptions outside of the JSON.
- Aggregate is non-mandatory, but if it is present, it should be included.
- For each aggregate, include at least 2 relevant entities if available in the requirement text or can be reasonably inferred.
- For each entity, include at least 8 relevant business attributes (before the default attributes) if available in the requirement text or can be reasonably inferred.
- Use the following JSON structure:
- The JSON should be **valid** and **well-structured**.

Business Requirement:
{{requirement_text}}

Return a JSON object in this format:

{DDD_JSON_STRUCTURE}
"""
    )
'''
def get_prompt_template():
    print("[INFO] Calling get_prompt_template")
    return PromptTemplate(
        input_variables=["requirement_text"],
        template=f"""
Extract the Domain-Driven Design (DDD) structure from the following business requirement as valid JSON only.

**Rules:**
- Domains can have multiple Subdomains, which may be nested. Only leaf Subdomains have Bounded Contexts.
- Each leaf Subdomain can have multiple Bounded Contexts (microservice boundaries).
- For every Entity, generate at least 8 business-relevant attributes based on the context
- No explanations or markdown, only JSON.
- Use this structure:

{DDD_JSON_STRUCTURE}

Business Requirement:
{{requirement_text}}
"""
    )

def initialize_llm(model_name="llama3.2", temperature=0, max_tokens=2000):
    print("[INFO] Calling initialize_llm")
    return ChatOllama(model=model_name, temperature=temperature, max_tokens=max_tokens)

def create_llm_chain(llm, prompt):
    print("[INFO] Calling create_llm_chain")
    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )

def extract_ddd_structure(chunks, llm_chain):
    print("[INFO] Calling extract_ddd_structure")
    extracted_data = []
    total = len(chunks)
    start_time = time.time()
    for i, chunk in enumerate(chunks):
        percent = int(((i + 1) / total) * 100)
        print(f"Processing chunk {i + 1}/{total} [{percent}%]")
        requirement_text = chunk.page_content
        response = llm_chain.run(requirement_text=requirement_text)
        extracted_data.append(response)
        print(f"Chunk {i + 1} processed successfully")
        print("-" * 80)
    elapsed = time.time() - start_time
    print(f"[INFO] Finished extract_ddd_structure in {elapsed:.2f} seconds")
    return extracted_data

def save_extracted_data(extracted_data, output_file_path):
    print("[INFO] Calling save_extracted_data")
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
    print(f"Extracted data saved to {output_file_path}")

def fix_and_merge_json_file(json_file_path, output_json_path, llm=None):
    print("[INFO] Calling fix_and_merge_json_file")
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    prompt = PromptTemplate(
        input_variables=["raw_json"],
        template=f"""
The following text contains multiple JSON objects concatenated together, which is not valid JSON.
Your task is to analyze ALL the content and create a single valid JSON object that follows the structure below.
The output JSON MUST strictly follow the format shown in the JSON structure, and MUST include all relevant information from the original content.

Do NOT omit any data or fields. Do NOT include explanations or markdown. Return only the corrected JSON.

JSON structure:
{DDD_JSON_STRUCTURE}

Original invalid JSON:
{{raw_json}}
"""
    )

    if llm is None:
        llm = initialize_llm()

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    fixed_json = llm_chain.run(raw_json=raw_content)

    # Use save_extracted_data to save the corrected JSON
    save_extracted_data([fixed_json], output_json_path)
    print("[INFO] Finished fix_and_merge_json_file")

def main():
    print("[INFO] Starting main")
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
    fix_and_merge_json_file(output_file_path, formatted_output_file_path, llm)
    print("[INFO] Finished main")

if __name__ == "__main__":
    main()