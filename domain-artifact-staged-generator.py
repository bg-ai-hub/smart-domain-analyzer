import os
import json
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader

DDD_STAGE1_JSON_STRUCTURE = """
{{
  "Domain": "...",
  "Subdomains": [
    {{
      "name": "...",
      "Subdomains": [  // Nested subdomains allowed
        // (optional, repeat structure)
      ],
      "BoundedContexts": [
        {{
          "name": "..."
        }}
      ]
    }}
  ]
}}
"""

def load_requirements(file_path):
    """Load requirements from a text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

def get_stage1_prompt():
    """Prompt for extracting Domain, Subdomains, and Bounded Contexts."""
    return PromptTemplate(
        input_variables=["requirement_text"],
        template = f"""
From the following requirement text, extract a hierarchical structure of business subdomains and their corresponding bounded contexts in valid JSON only.

- Include only Domain, Subdomains (can be nested), and Bounded Contexts.
- Each leaf-level Subdomain must include one or more Bounded Contexts.
- Bounded Context = Microservice boundary, must reflect Ubiquitous Language.
- Each Bounded Context must have a **distinct and non-overlapping** set of relevant sentences from the requirement. Do not duplicate the same paragraph across contexts.
- Use only sentences that are **specifically and uniquely relevant** to that context. Avoid generic overlaps.
- If two contexts share functionality, assign sentences only to the one that owns the business responsibility.
- No markdown, no explanations. Only JSON.

Use this structure:
{DDD_STAGE1_JSON_STRUCTURE}

Business Requirement:
{{requirement_text}}
"""

    )

def initialize_llm(model_name="llama3.2", temperature=0, max_tokens=2000):
    """Initialize the LLM."""
    return ChatOllama(model=model_name, temperature=temperature, max_tokens=max_tokens)

def extract_stage1_artifacts(requirement_text, llm_chain):
    """Extract Domain, Subdomains, and Bounded Contexts."""
    response = llm_chain.run(requirement_text=requirement_text)
    try:
        parsed = json.loads(response)
    except Exception:
        print("Warning: LLM output is not valid JSON. Raw output:")
        print(response)
        parsed = None
    return parsed

def extract_relevant_text(requirement_text, bc_name, llm):
    """
    Use LLM to extract the most relevant part of the requirement text for a given bounded context.
    """
    prompt = PromptTemplate(
        input_variables=["requirement_text", "bc_name"],
        template="""
Given the following business requirement and a bounded context name, extract the most relevant sentences or paragraphs from the requirement that describe or pertain to the bounded context "{bc_name}". 
Return only the relevant text, no explanations.

Business Requirement:
{requirement_text}
"""
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    relevant_text = llm_chain.run(requirement_text=requirement_text, bc_name=bc_name)
    return relevant_text.strip()

def build_subdomain_bc_dict(subdomains, requirement_text, llm):
    """
    Recursively build a dictionary with subdomain names as keys and their bounded contexts or nested subdomains.
    Attach relevant requirement text for each bounded context using LLM.
    """
    result = {}
    for sub in subdomains:
        name = sub.get("name")
        node = {}
        # Handle nested subdomains
        nested = sub.get("Subdomains")
        if nested:
            node["subdomains"] = build_subdomain_bc_dict(nested, requirement_text, llm)
        bcs = sub.get("BoundedContexts", [])
        if bcs:
            node["bounded_contexts"] = []
            for bc in bcs:
                bc_name = bc.get("name")
                bc_dict = {"name": bc_name}
                bc_dict["requirement_text"] = extract_relevant_text(requirement_text, bc_name, llm)
                node["bounded_contexts"].append(bc_dict)
        result[name] = node
    return result

def process_bc(bc, bc_req_text, llm):
    """
    Use LLM to identify aggregates and entities for a bounded context.
    """
    prompt = PromptTemplate(
        input_variables=["requirement_text", "bc_name"],
        template="""
Given the following business requirement and a bounded context name, identify aggregates and entities for the bounded context "{bc_name}" using Domain Driven Design principles.

Rules:
- Entities represent **database tables**. Only include entities if they represent distinct persistent data with their own identity or lifecycle.
- Ignore transient objects or value objects that do not require separate database tables.
- Aggregates are optional and should combine entities that are typically read, updated, or deleted together in a single business operation (i.e., transactionally consistent).
- An aggregate must have at least two entities.
- Output only aggregates and entities. No markdown, no explanations.

Output format:
{{
  "aggregates": {{
    "AggregateName1": ["EntityA", "EntityB"],
    "AggregateName2": ["EntityC", "EntityD"]
  }},
  "entities": ["EntityA", "EntityB", "EntityC", "EntityD", ...]
}}

Business Requirement:
{requirement_text}

"""
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    response = llm_chain.run(requirement_text=bc_req_text, bc_name=bc["name"])
    try:
        parsed = json.loads(response)
    except Exception:
        print(f"Warning: LLM output is not valid JSON for BC '{bc['name']}'. Raw output:")
        print(response)
        parsed = {"aggregates": {}, "entities": []}
    return parsed

def traverse_aggregates_entities(node, requirement_text, llm, result):
    """
    Recursively traverse the subdomain dict and process each bounded context.
    """
    # Traverse subdomains recursively
    if "subdomains" in node:
        for sub in node["subdomains"].values():
            traverse_aggregates_entities(sub, requirement_text, llm, result)
    # Process bounded contexts
    if "bounded_contexts" in node:
        for bc in node["bounded_contexts"]:
            bc_name = bc["name"]
            bc_req_text = bc.get("requirement_text", requirement_text)
            result[bc_name] = process_bc(bc, bc_req_text, llm)

def identify_aggregates_and_entities(subdomain_dict, requirement_text, llm):
    """
    For every bounded context in subdomain_dict, use LLM to identify aggregates and entities.
    Returns a dictionary: { bounded_context_name: { "aggregates": {aggregate_name: [entities]}, "entities": [entity_names] } }
    """
    result = {}
    for sub in subdomain_dict.values():
        traverse_aggregates_entities(sub, requirement_text, llm, result)
    return result

def deduplicate_entities_with_llm(aggregates_entities, subdomain_dict, llm):
    """
    Use LLM to resolve duplicate entities across bounded contexts.
    For each duplicate entity, use the requirement_text of each bounded context to decide which context should own the entity.
    Returns a new dictionary with duplicates removed and entities assigned to the most relevant context.
    """
    # Build a mapping: entity_name -> list of (bc_name, requirement_text)
    entity_map = {}
    bc_req_map = {}

    # Build bc_req_map for quick lookup
    def collect_bc_req(node):
        if "bounded_contexts" in node:
            for bc in node["bounded_contexts"]:
                bc_req_map[bc["name"]] = bc.get("requirement_text", "")
        if "subdomains" in node:
            for sub in node["subdomains"].values():
                collect_bc_req(sub)
    for sub in subdomain_dict.values():
        collect_bc_req(sub)

    # Collect all entities and their bounded contexts
    for bc_name, data in aggregates_entities.items():
        for entity in data.get("entities", []):
            entity_map.setdefault(entity, []).append(bc_name)

    # For each entity that appears in multiple contexts, use LLM to decide the best context
    entity_owner = {}
    for entity, bc_list in entity_map.items():
        if len(bc_list) == 1:
            entity_owner[entity] = bc_list[0]
        else:
            # Use LLM to decide
            prompt = PromptTemplate(
                input_variables=["entity", "bc_contexts"],
                template="""
You are given an entity "{entity}" that appears in multiple bounded contexts. 
For each context, you are given the requirement text relevant to that context.

Decide which bounded context is the most appropriate owner for the entity "{entity}" based on the requirement text. 
Return only the name of the most relevant bounded context. No explanations.

Contexts:
{bc_contexts}
"""
            )
            # Prepare context text for LLM
            bc_contexts = ""
            for bc in bc_list:
                bc_contexts += f"\nBounded Context: {bc}\nRequirement Text: {bc_req_map.get(bc, '')}\n"
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
            owner = llm_chain.run(entity=entity, bc_contexts=bc_contexts).strip()
            # Fallback if LLM output is not in bc_list
            if owner not in bc_list:
                owner = bc_list[0]
            entity_owner[entity] = owner

    # Build new aggregates_entities with deduplicated entities
    deduped = {}
    for bc_name, data in aggregates_entities.items():
        deduped[bc_name] = {
            "aggregates": {},
            "entities": []
        }
        # Entities: only include if this context is the owner
        deduped[bc_name]["entities"] = [
            entity for entity in data.get("entities", []) if entity_owner.get(entity) == bc_name
        ]
        # Aggregates: filter entities in aggregates as well
        for agg, entities in data.get("aggregates", {}).items():
            filtered_entities = [e for e in entities if entity_owner.get(e) == bc_name]
            if len(filtered_entities) >= 2:
                deduped[bc_name]["aggregates"][agg] = filtered_entities

    return deduped

def remove_empty_bounded_contexts(deduped_entities):
    """
    Remove bounded contexts that have both empty aggregates and entities.
    Also, remove entity names from the entities array if they are present in any aggregate array for that context.
    Returns a new dictionary with only non-empty bounded contexts and cleaned entities lists.
    """
    cleaned = {}
    for bc_name, data in deduped_entities.items():
        # Collect all entities that are part of aggregates
        aggregate_entities = set()
        for entities in data.get("aggregates", {}).values():
            aggregate_entities.update(entities)
        # Remove aggregate entities from the entities list
        cleaned_entities = [e for e in data.get("entities", []) if e not in aggregate_entities]
        # Only keep if at least one entity or aggregate is present (non-empty)
        if cleaned_entities or (data.get("aggregates") and len(data.get("aggregates")) > 0):
            cleaned[bc_name] = {
                "aggregates": data.get("aggregates", {}),
                "entities": cleaned_entities
            }
    return cleaned

def main():
    input_file = "ddd/input/ddd_requirement.txt"
    print("[INFO] Loading requirements...")
    requirement_text = load_requirements(input_file)
    print("[INFO] Initializing LLM and prompt...")
    prompt = get_stage1_prompt()
    llm = initialize_llm()
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    print("[INFO] Extracting DDD artifacts (Domain, Subdomains, Bounded Contexts)...")
    result = extract_stage1_artifacts(requirement_text, llm_chain)

    # Build and print the subdomain-bounded context dictionary with relevant text using LLM
    subdomain_dict = build_subdomain_bc_dict(result.get("Subdomains", []), requirement_text, llm)
    #print("[INFO] Subdomain-BoundedContext dictionary with relevant text:")
    #print(json.dumps(subdomain_dict, indent=2))

    print("[INFO] Identifying aggregates and entities for each bounded context...")
    aggregates_entities = identify_aggregates_and_entities(subdomain_dict, requirement_text, llm)
    #print("[INFO] Aggregates and Entities by Bounded Context:")
    #print(json.dumps(aggregates_entities, indent=2))

    print("[INFO] Deduplicating entities across bounded contexts...")
    deduped_entities = deduplicate_entities_with_llm(aggregates_entities, subdomain_dict, llm)
    #print("[INFO] Aggregates and Entities after deduplication:")
    #print(json.dumps(deduped_entities, indent=2))

    print("[INFO] Removing empty bounded contexts...")
    final_result = remove_empty_bounded_contexts(deduped_entities)
    print("[INFO] Final Aggregates and Entities:")
    print(json.dumps(final_result, indent=2))

if __name__ == "__main__":
    main()