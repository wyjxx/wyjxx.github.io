import os
import base64
import requests
import argparse
import json
from openai import OpenAI

### NER Match: Iterate over paragraphs and match with entities list
### Input: paragraph json and entity list json
### Output: matched clues and locations with supporting clues json
"""
match.py
--------
This module matches reasoning paragraphs with extracted entities:
- Identify clue entities (visual + inference)
- Identify location entities (with status: excluded/included/concluded)
- Link locations to supporting clue entities
- Save per-paragraph match result as para_match.json

Functions:
- check_fix_json(str): validate and fix JSON format using LLM
- match(entity_path, reasoning_path, output_dir): main matching function
"""
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

# check and fix json format using llm
def check_fix_json(str):
    try:
        json.loads(str)
        return str
    except json.JSONDecodeError as e:
        print("JSON is invalid:", e)
        
    prompt = """
        Fix the following JSON string by ensuring it is properly formatted and contains valid JSON syntax. Please output only raw JSON. Do not use any Markdown syntax. Do not modify the original content.
        """
    response = client.responses.create(
        model = "o4-mini",
        reasoning = { 
            "effort": "medium"
        }, 
        input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text":prompt + str},               
                ]
            }
        ]
    )
    fixed_json = response.output_text
    print("Finish Correction: ", fixed_json)

    try:
        json.loads(fixed_json)
        print("Fixed JSON is now valid.")
        return fixed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON after fixing: {e}\nGot: {fixed_json}")

# semantic matching
def match(entity_path, reasoning_path, output_dir):
    
    with open(entity_path, 'r', encoding='utf-8') as f:
        entity_json = json.load(f)
    with open(reasoning_path, 'r', encoding='utf-8') as f:
        paragraph_json = json.load(f)

    # entity_json to text
    entity_list = json.dumps(entity_json, ensure_ascii=False, indent=2)
            

    prompt ="""
    You are responsible for handling a semantic entity matching task. Given a paragraph of text and a list of pre-defined entity terms labeled with their types (including visual elements, inference/knowledge terms, and locations), your task is to semantically match the content of the paragraph with the given entity terms and complete the following two subtasks:
    Step 1. Clue Extraction:
    List all words or phrases in the paragraph that semantically match any entity terms of type v (visual elements) or i (inference/knowledge). These matched items will serve as clues.
    Step 2. Location Mention Detection:
    List all words or phrases in the paragraph that semantically match any entity terms of type l (locations). For each matched location: 
    - Determine in what context the model mentions or considers this location and classify the location into one of the following statuses by assigning a number:
    1. Excluded: Explicitly ruled out, with contradictions pointed out, impossible 
    2. Included: Explored, recalled, relevant knowledge listed and compared, possible candidate  
    3. Concluded: Narrowed down, confirmed, specified, high likely based on evidence
    - Also, list the corresponding clue terms (from step 1 Clue Extraction, and only those) that support this judgment.
    
    Important Constraints:
    You must only use the given list of entity terms for matching. Do not add, modify, or infer terms beyond the provided list.

    Matching should allow for fuzzy semantic recognition—an exact textual match is not required as long as the semantic meaning clearly corresponds.

   Important: Do not miss any word! 
   Important: You must **copy exactly** the string from the provided "entity" field in the input list, without changing, shortening, or removing any part, including parentheses and text inside them. The match must be letter-for-letter identical.

    The output structure must strictly follow the JSON format below:
    {
        "paragraph":1,
        "clue":["clue1","clue2","clue3"],
        "loc-clue":
        [
            {
                "loc":"loc1",
                "status":1,
                "related_clue":["clue1","clue2"]
            },
            {
                "loc":"loc2",
                "status":3,
                "related_clue":["clue3"]
            }
        ]
    }
    The following is the list of entity terms to be used for matching:
    """
    tokens = 0
    para_match = '['
    for i, p in enumerate(paragraph_json):
        
        content = json.dumps(p['content'], ensure_ascii=False, indent=2)
        
        if i == 0:
            previous_response = client.responses.create(
                model = "o4-mini",
                reasoning = { 
                "effort": "medium"
                }, 
                input = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": prompt + entity_list},
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"This is paragraph {i+1} :" + content},
                        ]
                    }
                ]
            )
            tokens += previous_response.usage.total_tokens
        else: 
            response = client.responses.create(
                model = "o4-mini", 
                reasoning = { 
                    "effort": "medium"
                },
                previous_response_id = previous_response.id,
                input = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"This is paragraph {i+1} :" + content},
                        ]
                    }
                ]
            )
            previous_response = response
            tokens += response.usage.total_tokens

        match_output = previous_response.output_text
        match_output = check_fix_json(match_output)
        print(match_output)
        para_match += match_output + ','

    para_match = para_match.rstrip(',') + ']'

    with open(output_dir + "para_match.json", 'w', encoding='utf-8') as f:
        f.write(para_match)
    print(f"Finish match! Written in {output_dir}")

    return tokens