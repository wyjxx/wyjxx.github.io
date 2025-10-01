import os
import base64
import re
import json
import requests
import argparse
import backoff
from openai import OpenAI, APIError, RateLimitError
from google import genai
from google.genai import types
import anthropic
import time

"""
reasoning.py
------------
- Handles reasoning generation with different LLMs.
(Supported models: ChatGPT (o4-mini), Claude, Gemini, Qwen, GPT-5, OpenRouter API)
- Evaluates reasoning accuracy in Granularity Score (0-3)
- Detects reasoning patterns (BF/DF/Switch)
- Main outputs: reasoning.json, step_acc.json, pattern.json

Functions:
- reasoning functions for each model
- step_accuracy(): evaluate step-wise correctness (0-3)
- detect_pattern(): annotate BF/DF/Switch patterns
"""

# opeanai api key
client_chatgpt = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

# openrouter api key
client_openrouter = OpenAI(api_key="YOUR_API_KEY",base_url="https://openrouter.ai/api/v1")

# claude api key
client_claude = anthropic.Anthropic(
    # os.environ.get("ANTHROPIC_API_KEY")
    api_key="YOUR_API_KEY",
)

# qwen api key
client_qwen = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# gemini api key
client_gemini = genai.Client(api_key="YOUR_API_KEY")


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
    response = client_chatgpt.responses.create(
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
    
# reasoning segmentation
def split_to_paragraph_llm(text):
    prompt2 = """
    Keep the full original text unchanged. Split the text into semantic paragraphs. If the text already has titles/headings, use them for paragraph division. If not, split according to meaning. Control the length of each paragraph so it is neither too short nor too long: merge overly short ones, and split overly long ones if necessary. 
    Output the result in JSON format like this:
    [
    {
        "title": "title1",
        "content": "content balabalala"
    },
    {
        "title": "title2",
        "content": "content balabalala"
    }
    ]
    Important:
        - Strictly follow the example output format
        - Keep the full original text unchanged!
        - Please output only raw JSON. Do not use any Markdown syntax
    Your task:
    """
    total_tokens = 0
    response = client_chatgpt.responses.create(
        model = "o4-mini", 
        input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text":prompt2 + text},               
                ]
            }
        ]
    )
    total_tokens = response.usage.total_tokens
    return response.output_text, total_tokens


def extract_final_conclusion(text):
    match = re.search(r"Final Conclusion:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return None

prompt = """
    Your task is to deduce the specific location where a photo was taken, with the result refined down to a neighborhood within a city.
    A single image contains many types of detailed features, which we call clues. Clues can be categorized into the following main groups:

    1. Man-made objects: including but not limited to building types and styles, window designs, brick/tile materials and colors, contractor markings, railing types, utility pole styles, wiring layouts, road paving styles, road signs and markings, curb details, vehicle and license plate designs, signage text, business names, clothing and skin tones of people.

    2. Natural elements: including but not limited to landforms, elevation changes, rock colors, soil types, ground vegetation, tree or shrub species, distant mountain ridges or coastlines, shadow length and angle, shadow intensity, cloud height and shape, position of sunlight source.

    After identifying one or more clues, you must analyze what possible location(s) those clues suggest. For example:
    - Man-made objects may help infer the country or region.
    - Natural elements may suggest climate zone, elevation, or type of natural environment.
    - Sky and lighting features may help infer northern/southern hemisphere, latitude, or climate.

    Some clues may support a candidate location, while others may contradict it. You must stay open-minded and avoid prematurely ruling out possibilities. When clues lean in favor, you can add candidate locations. When too many clues contradict a region, you may eliminate it. When clear and specific clues appear, you can narrow down to finer levels. This stage is crucial and where you are most likely to make mistakes. Constantly ask yourself:

    “Wait! Did I exclude other regions too early? Are there nearby areas with the same clues?”  
    List possible options. Actively look for evidence supporting them. Compare them directly with your leading guess without bias.  
    How compatible is each clue with different locations? How strong is the evidence?

    Through iterative hypothesis and elimination, you must continuously narrow down the estimated location until it is refined to a neighborhood within a city.  
    Make your final decision only when you are fully confident in your conclusion.

    Your response output contains a single final location as answer, it must strictly follow the following example string format:
    "Final Conclusion: Kurfürstenstrasse, Berlin, Germany, Europe"

    Cautions:
    - The image may not come from Google Street View; many are personally taken photos.  
    - Metadata such as image names or EXIF info must not be treated as valid clues.  
    - Do not jump to a final conclusion when multiple regions are still possible.
    - Important: Response output contains only a single conclusion!!!
    
    Let's think step by step.
"""
# reasoning with chatgpt
def reasoning_chatgpt(image_path, output_dir):

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    output =""
    total_tokens = 0
    response_tokens = 0
    reasoning_tokens = 0

    t0 = time.time()
    response = client_chatgpt.responses.create(
        model = "o4-mini", 
        tools = [ { "type": "web_search_preview" ,"search_context_size": "low"} ],
        reasoning = { 
            "effort": "medium",
            "summary": "detailed" 
        },
        stream = True,
        max_output_tokens=5000,
        input = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Here is the image I want you to analyze:"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}","detail":"low"}
                ]
            }
        ]
    )

    for event in response:
        if event.type == 'response.reasoning_summary_text.done' or event.type == 'response.output_text.done':
            output += event.text
            print(event.text)
        elif event.type == "response.completed":
            response_tokens += event.response.usage.total_tokens
            total_tokens += event.response.usage.total_tokens
            reasoning_tokens += event.response.usage.output_tokens_details.reasoning_tokens
            print("Total tokens:", total_tokens)
            print("Reasoning tokens:", reasoning_tokens)
            break
    t1 = time.time()

    response_time = t1 - t0
        
    paragraph, tokens = split_to_paragraph_llm(output)
    
    paragraph = check_fix_json(paragraph)

    with open(output_dir + "reasoning.json", "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")
    print(paragraph + "\n")
    print(f"Finish reasoning! Written in {output_dir}" + "\n")
    
    total_tokens += tokens
    return total_tokens, response_tokens, reasoning_tokens, response_time

# reasoning with gpt-5 using openrouter
def reasoning_gpt5(image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    reasoning_content = ""  
    answer_content = ""    
    output ="" 
    total_tokens = 0
    response_tokens = 0
    reasoning_tokens = 0

    t0 = time.time()

    completion = client_openrouter.chat.completions.create(
        model="openai/gpt-5",  
        messages=[
            {
                "role": "system", 
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "Here is the image I want you to analyze:"},
                ],
            },
        ],
        stream_options={
            "include_usage": True
        },
        extra_body={"enable_thinking": True,"enable_search": True}
    )
    t1 = time.time()
    response_time = t1 - t0
   
    reasoning_content = completion.choices[0].message.reasoning
    answer_content = completion.choices[0].message.content
    output = reasoning_content + "\n" + answer_content
    
    response_tokens += completion.usage.total_tokens
    total_tokens += response_tokens
    reasoning_tokens += completion.usage.completion_tokens_details.reasoning_tokens

    paragraph, tokens = split_to_paragraph_llm(output)
    
    paragraph = check_fix_json(paragraph)

    with open(output_dir + "reasoning.json", "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")
    print(paragraph + "\n")
    print(f"Finish reasoning! Written in {output_dir}" + "\n")
    
    total_tokens += tokens
    return total_tokens, response_tokens, reasoning_tokens, response_time

# reasoning with gemini using openrouter
def reasoning_gemini(image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    reasoning_content = ""  
    answer_content = ""    
    output ="" 
    total_tokens = 0
    response_tokens = 0
    reasoning_tokens = 0

    t0 = time.time()

    completion = client_openrouter.chat.completions.create(
        model="google/gemini-2.5-pro",  # google/gemini-2.5-pro 
        messages=[
            {
                "role": "system", 
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "Here is the image I want you to analyze:"},
                ],
            },
        ],
        stream_options={
            "include_usage": True
        },
        extra_body={"enable_thinking": True,"enable_search": True}
    )
    t1 = time.time()
    response_time = t1 - t0
   
    reasoning_content = completion.choices[0].message.reasoning
    answer_content = completion.choices[0].message.content
    output = reasoning_content + "\n" + answer_content
    
    response_tokens += completion.usage.total_tokens
    total_tokens += response_tokens
    reasoning_tokens += completion.usage.completion_tokens_details.reasoning_tokens

    paragraph, tokens = split_to_paragraph_llm(output)
    
    paragraph = check_fix_json(paragraph)

    with open(output_dir + "reasoning.json", "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")
    print(paragraph + "\n")
    print(f"Finish reasoning! Written in {output_dir}" + "\n")
    
    total_tokens += tokens
    return total_tokens, response_tokens, reasoning_tokens, response_time

# reasoning with claude
def reasoning_claude(image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    output =""
    total_tokens = 0
    response_tokens = 0
    reasoning_tokens = 0

    t0 = time.time()

    response = client_claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"{prompt}Here is the image I want you to analyze:"
                }
            ],
        }
    ],
    tools=[{
        "type": "web_search_20250305",
        "name": "web_search"
    }]
    )
    t1 = time.time()
    response_time = t1 - t0

    # The response will contain summarized thinking blocks and text blocks
    for block in response.content:
        if block.type == "thinking":
            output += block.thinking
            print(f"\nThinking summary: {block.thinking}")
        elif block.type == "text":
            print(f"\nResponse: {block.text}")
            if extract_final_conclusion(block.text):
                output += extract_final_conclusion(block.text)
    
    print(response.usage.output_tokens)
    response_tokens += response.usage.output_tokens
    total_tokens += response_tokens
    
    paragraph, tokens = split_to_paragraph_llm(output)
    
    paragraph = check_fix_json(paragraph)

    with open(output_dir + "reasoning.json", "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")
    print(paragraph + "\n")
    print(f"Finish reasoning! Written in {output_dir}" + "\n")
    
    total_tokens += tokens
    return total_tokens, response_tokens, reasoning_tokens, response_time

# reasoning with qwen
def reasoning_qwen(image_path, output_dir):
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    output ="" # 最终输出
    is_answering = False   # 判断是否结束思考过程并开始回复
    total_tokens = 0
    response_tokens = 0
    reasoning_tokens = 0

    t0 = time.time()
    # 创建聊天完成请求
    completion = client_qwen.chat.completions.create(
        model="qvq-max",  # 此处以 qvq-max 为例，可按需更换模型名称
        messages=[
            {
                "role": "system", 
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "Here is the image I want you to analyze:"},
                ],
            },
        ],
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        stream_options={
            "include_usage": True
        }
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
            response_tokens += chunk.usage.total_tokens
            total_tokens += chunk.usage.total_tokens
            reasoning_tokens += chunk.usage.completion_tokens_details.reasoning_tokens
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content
    t1 = time.time()
    response_time = t1 - t0
    
    output = reasoning_content + "\n" + answer_content

    paragraph, tokens = split_to_paragraph_llm(output)
    
    paragraph = check_fix_json(paragraph)

    with open(output_dir + "reasoning.json", "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")
    print(paragraph + "\n")
    print(f"Finish reasoning! Written in {output_dir}" + "\n")
    
    total_tokens += tokens
    return total_tokens, response_tokens, reasoning_tokens, response_time

# reasoning with gemini using google genai
def reasoning_gemini_genai(image_path, output_dir):
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    output = ""
    total_tokens = 0
    response_tokens = 0
    reasoning_tokens = 0

    # Define the grounding tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    t0 = time.time()
    response = client_gemini.models.generate_content(
        model='gemini-2.5-pro',
        contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png',
        ),
        prompt
        ],
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            ),
            tools=[grounding_tool]
        )
    )

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            print("Thought summary:")
            print(part.text)
            print()
            output += part.text

        else:
            print("Answer:")
            print(part.text)
            print()
            output += part.text
    t1 = time.time()

    response_time = t1 - t0

    response_tokens = response.usage_metadata.total_token_count
    total_tokens += response.usage_metadata.total_token_count
    reasoning_tokens += response.usage_metadata.thoughts_token_count
    
    paragraph, tokens = split_to_paragraph_llm(output)
    
    paragraph = check_fix_json(paragraph)

    with open(output_dir + "reasoning.json", "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")
    print(paragraph + "\n")
    print(f"Finish reasoning! Written in {output_dir}" + "\n")
    
    total_tokens += tokens
    return total_tokens, response_tokens, reasoning_tokens, response_time


# check the accuracy/correctness of each step in Granualrity Score
def step_accuracy(reasoning_path, ground_truth_path, pic, output_dir):

    with open(reasoning_path, 'r', encoding='utf-8') as f:
        reasoning = json.load(f)
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gps_json = json.load(f)
    country = gps_json[pic]['COUNTRY']
    city = gps_json[pic]['CITY']
    street = gps_json[pic]['STREET']
    ground_truth = f'country: {country}, city: {city}, street: {street}'

    prompt_stepAcc = """
    You are an helpful assistant to evaluate the accuracy of location conclusions.
    Given a text, for each paragraph, compare the hypothesis or conclusion by the end of it with the ground truth location, and rate accuracy: 
    0: No clear hypothesis or conclusion, or completely wrong at all levels. 
    1: Correct at country level.
    2: Correct at city level.
    3: Correct at street/neighborhood level.
    
    Output strictly in the following JSON format: [{"step":1,"location":"Hypothesis or Conclusion location","accuracy":0},{"step":2, "location":"Hypothesis or Conclusion location","accuracy": 1}]
    Important: 
        - Strictly follow the output format 
        - Please output only raw JSON. Do not use any Markdown syntax
    """
    response = client_chatgpt.responses.create(
        model = "o4-mini", 
        input = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text":prompt_stepAcc},               
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text":f"Here is the reasoning text:{reasoning}, Here is the ground truth location:{ground_truth}"},               
                ]
            }
        ]
    )
    output = check_fix_json(response.output_text)
    step_acc = json.loads(output)
    # last step accuracy
    final_acc = step_acc[-1]['accuracy']

    with open(output_dir + "step_acc.json", "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"Accuracy Rating for {pic}: {output}")
    
    return final_acc, response.usage.total_tokens

# detect the pattern of each step
def detect_pattern(reasoning_path, ground_truth_path, pic, output_dir):

    with open(reasoning_path, 'r', encoding='utf-8') as f:
        reasoning = json.load(f)
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gps_json = json.load(f)
    country = gps_json[pic]['COUNTRY']
    city = gps_json[pic]['CITY']
    street = gps_json[pic]['STREET']
    ground_truth = f'country: {country}, city: {city}, street: {street}'

    prompt_pattern = """
    You are a research expert specializing in analyzing LLM reasoning processes. Your task is to annotate and analyze LLM geolocation reasoning trajectories based on a defined theoretical framework.

    1. You will analyze a series of reasoning steps. After careful analysis, classify each paragraph according to its main content into one of the following two patterns:

    a) **Breadth-First**:
    This is a divergent exploration mode, which can appear at different stages of reasoning:

    * *Clue collection*: When there is no specific hypothesis, in order to gather information, the model broadly and descriptively observes the image, lists clues, and obtains an overall impression.
    * *Hypothesis generation*: When key clues are lacking, it jumpingly mentions multiple hypothetical locations, briefly recalls their features, and performs quick, shallow comparisons for verification.
    * *Reference keywords*: consider, observe, try, wonder, notice, might, could

    b) **Depth-First**:
    This is a convergent verification analysis mode, which can also appear at different stages of reasoning:

    * *Clue collection*: When there is a specific hypothesis, in order to verify it, the model observes the image specifically and purposefully, searching for particular detail clues.
    * *Hypothesis generation*: Based on key clues, the model derives a hypothesis, carefully recalls location features or searches online for more information, and after comparison, either confirms or eliminates that location hypothesis.
    * *Reference keywords*: focus, detail, lead to, indicate, hint, confirm, specific, narrow down, point

    2. In addition, you also need to annotate all paragraphs where a **switch** occurs. Please refer to the following definition:

    c) **Breadth-Depth Switch**:
    This is the node where the reasoning mode changes, usually occurring in the following scenarios:
    i) *Switch to breadth mode*:
    * When high-confidence clues are lacking, turning to divergent exploration
    * After verification confirms the current granularity hypothesis, shifting to explore finer-grained locations
    * After verification encounters contradictions, eliminating the current hypothesis and turning to search for another
    ii) *Switch to depth mode*:
    * Attention in breadth mode gradually converges on local key clues, or a specific hypothesis is proposed, thereby turning to the verification analysis of depth mode

    3. Finally, briefly explain your annotation choices.

    Please refer to some examples:

    a) **Breadth-First**:
    i) The user is interested in pinpointing a neighborhood from a street photo. The image features European-style houses with red-brown tiled roofs and white plaster walls, typical of southern Germany or Austria. The architecture suggests it could be Bavaria, with steep gabled roofs and pastel-color facades. Notably, there’s a tree and a parked Mercedes with a likely German license plate. There are distinct characteristics like wooden picket fences and horizontal slats on windows, enhancing the local vibe.
    ii) I'm considering options like Eichenau or Puchheim near Munich, but Puchheim has more modern buildings. Germering seems too large, so I’m thinking smaller suburbs like Gröbenzell or Karlsfeld, yet they appear standard. Could it be in southwestern Germany, such as towns in Baden-Württemberg? That would align with certain architectural styles. I see ivy and dormer windows on houses, which further narrows it down. There’s a faint sign with "Gaststube," suggesting a tavern. It might be in a small village like Emmering, in Fürstenfeldbruck, but I’m unsure. The blue recycling bins suggest a local standard, likely confirming it’s in Munich. The license plate indicating "M" supports this too. My guess could point towards Untermenzing or Obermenzing, particularly along smaller streets.

    b) **Depth-First**:
    i) I’m analyzing the area of Dorfstrasse in Obermenzing, particularly noting traditional buildings like the Gasthof zum Alten Wirt. The photo shows small houses, yet not the larger inn. The nearby area features St. Georg church, but no church tower is visible. I believe the tall poplar tree behind may indicate farmland. It seems this isn’t Obermenzing but Untermenzing, where Eversbuschstrasse has preserved gable-style houses. I'll look up images for Eversbuschstrasse in Untermenzing to confirm details.
    ii) The wiki mentions that typical gable-facing buildings line Eversbuschstraße, but our photo seems to show a narrow side street instead. In the image, gables face us, indicating the street runs perpendicular to the ridge. Eversbuschstraße does have gable-facing houses, though. I wonder if we need specific house numbers like 62 or 54 since they’re listed as former farmhouses. Yet, the modern look might complicate identification. It’s possible this is a different area, maybe in Bavaria, but not in Munich.

    c) **Switches**:

    * *Switch to Breadth*: It’s possible this is a different area, maybe in Bavaria, but not in Munich. I’m considering if this photo could be from villages near Ingolstadt.
    * *Switch to Depth*: The blue recycling bins suggest a local standard, likely confirming it’s in Munich. The license plate indicating "M" supports this too. My guess could point towards Untermenzing or Obermenzing, particularly along smaller streets.

    ---

    **Your output format must be JSON and strictly follow the form below:**
    {
    "Breadth-First":[{"Step":1,"Explanation":"Explain shortly your decision here"},{"Step":3,"Explanation":"Explain shortly your decision here"}],
    "Depth-First":[{"Step":2,"Explanation":"Explain shortly your decision here","KeyElement":"List key hypothesis or clue to be verified or analyzed"}],
    "Breadth-Depth Switch":[{"FromStep":1,"ToStep":2,"SwitchType":"ToDepth","Explanation":"Explain shortly your decision here"},{"FromStep":2,"ToStep":3,"SwitchType":"ToBreadth","Explanation":"Explain shortly your decision here"}]
    }
    Important:

    * Do not omit anything
    * Strictly follow JSON format, do not output in markdown format

    """
    response = client_chatgpt.responses.create(
        model = "o4-mini",
        reasoning = { 
            "effort": "medium"
        }, 
        input = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": prompt_pattern},               
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Your task is to analyze the text:{reasoning}"},               
                ]
            }
        ]
    )
    output = check_fix_json(response.output_text)

    with open(output_dir + "pattern.json", "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(output + "\n")
    print(f"Finish Detecting Pattern! Written in {output_dir}" + "\n")

    return response.usage.total_tokens
    