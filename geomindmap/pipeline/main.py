import time
import reasoning
import extract
import match
import coordinate
import os
import json
"""
main.py
-------
Pipeline controller script. Handles batch image processing for GeoMindMap.
Steps for each image:
1. Run reasoning (ChatGPT / Claude / Gemini)
2. Evaluate accuracy (correctness in Granularity Score)
3. Extract entities + vi_map + l_map
4. Match entities to paragraphs
5. Compute layout coordinates
6. Save process info (tokens, time, accuracy)

Functions:
- build_pic_list(image_folder_path): build index.json of available images
- process_single(pic, model): run pipeline on one image
- batch(pic_list, model): run pipeline on multiple images
"""
# build picture menu list and save as index.json
def build_pic_list(image_folder_path):
    all_files = os.listdir(image_folder_path)
    image_names = [file for file in all_files if file.endswith('.png')]
    with open(image_folder_path + "index.json", 'w', encoding='utf-8') as f:
        json.dump(image_names, f, indent=4, ensure_ascii=False)
    print(f"Pictures list is saved to {image_folder_path + 'index.json'}")
    return image_names

# process single image
def process_single(pic,model):
        
        pic_name = os.path.splitext(pic)[0]
        
        image_path = f"geomindmap/pictures/{pic}"
        output_dir = f"geomindmap/data/{model}/{pic_name}/"
        print(f"Image Path: {image_path}")
        print(f"Output Directory: {output_dir}")
        
        t0 = time.time()
        
        # Step 1: choose model and generate reasoning response and segmentation
        if model == "chatgpt":
            tokens_reasoning_paragraph, response_tokens, reasoning_tokens, response_time = reasoning.reasoning_chatgpt(image_path, output_dir)
        elif model == "claude":
            tokens_reasoning_paragraph, response_tokens, reasoning_tokens, response_time = reasoning.reasoning_claude(image_path, output_dir)
        elif model == "gemini":
            tokens_reasoning_paragraph, response_tokens, reasoning_tokens, response_time = reasoning.reasoning_gemini(image_path, output_dir)
        
        # Evaluate reasoning accuracy/correctness in Granularity Score
        accuracy, tokens_acc = reasoning.step_accuracy(output_dir + "reasoning.json", "geomindmap/pictures/gps.json", pic, output_dir)
        
        
        # Step 2: extract entities and build map layout info
        tokens_extract = extract.extract(image_path, output_dir + "reasoning.json", output_dir)

        # Step 3: match entities to paragraphs
        tokens_match = match.match(output_dir + "entity.json", output_dir + "reasoning.json", output_dir)
        
        # Step 4: calculate coordinates for map layout
        coordinate.calculate_coordinates(output_dir + "vi_map_info.json", output_dir, "vi")
        coordinate.calculate_coordinates(output_dir + "l_map_info.json", output_dir, "l")
        t4 = time.time()
        
        # Print time and token usage
        
        t_total = t4 - t0
        print(f"total time: {t_total:.2f} s")

        tokens_total = tokens_reasoning_paragraph + tokens_extract + tokens_match
        print(f"Total tokens used: {tokens_total}")
        
        print(f"Finished! {pic_name} is done.")

        return pic_name, tokens_total, t_total, reasoning_tokens, response_tokens, response_time, accuracy

# process a batch of images and save process info
def batch(pic_list, model):

    print('Hello')
    results = []

    # save info to a json file
    first_name = os.path.splitext(pic_list[0])[0]
    last_name = os.path.splitext(pic_list[-1])[0]
    out_file = f"geomindmap/data/{model}/info/{first_name}_to_{last_name}.json"

    # process each picture and collect token usage and time info
    for pic in pic_list:
        pic_name, tokens_total, t_total, reasoning_tokens, response_tokens, response_time, accuracy = process_single(pic,model)
        info = {
            "picture": pic_name,
            # tokens & time of pipeline
            "tokens_total": tokens_total, 
            "time_total": t_total,
            # tokens & time of reasoning response generation
            "tokens_response": response_tokens,
            "tokens_reasoning": reasoning_tokens, 
            "time_response": response_time,
            "accuracy": accuracy
        }
        results.append(info)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"All finished! Process info is saved to {out_file}")        



if __name__ == "__main__":
    
    # read picture list
    with open("geomindmap/pictures/index.json", 'r', encoding='utf-8') as f:
        all_pic_list = json.load(f)

    # define the list of pictures to process
    pic_list =[
    "pic101.png",
    "pic34.png",
    "pic38.png"
    ]
    
    # batch process to generate GeoMindMap in pipeline
    batch(pic_list,"claude")

    # detect reasoning pattern and save to reasoning.json
    for pic in pic_list:
        pic_name = os.path.splitext(pic)[0]
        reasoning.detect_pattern(f"geomindmap/data/claude/{pic_name}/reasoning.json", "geomindmap/pictures/gps.json", pic, f"geomindmap/data/claude/{pic_name}/")
    
    


    