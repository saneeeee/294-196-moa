from dotenv import load_dotenv
from openai import OpenAI
import json
import re
import os

from moatask.configs.sft_prompt import get_system_prompt, get_user_prompt

def sft_qa_pairs(scrape_file, oaiclient):
    user_prompt = get_user_prompt(scrape_file)
    completion = oaiclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    content = completion.choices[0].message.content
    try:
        pattern = r'(\[{.*?}\])'
        matches = re.findall(pattern, content, re.DOTALL)
        split = [match.strip() for match in matches]
        parsed_result = [json.loads(item) for item in split]
        return completion, parsed_result
    except:
        return completion, None

def sft_qa_pair(from_directory_name, oaiclient, max_entries=None, save=False, savedir=None):
    entries = 0
    for root, dirs, files in os.walk(from_directory_name):
        for file in files:
            if file.endswith(".txt"):
                if (max_entries and entries >= max_entries):
                    return
                file_path = os.path.join(root, file)
                completion, parsed_result = sft_qa_pairs(file_path, oaiclient)

                entries += 1

                if save and savedir:
                    relative_path = os.path.relpath(root, from_directory_name)
                    save_subdir = os.path.join(savedir, relative_path)
                    os.makedirs(save_subdir, exist_ok=True)
                    base_filename = os.path.splitext(file)[0]
                    
                    txt_filename = f"{base_filename}_model_output.txt"
                    txt_path = os.path.join(save_subdir, txt_filename)

                    with open(txt_path, 'w') as txt_file:
                        txt_file.write(completion.choices[0].message.content)

                    if parsed_result != None: 
                        json_filename = f"{base_filename}_qa_pairs.json"
                        json_path = os.path.join(save_subdir, json_filename)

                        with open(json_path, 'w') as json_file:
                            json.dump(parsed_result, json_file, indent=2)
                    
                    print(f"Saved QA pairs for {file} to {json_path}")

if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
    client = OpenAI(api_key=OPENAI_API_KEY)

    sft_qa_pair("./dataset", client, max_entries=3, save=True, savedir="./qa_pairs")
