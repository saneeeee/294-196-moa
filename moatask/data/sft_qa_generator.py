from dotenv import load_dotenv
from openai import OpenAI
import json
import re
import os

from moatask.configs.sft_prompt import get_system_prompt, get_user_prompt

def sft_qa_pairs(scrape_file_name, oaiclient, n_questions=100):

    completion = oaiclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {
                "role": "user",
                "content": get_user_prompt(scrape_file_name, n_questions)
            }
        ]
    )
    content = completion.choices[0].message.content
    
    pattern = r'\[\d+/\d+\]\s+'
    split = re.split(pattern, content)
    split = [item for item in split if item]
    parsed_result = [json.loads(item) for item in split]
    return parsed_result

def sft_qa_pair(from_directory_name, oaiclient, n_questions=100, save=False, savedir=None):
    for root, _, files in os.walk(from_directory_name):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                parsed_result = sft_qa_pairs(file_path, oaiclient, n_questions)
                
                if save and savedir:
                    
                    os.makedirs(savedir, exist_ok=True)
                    
                    base_filename = os.path.splitext(file)[0]
                    json_filename = f"{base_filename}_qa_pairs.json"
                    json_path = os.path.join(savedir, json_filename)
                    
                    with open(json_path, 'w') as json_file:
                        json.dump(parsed_result, json_file, indent=2)
                    
                    print(f"Saved QA pairs for {file} to {json_path}")

if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
    client = OpenAI(api_key=OPENAI_API_KEY)

    sft_qa_pair("./dataset", client, n_questions=100, save=True, savedir="./qa_pairs")
