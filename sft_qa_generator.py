from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 


from openai import OpenAI
import re
import json


def user_prompt_for(scrape_file):
    knowledge_base = ''
    with open(scrape_file, 'r') as f:
        knowledge_base += f.read() + "\n"
    return USER_PROMPT + knowledge_base + "\n<end_knowledge_text>"