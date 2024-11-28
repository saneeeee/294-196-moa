from huggingface_hub import HfApi, upload_folder, ModelCard, ModelCardData
import os
from dotenv import load_dotenv
import json

username = "Phudish"
repo_name = "meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-first-agent"
folder_path = "meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-first-agent"
repo_id = f"{username}/{repo_name}"

load_dotenv()
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
api = HfApi()
try:
    repo_info = api.repo_info(repo_id=repo_id, token=hf_token)
    print(f"Repository {repo_id} already exists.")
    repo_url = f"https://huggingface.co/{repo_id}"
except Exception:
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            token=hf_token
        )
        print(f"Repository created at {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        exit(1)

print("Uploading folder...")
upload_folder(
    folder_path=folder_path,
    path_in_repo=".",  
    repo_id=repo_id,
    repo_type="model",
    token=hf_token
)

print(f"Folder '{folder_path}' successfully uploaded to Hugging Face at {repo_url}")

