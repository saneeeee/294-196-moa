from huggingface_hub import HfApi, upload_folder
import os
from dotenv import load_dotenv


username = "Phudish"
repo_name = "meta-llama-qa-llama-3.1-70B-Instruct-10-epochs-third-agent"
folder_path = "meta-llama-qa-llama-3.1-70B-Instruct-10-epochs-third-agent"
repo_id = f"{username}/{repo_name}"

load_dotenv()
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
api = HfApi()
try:
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        token=hf_token
    )
    print(f"Repository created at {repo_url}")
except Exception as e:
    print(f"Repository might already exist or there was an error: {e}")

print("Uploading folder...")
upload_folder(
    folder_path=folder_path,
    path_in_repo=".",  
    repo_id=repo_id,
    repo_type="model",
    token=hf_token
)

print(f"Folder '{folder_path}' successfully uploaded to Hugging Face at {repo_url}")