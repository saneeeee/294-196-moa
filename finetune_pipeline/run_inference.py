from config import huggingface_token
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from huggingface_hub import login
import time 
from config import train_config
if __name__ == "__main__":
    model_dir = train_config.output_dir
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #print(f"Model Name: {model_dir}")
    model = LlamaForCausalLM.from_pretrained(model_dir)
    prompt = "Answer the following questions:\nWhat is reg 5?\n---\nAnswer:\n"
    model.eval()
    start_time = time.time()
    with torch.inference_mode():
        response = tokenizer.decode(model.generate(tokenizer.encode(prompt, return_tensors="pt"), max_length=100)[0], skip_special_tokens=True)
    end_time = time.time() 
    inference_time = end_time-start_time
    with open("response_3.2_3B.txt", "a") as f:
        f.write(f"Model Name: {model_dir}\nResponse: {response}\nInference Time: {inference_time:.2f} seconds\n\n")