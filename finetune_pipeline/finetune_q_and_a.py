import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from llama_recipes.configs import train_config as TRAIN_CONFIG
import huggingface_hub
from dotenv import load_dotenv
import os

load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
huggingface_hub.login(token=huggingface_token)
train_config = TRAIN_CONFIG()
train_config.model_name = "meta-llama/Llama-3.2-3B"
train_config.run_validation = False
train_config.num_epochs = 1
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048
train_config.batching_strategy = "packing"
train_config.output_dir = "meta-llama-samsum"
train_config.use_peft = True

from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = LlamaForCausalLM.from_pretrained(
    train_config.model_name,
    device_map="auto",
    quantization_config=config,
    use_cache=False,
    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# eval_prompt = """
# "<User>: Where can I find the Vehicle Industry Registration Procedures Manual?\n<Assistant>:
# """
eval_prompt = """
"<User>: What is 1 + 1?\n<Assistant>:
"""
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.inference_mode():
    print(tokenizer.decode(model.generate(**model_input, max_length=100)[0], skip_special_tokens=True))