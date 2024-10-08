import os
from dotenv import load_dotenv
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import lora_config as LORA_CONFIG
import torch
import huggingface_hub

load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
huggingface_hub.login(token=huggingface_token)
huggingface_hub
train_config = TRAIN_CONFIG()
train_config.model_name = "meta-llama/Llama-3.1-8B-Instruct"
train_config.run_validation = False
train_config.num_epochs = 60
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048
train_config.batching_strategy = "packing"
train_config.output_dir = "meta-llama-qa-llama-3.1-8B-Instruct-60-epochs"
train_config.use_peft = True

lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_dropout: float = 0.01
