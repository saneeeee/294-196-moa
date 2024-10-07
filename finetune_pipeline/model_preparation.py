import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict

def prepare_model(train_config, lora_config):
    config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        device_map="auto",
        quantization_config=config,
        use_cache=False,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        torch_dtype=torch.float16,
    )
    
    peft_config = LoraConfig(**asdict(lora_config))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    return model