import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from llama_recipes.utils.train_utils import train
from huggingface_hub import login
from config import train_config, lora_config, huggingface_token
from data_preparation import prepare_data, tokenizer
from model_preparation import prepare_model
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

def main():
    login(token=huggingface_token)
    #chapters = ["ch01","ch02","ch05","ch06","ch10","ch11","ch12","ch13","ch14","ch15"]
    #chapters = ["ch03","ch04","ch08","ch09","ch25","ch26"]
    chapters = ["ch07","ch16","ch17","ch18","ch19","ch20","ch21","ch22","ch23","ch24"]
    train_loader, val_loader = prepare_data("../qa_pairs", chapters,tokenizer, train_config)
    model = prepare_model(train_config, lora_config)
    
    model.train()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    results = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run=None,
    )
    model.save_pretrained(train_config.output_dir)

if __name__ == "__main__":
    main()
    model_dir = train_config.output_dir
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_dir)
    prompt = "Answer the following questions:\nWhat are county codes used for?\n---\nAnswer:\n"
    model.eval()
    with torch.inference_mode():
        print(tokenizer.decode(model.generate(tokenizer.encode(prompt, return_tensors="pt"), max_length=100)[0], skip_special_tokens=True))