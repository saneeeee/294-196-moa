import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from llama_recipes.utils.train_utils import train
from huggingface_hub import login
from config import train_config, lora_config, huggingface_token
from data_preparation import prepare_data, tokenizer
from model_preparation import prepare_model
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import argparse

def parse_chapters(chapters_str):
    return chapters_str.split(',')

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the model with specific chapters")
    parser.add_argument(
        "--chapters",
        type=str,
        default="ch01,ch02,ch05,ch06,ch10,ch11,ch12,ch13,ch14,ch15",
        help="Comma-separated list of chapters (e.g., 'ch01,ch02,ch03')"
    )
    parser.add_argument(
        "--agent",
        type=int,
        choices=[1, 2, 3, 4],
        help="Select predefined agent chapters (1, 2, or 3)"
    )
    args = parser.parse_args()

    login(token=huggingface_token)

    # Predefined chapter sets
    agent_chapters = {
        1: ["ch01","ch02","ch05","ch06","ch10","ch11","ch12","ch13","ch14","ch15"],
        2: ["ch03","ch04","ch08","ch09","ch25","ch26"],
        3: ["ch07","ch16","ch17","ch18","ch19","ch20","ch21","ch22","ch23","ch24"],
        4 : ["ch01","ch02","ch03","ch04","ch05","ch06","ch07","ch08","ch09","ch10","ch11","ch12","ch13","ch14","ch15","ch16","ch17","ch18","ch19","ch20","ch21","ch22","ch23","ch24","ch25","ch26"]
    }

    # Use agent chapters if specified, otherwise use chapters from command line
    chapters = agent_chapters.get(args.agent) if args.agent else parse_chapters(args.chapters)
    
    train_loader, val_loader = prepare_data("../qa_pairs", chapters, tokenizer, train_config)
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
    # model_dir = train_config.output_dir
    # tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # model = LlamaForCausalLM.from_pretrained(model_dir)
    # prompt = "Answer the following questions:\nWhat are county codes used for?\n---\nAnswer:\n"
    # model.eval()
    # with torch.inference_mode():
    #     print(tokenizer.decode(model.generate(tokenizer.encode(prompt, return_tensors="pt"), max_length=100)[0], skip_special_tokens=True))