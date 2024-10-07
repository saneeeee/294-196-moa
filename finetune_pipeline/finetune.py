import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from llama_recipes.utils.train_utils import train
from huggingface_hub import login
from config import train_config, lora_config, huggingface_token
from data_preparation import prepare_data, tokenizer
from model_preparation import prepare_model

def main():
    login(token=huggingface_token)
    
    train_loader, val_loader = prepare_data("../qa_pairs", tokenizer, train_config)
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

if __name__ == "__main__":
    main()