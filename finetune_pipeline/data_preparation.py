import os
from datasets import Dataset
from load_data import load_data, preprocess_data, load_data_for_orchestrator
from preprocess import get_dataloader
from transformers import AutoTokenizer
from config import train_config

def prepare_data(directory, chapters, tokenizer, train_config):
    agent_type = ""
    if chapters == ["ch01","ch02","ch05","ch06","ch10","ch11","ch12","ch13","ch14","ch15"]:
        agent_type = "first_agent"
    elif chapters == ["ch03","ch04","ch08","ch09","ch25","ch26"]:
        agent_type = "second_agent"
    elif chapters == ["ch07","ch16","ch17","ch18","ch19","ch20","ch21","ch22","ch23","ch24"]:
        agent_type = "third_agent"
    else:
        agent_type = "orchestrator"
        # Load combined data in order from existing agent files
        train_dialogs = load_data_for_orchestrator(directory,split="train")
        test_dialogs = load_data_for_orchestrator(directory,split="test")
        all_dialogs = train_dialogs + test_dialogs        
        qa_pairs = Dataset.from_dict({"input_text": [d['input_text'] for d in all_dialogs], 
                                    "target_text": [d['target_text'] for d in all_dialogs]})
        
        # Split while maintaining order
        train_size = int(0.8 * len(qa_pairs))
        qa_pairs = qa_pairs.train_test_split(
            train_size=train_size,
            shuffle=False 
        )
        
        os.makedirs(f"{agent_type}_new", exist_ok=True)
        qa_pairs["train"].to_json(f"{agent_type}_new/{agent_type}_train.json")
        qa_pairs["test"].to_json(f"{agent_type}_new/{agent_type}_test.json")
        
        train_loader = get_dataloader(tokenizer, qa_pairs, train_config, split="train")
        val_loader = get_dataloader(tokenizer, qa_pairs, train_config, split="test")
        
        return train_loader, val_loader

    all_dialogs = load_data(directory, chapters)
    inputs, targets = preprocess_data(all_dialogs)
    qa_pairs = Dataset.from_dict({"input_text": inputs, "target_text": targets})
    qa_pairs = qa_pairs.train_test_split(test_size=0.2, shuffle=True, seed=196)

    os.makedirs(agent_type, exist_ok=True)
    qa_pairs["train"].to_json(f"{agent_type}/{agent_type}_train.json")
    qa_pairs["test"].to_json(f"{agent_type}/{agent_type}_test.json")

    train_loader = get_dataloader(tokenizer, qa_pairs, train_config, split="train")
    val_loader = get_dataloader(tokenizer, qa_pairs, train_config, split="test")
    
    return train_loader, val_loader
tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token