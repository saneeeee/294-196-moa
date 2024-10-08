import os
from datasets import Dataset
from load_data import load_data, preprocess_data
from preprocess import get_dataloader
from transformers import AutoTokenizer
from config import train_config

def prepare_data(directory, tokenizer, train_config):
    all_dialogs = load_data(directory)
    inputs, targets = preprocess_data(all_dialogs)
    qa_pairs = Dataset.from_dict({"input_text": inputs, "target_text": targets})
    qa_pairs = qa_pairs.train_test_split(test_size=0.1, shuffle=False)
    
    train_loader = get_dataloader(tokenizer, qa_pairs, train_config, split="train")
    val_loader = get_dataloader(tokenizer, qa_pairs, train_config, split="test")
    return train_loader, val_loader
tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token