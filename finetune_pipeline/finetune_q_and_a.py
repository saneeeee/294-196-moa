import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.data import DataCollatorForSeq2Seq
from torch.utils.data import DistributedSampler
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.configs import train_config as TRAIN_CONFIG
import torch.distributed as dist
from transformers import default_data_collator
import huggingface_hub
from dotenv import load_dotenv
import os
import json 
from glob import glob 
import datasets
from datasets import Dataset, DatasetDict
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


# load all of the files in the qa_pairs directory
qa_pairs = "../qa_pairs"
json_files = glob(os.path.join(qa_pairs, "**", "*.json"), recursive=True)
json_files = sorted(json_files)

all_dialogs = []
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
        all_dialogs.extend(data)

# Preprocess the data to create input-target pairs
def preprocess_data(data):
    """"
    Function to preprocess our data to create the input-target pairs
    
    Args:
    data: list of dictionaries, where each dictionary represents a dialogue
    
    Returns:
    inputs: list of strings, where each string is a prompt
    targets: list of strings, where each string is a response
    """
    inputs = []
    targets = []
    for dialogue in data:
        for turn_index in range(len(dialogue)):
            turn = dialogue[turn_index]
            if turn['role'].lower() == 'user':
                user_content = turn['content'].strip()
                if turn_index + 1 < len(dialogue):
                    next_turn = dialogue[turn_index + 1]
                    if next_turn['role'].lower() == 'assistant':
                        assistant_content = next_turn['content'].strip()
                        # Create a prompt similar to SAMSum's prompt
                        prompt = f"Question: {user_content}\nAnswer:"
                        inputs.append(prompt)
                        targets.append(assistant_content)
    return inputs, targets

# maket the dataset to be the DatasetDict as well
inputs, targets = preprocess_data(all_dialogs)
qa_pairs = Dataset.from_dict({"input_text": inputs, "target_text": targets})
qa_pairs = dataset.train_test_split(test_size=0.1)

# load the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

special_tokens_dict = {"additional_special_tokens": ["<user>", "<assistant>"]}
tokenizer.add_special_tokens(special_tokens_dict)

def get_preprocessed_qa(dataset_config, tokenizer,split):
    prompt = (
        f"Answer the following question:\n"
    )
    
    def apply_prompt_template(sample):
        return {
            "prompt" : prompt,
            "answer" : sample["target_text"]
        }
    
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["target_text"] + tokenizer.eos_token, add_special_tokens=False)
        
        sample = {
            "input_ids": prompt + answer,
            "attention_mask": [1] * len(prompt) + [1] * len(answer),
            "labels": [-100] * len(prompt) + answer
        }
        
        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=;ist(dataset.features))
    return dataset


def get_dataloader_kwargs(train_config, dataset, dataset_processer, mode):
    kwargs = {}
    batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
    if train_config.batching_strategy == "padding":
        if train_config.enable_fsdp:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
        else:
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(dataset_processer)
    elif train_config.batching_strategy == "packing":
        if train_config.enable_fsdp:
            kwargs["sampler"] = DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=mode=="train",
            drop_last=True,
        )
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
        kwargs["collate_fn"] = default_data_collator
    else:
        raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")
    return kwargs


def get_dataloader(tokenizer, dataset_config, train_config, split = "train"):
    dataset = get_preprocessed_qa(dataset, dataset_config, tokenizer, split)
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)
    
    if split == "train" and train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)
    
    dataloader = torch.utils.data.DataLoader(dataset, **dl_kwargs)
    return dataloader

load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
huggingface_hub.login(token=huggingface_token)
train_config = TRAIN_CONFIG()
huggingface_hub
train_config.model_name = "meta-llama/Llama-3.2-3B"
train_config.run_validation = False
train_config.num_epochs = 1
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.output_dir = "meta-llama-samsum"
train_config.use_peft = True

train_loader = get_dataloader(tokenizer, dataset, train_config, split="train")