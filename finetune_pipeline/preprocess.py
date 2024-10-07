import torch
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from transformers.data import DataCollatorForSeq2Seq
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from llama_recipes.data.concatenator import ConcatDataset
from transformers import default_data_collator

def get_preprocessed_data(dataset, tokenizer, split):
    """
    Function use to preprocess a dataset for a question-answering task by applying a prompt template and tokenizing the data
    
    Args:
    dataset: Dataset, the dataset to preprocess
    tokenizer: AutoTokenizer, the tokenizer to use for tokenization
    split: str, the split of the dataset either "train" or "test" 
    
    Returns:
    Dataset: The preprocessed dataset with "input_ids", "attention_mask", and "labels" columns
    """
    prompt = (
        f"Answer the following questions:\n{{question}}\n---\nAnswer:\n"
    )
    
    def apply_prompt_template(sample):
        """
        Function use to apply the prompt template to the dataset
        Args:
        sample: dict, a sample from the dataset
        
        Returns:
        dict: The sample with the prompt applied
        """
        return {
            "prompt" : prompt.format(question=sample["input_text"]),
            "answer" : sample["target_text"]
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=["input_text", "target_text"])
    def tokenize_add_label(sample):
        """
        Function use to tokenize the data and add the labels
        Args:
        sample: dict, a sample from the dataset
        Returns:
        dict: The sample with the tokenized data and labels
        """
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] + tokenizer.eos_token, add_special_tokens=False)
        sample = {
            "input_ids": prompt + answer,
            "attention_mask": [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer
        }
        return sample
    dataset = dataset.map(tokenize_add_label, remove_columns=["prompt", "answer"])
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

def get_dataloader(tokenizer, dataset, train_config, split):
    """
    Function use to get the dataloader for the dataset
    
    Args:
    tokenizer: AutoTokenizer, the tokenizer to use for tokenization
    dataset: Dataset, the dataset to preprocess
    train_config: dict, the training configuration
    split: str, the split of the dataset either "train" or "test"
    
    Returns:
    DataLoader: The dataloader for the dataset
    """
    dataset = get_preprocessed_data(dataset, tokenizer, split)[split]
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)
    if split == "train" and train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)
    
    dataloader = torch.utils.data.DataLoader(dataset, **dl_kwargs)
    return dataloader
