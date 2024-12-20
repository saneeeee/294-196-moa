import os 
import json 
from glob import glob
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
from config import train_config
import torch
import time 
# load all of the files in the qa_pairs directory
def load_data(directory, chapters=None):
    """
    Function to load all of the files in the directory
    
    Args:
    directory: str, path to the directory
    
    Returns:
    all_dialogs: list of dictionaries, where each dictionary represents a dialogue
    """
    json_files = glob(os.path.join(directory, "**", "*.json"), recursive=True)
    json_files = sorted(json_files)
    if chapters:
        json_files = [f for f in json_files if any(chapter in f for chapter in chapters)]
    all_dialogs = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            all_dialogs.extend(data)
    return all_dialogs

def preprocess_data(data):
    """"
    Function to preprocess our data to create the input-target pairs
    
    Args:
    data: list of dictionaries, where each dictionary represents a dialogue
    
    Returns:
    inputs: list of strings, where each string is a prompt
    targets: list of strings, where each string is a response
    """
    qa_pairs = []
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
                        prompt = f"{user_content}"
                        qa_pairs.append((prompt, assistant_content))
                        
    return qa_pairs

def build_question_embeddings(questions):
    """
    Function use to build question embeddings
    
    Args:
    questions: list of strings, where each string is a question
    
    Returns:
    question_embeddings: np.array, where each row is an embedding for a question
    """
    model_encode = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    question_embeddings = model_encode.encode(questions)
    return question_embeddings

def build_nn(question_embeddings):
    """
    Function to build a nearest neighbors model

    Args:
    question_embeddings: np.array, where each row is an embedding for a question
    
    Returns:
    nn: NearestNeighbors object
    """
    nn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(question_embeddings)
    return nn

def retrieve_answer(query, nn, sentence_model, llama_model, tokenizer, answers):
    """
    Function use to retrieve an answer based on a query and a RAG model
    
    Args:
    query: str, the query
    nn: NearestNeighbors object
    sentence_model: SentenceTransformer object
    llama_model: LlamaForCausalLM object
    tokenizer: AutoTokenizer object
    answers: list of strings, where each string is an answer
    
    Returns:
    response: str, the response
    """
    query_embedding = sentence_model.encode([query])
    distances, indices = nn.kneighbors(query_embedding)
    retrieved_answers = [answers[idx] for idx in indices[0]]
    print(retrieved_answers)
    # prompt = "Answer the following question based on the information below:\n" + \
    #          "\n".join(retrieved_answers) + "\n---\nQuestion: " + query + "\nAnswer:"
    prompt = (
    "You are a knowledgeable assistant. Based on the information provided below, answer the user's question as accurately as possible. "
    "If the information is unclear or incomplete, make your best attempt to provide a helpful answer using your knowledge.\n\n"
    "Context:\n" + "\n".join(retrieved_answers) + 
    "\n---\n"
    "Question: " + query + 
    "\nAnswer:")
    llama_model.eval()
    start_time = time.time()
    with torch.inference_mode():
        response = tokenizer.decode(llama_model.generate(tokenizer.encode(prompt, return_tensors="pt"), max_new_tokens = 100)[0], skip_special_tokens=True)
    end_time = time.time()
    inference_time = end_time-start_time
    return response, inference_time

def run_inference(data, query, sentence_model, llama_model, tokenizer):
    """
    Function to run inference
    Args:
    data: list of tuples, where each tuple is a pair of question and answer
    query: str, the query
    sentence_model: SentenceTransformer object
    llama_model: LlamaForCausalLM object
    tokenizer: AutoTokenizer object
    
    Returns:
    response: str, the response
    """
    questions = [pair[0] for pair in data]
    answers = [pair[1] for pair in data]
    question_embeddings = build_question_embeddings(questions)
    nn = build_nn(question_embeddings)
    answer,inference_time = retrieve_answer(query, nn, sentence_model, llama_model, tokenizer, answers)
    return answer, inference_time

if __name__ == "__main__":
    directory = "../qa_pairs"
    # first agent
    chapters = ["ch01","ch02","ch05","ch06","ch10","ch11","ch12","ch13","ch14","ch15"]
    # second agent 
    #chapters = ["ch03","ch04","ch08","ch09","ch25","ch26"]
    # third agent
    #chapters = ["ch07","ch16","ch17","ch18","ch19","ch20","ch21","ch22","ch23","ch24"]
    # all agents
    #chapters = ["ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09", "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "ch16", "ch17", "ch18", "ch19", "ch20", "ch21", "ch22", "ch23", "ch24", "ch25", "ch26"]
    data = load_data(directory, chapters)
    qa_pairs= preprocess_data(data)
    query = "What is reg 5?"
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model_dir = train_config.output_dir
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    llama_model = LlamaForCausalLM.from_pretrained(model_dir)
    llama_model.eval()
    # TODO inference time and all the respone from three model and big one model 
    response, inference_time = run_inference(qa_pairs, query, sentence_model, llama_model, tokenizer)
    with open("response_rag.txt", "a") as f:
        f.write(f"Model Name: {model_dir}\nResponse: {response}\nInference Time: {inference_time:.2f} seconds\n\n")
 
    #print(f"response : {response}")