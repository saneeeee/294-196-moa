import os
import json
from glob import glob
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from config import train_config
import torch
import time
import evaluate  # For evaluation metrics
from tqdm import tqdm  # For progress tracking

# Load the evaluation metrics
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')

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

def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_json_data(data):
    qa_pairs = [(item['input_text'], item['target_text']) for item in data]
    return qa_pairs

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
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(llama_model.device)
        response = tokenizer.decode(llama_model.generate(input_ids, max_new_tokens=100)[0], skip_special_tokens=True)
    end_time = time.time()
    inference_time = end_time - start_time
    # find after only the Answer:
    answer_only = response[response.find("Answer:") + len("Answer:"):]
    return response, inference_time, answer_only

def run_inference(data, queries, sentence_model, llama_model, tokenizer, batch_size=1):
    """
    Function to run inference in batches
    Args:
    data: list of tuples, where each tuple is a pair of question and answer
    queries: list of str, the queries
    sentence_model: SentenceTransformer object
    llama_model: LlamaForCausalLM object
    tokenizer: AutoTokenizer object
    batch_size: int, the batch size for processing queries
    
    Returns:
    responses: list of str, the responses
    inference_times: list of float, the inference times
    """
    questions = [pair[0] for pair in data]
    answers = [pair[1] for pair in data]
    question_embeddings = build_question_embeddings(questions)
    nn = build_nn(question_embeddings)
    
    responses = []
    inference_times = []
    answer_only = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        start_time = time.time()
        batch_responses = [retrieve_answer(query, nn, sentence_model, llama_model, tokenizer, answers) for query in batch_queries]
        end_time = time.time()
        batch_inference_time = (end_time - start_time) / len(batch_queries)
        
        responses.extend([response for response, _, _ in batch_responses])
        answer_only.extend([answer for _, _, answer in batch_responses])
        inference_times.extend([batch_inference_time] * len(batch_queries))
    
    return responses, inference_times, answer_only

if __name__ == "__main__":
    train_json_file_path = "/accounts/grad/phudish_p/294-196-moa/finetune_pipeline/orchestrator_new/orchestrator_train.json"
    test_json_file_path = "/accounts/grad/phudish_p/294-196-moa/finetune_pipeline/orchestrator_new/orchestrator_test.json"
    
    # Load and preprocess training data
    train_data = load_json_data(train_json_file_path)
    qa_pairs = preprocess_json_data(train_data)
    
    # Load and preprocess test data
    test_data = load_json_data(test_json_file_path)
    test_qa_pairs = preprocess_json_data(test_data)
    #test_qa_pairs = test_qa_pairs[:8]
    
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model_dir = train_config.output_dir
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model in 4-bit precision
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    llama_model = LlamaForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)
    
    # No need to manually move the model to the device
    llama_model.eval()
    
    # Prepare the question embeddings and NN model
    # questions = [pair[0] for pair in qa_pairs]
    # print(f"Questions: {questions[0:5]}")
    # answers = [pair[1] for pair in qa_pairs]
    # question_embeddings = build_question_embeddings(questions)
    # nn = build_nn(question_embeddings)

    results = []
    queries = [pair[0] for pair in test_qa_pairs]
    target_texts = [pair[1] for pair in test_qa_pairs]
    
    start_time = time.time()  # Start time for the entire process
         
    for i in tqdm(range(0, len(queries), 1), desc="Processing Batches"):
        batch_queries = queries[i:i+1]
        batch_target_texts = target_texts[i:i+1]
        batch_responses, batch_inference_times, batch_answer_only = run_inference(qa_pairs, batch_queries, sentence_model, llama_model, tokenizer)
        
        for query, target_text, response, inference_time, answer_only in zip(batch_queries, batch_target_texts, batch_responses, batch_inference_times, batch_answer_only):
            # Compute evaluation metrics
            bleu_score = bleu.compute(predictions=[answer_only], references=[target_text])['bleu']
            rouge_score = rouge.compute(predictions=[answer_only], references=[target_text])

            result = {
                'query': query,
                'target_text': target_text,
                'response': response,
                'inference_time': inference_time,
                'answer_only': answer_only,
                'bleu_score': bleu_score,
                'rouge_score': rouge_score
            }
            results.append(result)

            # Write the result to the file
            with open("evaluation_results_baseline_2.json", "a") as f:
                f.write(json.dumps(result) + "\n")

            # Print the results
            print(f"Query: {query}")
            print(f"Target Text: {target_text}")
            print(f"Response: {response}")
            print(f"Inference Time: {inference_time:.2f} seconds")
            print(f"Answer Only: {answer_only}")
            print(f"BLEU Score: {bleu_score}")
            print(f"ROUGE Score: {rouge_score}")
            print("\n")
    
    end_time = time.time()  
    total_time = end_time - start_time
    print(f"Total Time: {total_time} seconds")
    with open("evaluation_results.json", "a") as f:
        f.write(f"Total Time: {total_time} seconds\n")
    print("Evaluation complete")