import os
from framework import excutable, load_json_data, preprocess_json_data
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

# def load_json_data(file_path):
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# def preprocess_json_data(data):
#     qa_pairs = [(item['input_text'], item['target_text']) for item in data]
#     return qa_pairs

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



def retrieve_answer(query, nn, sentence_model, answers):
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
    "\n---\n")
    
    
    start_time = time.time()
        
    end_time = time.time()
    inference_time = end_time - start_time
    return response, inference_time

# def run_rag(train_data, query):
#     train_question = [pair[0] for pair in train_data]
#     train_answer = [pair[1] for pair in train_data]
#     sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#     train_data_embedding = sentence_model.encode(train_question)
#     nn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(train_data_embedding)
#     query_embedding = sentence_model.encode(query)
#     # Find the nearest neighbors
#     distances, indices = nn.kneighbors(query_embedding)
#     retrieved_answers = [train_answer[idx] for idx in indices[0]]
#     prompt = (
#     "You are a knowledgeable assistant. Based on the information provided below, answer the user's question as accurately as possible. "
#     "If the information is unclear or incomplete, make your best attempt to provide a helpful answer using your knowledge.\n\n"
#     "Context:\n" + "\n".join(retrieved_answers)
#     )
#     return prompt
    
     
    
    


# def run_inference(data, queries, sentence_model, llama_model, tokenizer, batch_size=8):
#     """
#     Function to run inference in batches
#     Args:
#     data: list of tuples, where each tuple is a pair of question and answer
#     queries: list of str, the queries
#     sentence_model: SentenceTransformer object
#     llama_model: LlamaForCausalLM object
#     tokenizer: AutoTokenizer object
#     batch_size: int, the batch size for processing queries
    
#     Returns:
#     responses: list of str, the responses
#     inference_times: list of float, the inference times
#     """
#     questions = [pair[0] for pair in data]
#     answers = [pair[1] for pair in data]
#     question_embeddings = build_question_embeddings(questions)
#     nn = build_nn(question_embeddings)
    
#     responses = []
#     inference_times = []
    
#     for i in range(0, len(queries), batch_size):
#         batch_queries = queries[i:i+batch_size]
#         start_time = time.time()
#         batch_responses = [retrieve_answer(query, nn, sentence_model, llama_model, tokenizer, answers) for query in batch_queries]
#         end_time = time.time()
#         batch_inference_time = (end_time - start_time) / len(batch_queries)
        
#         responses.extend([response for response, _ in batch_responses])
#         inference_times.extend([batch_inference_time] * len(batch_queries))
    
#     return responses, inference_times



# def run_infernece(query,batch_size=1,roles_str="Agent1 is resonsible for General Vehicle Registration and Licensing. Agent2 is responsible for Fees, Taxes, and Financial Management. Agent3 is resposible for Special Vehicles, Plates, and Documentation. Do not answer the question directly. In your response, call the appropriate agent, either agent1, agent2 or agent3, based on the question. Your reponse should only contain Invoke agentx where x can be either 1,2 or 3. Question:"):
#     responses = []
#     inference_times = []
    
#     for i in range(0, len(query), batch_size):
#         batch_queries = query[i:i+batch_size]
#         start_time = time.time()
#         batch_responses = [excutable(question_str=query, roles_str=roles_str) for query in batch_queries]
#         end_time = time.time()
#         batch_inference_time = (end_time - start_time) / len(batch_queries)
        
#         responses.extend([response for response in batch_responses])
#         inference_times.extend([batch_inference_time] * len(batch_queries))
    
#     return responses, inference_times
def run_infernece(query, roles_str="Agent1 is resonsible for General Vehicle Registration and Licensing. Agent2 is responsible for Fees, Taxes, and Financial Management. Agent3 is resposible for Special Vehicles, Plates, and Documentation. Do not answer the question directly. In your response, call the appropriate agent, either agent1, agent2 or agent3, based on the question. Your reponse should only contain Invoke agentx where x can be either 1,2 or 3. Question:"):
    start_time = time.time()
    response = excutable(question_str=query, roles_str=roles_str)
    end_time = time.time()
    inference_time = end_time - start_time
    return response, inference_time

if __name__ == "__main__":
    
    # excutable(question_str="How much does it cost to register a car ?", 
    #           roles_str="Agent1 is resonsible for General Vehicle Registration and Licensing. Agent2 is responsible for Fees, Taxes, and Financial Management. Agent3 is resposible for Special Vehicles, Plates, and Documentation. Do not answer the question directly. In your response, call the appropriate agent, either agent1, agent2 or agent3, based on the question. Your reponse should only contain Invoke agentx where x can be either 1,2 or 3. Question:")
    
        
    test_json_file_path = "/accounts/grad/phudish_p/294-196-moa/finetune_pipeline/orchestrator_new/orchestrator_test.json" 
    test_data = load_json_data(test_json_file_path)
    test_qa_pairs = preprocess_json_data(test_data)
    results = []
    queries = [pair[0] for pair in test_qa_pairs]
    target_texts = [pair[1] for pair in test_qa_pairs]
    
    start_time = time.time()  # Start time for the entire process
    
         
        # for i in tqdm(range(0, len(queries), 8), desc="Processing Batches"):
        #     batch_queries = queries[i:i+8]
        #     batch_target_texts = target_texts[i:i+8]
        #     batch_responses, batch_inference_times = run_infernece(batch_queries)
        #     for query, target_texts, response, inference_time in zip(batch_queries, batch_target_texts, batch_responses, batch_inference_times):
        #         bleu_score = bleu.compute(predictions=[response], references=[target_texts])['bleu']
        #         rouge_score = rouge.compute(predictions=[response], references=[target_texts])
        #         result = {
        #             'query': query,
        #             'target_text': target_texts,
        #             'response': response,
        #             'inference_time': inference_time,
        #             'bleu_score': bleu_score,
        #             'rouge_score': rouge_score
        #         }
        #         results.append(result)
        #         f.write(json.dumps(result) + "\n")
        #         print(f"Query: {query}")
        #         print(f"Target Text: {target_texts}")
        #         print(f"Response: {response}")
        #         print(f"Inference Time: {inference_time:.2f} seconds")
        #         print(f"BLEU Score: {bleu_score}")
        #         print(f"ROUGE Score: {rouge_score}")
        #         print("\n")
        
    for (i, query) in tqdm(enumerate(queries), desc="Processing Queries"):
        response, inference_time = run_infernece(query)
        #answer_only = response[response.find(query) + len(query):]
        print("Inference time ", inference_time)
        bleu_score = bleu.compute(predictions=[response], references=[target_texts[i]])['bleu']
        print("bleu computed")
        rouge_score = rouge.compute(predictions=[response], references=[target_texts[i]])
        print("rouge computed")
        result = {
            'query': query,
            'target_text': target_texts[i],
            'response': response,
            'inference_time': inference_time,
            #'answer_only': answer_only,
            'bleu_score': bleu_score,
            'rouge_score': rouge_score
        }
        results.append(result)
        with open("evaluation_results_architecture_2.json", "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"Query: {query}")
        print(f"Target Text: {target_texts[i]}")
        print(f"Response: {response}")
        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"BLEU Score: {bleu_score}")
        print(f"ROUGE Score: {rouge_score}")
        print("\n")

    end_time = time.time()
                
    total_time = end_time - start_time
    with open("evaluation_results_architecture.json", "a") as f:
        f.write(f"Total Time: {total_time} seconds\n")
    print("Evaluation complete")