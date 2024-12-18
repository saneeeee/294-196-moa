import os 
import json 
from glob import glob

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

def load_data_for_orchestrator(directory, split):
    """
    Load training data maintaining the exact order from each agent's training set
    """
    combined_data = []
    
    def read_json_file(filepath):
        data = []
        with open(filepath, "r") as f:
            content = f.read()
            for line in content.strip().split('\n'):
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    # Read first agent training data
    first_agent_data = read_json_file(f"first_agent/first_agent_{split}.json")
    combined_data.extend(first_agent_data)
    
    # Read second agent training data 
    second_agent_data = read_json_file(f"second_agent/second_agent_{split}.json")
    combined_data.extend(second_agent_data)
    
    # Read third agent training data
    third_agent_data = read_json_file(f"third_agent/third_agent_{split}.json")
    combined_data.extend(third_agent_data)
    
    return combined_data

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
                        prompt = f"{user_content}"
                        inputs.append(prompt)
                        targets.append(assistant_content)
    return inputs, targets
