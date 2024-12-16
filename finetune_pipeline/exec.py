from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from peft import PeftModelForCausalLM, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM
import os
from transformers import AutoTokenizer, pipeline
import torch


# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder
    return "DMV registers vehicles"


tools = [search]
token = os.getenv("HUGGINGFACE_TOKEN")

tool_node = ToolNode(tools)

if torch.cuda.is_available():
    print ("Using cuda")
    mps_device = torch.device("cuda")

elif torch.backends.mps.is_available():
    print ("Using mps")
    mps_device = torch.device("mps")
else:
    print ("Using cpu")
    mps_device = torch.device("cpu")
    
torch.cuda.empty_cache()
# Initialize smaller language models as separate agents (nodes in the graph) - gen_model, gen_model1, gen_model2, orch
config = PeftConfig.from_pretrained("Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-first-agent")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=token, load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, "Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-first-agent", token=token, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=token)

gen_model = pipeline("text-generation", model=base_model, tokenizer=tokenizer, torch_dtype=torch.float16)
gen_model.model = model  # This is a workaround

config1 = PeftConfig.from_pretrained("Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-second-agent")
model1 = PeftModel.from_pretrained(base_model, "Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-second-agent", token=token, load_in_4bit=True)

gen_model1 = pipeline("text-generation", model=base_model, tokenizer=tokenizer, torch_dtype=torch.float16)
gen_model.model = model1  # This is a workaround

config2 = PeftConfig.from_pretrained("Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-third-agent")
model2 = PeftModel.from_pretrained(base_model, "Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-third-agent", token=token, load_in_4bit=True)

gen_model2 = pipeline("text-generation", model=base_model, tokenizer=tokenizer, torch_dtype=torch.float16)
gen_model.model = model2  # This is a workaround

config_orch = PeftConfig.from_pretrained("Phudish/meta-llama-qa-llama-3.1-70B-Instruct-10-epochs-orchestrator")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", token=token, load_in_4bit=True)
model_orch = PeftModel.from_pretrained(base_model, "Phudish/meta-llama-qa-llama-3.1-70B-Instruct-10-epochs-orchestrator", token=token, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", token=token)

gen_model_orch = pipeline("text-generation", model=base_model, tokenizer=tokenizer, torch_dtype=torch.float16)
model_orch.gradient_checkpointing_enable()
# print("Start test")
# output = gen_model("How to register car", max_new_tokens=1)
# print(output)
# print("End test")

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    print("last message ", last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if isinstance(last_message, HumanMessage) and "tool_calls" in last_message.content:
        return "tools"
    # Otherwise, we stop (reply to the user)
    # Change this for arch
    return END

def next_agent(state:MessagesState):
    messages = state['messages']
    last_message = messages[-1]

    # Replace this with a parser
    if last_message and "agent1" in last_message.content:
        return "agent1"
    elif last_message and "agent2" in last_message.content:
        return "agent2"
    elif last_message and "agent3" in last_message.content:
        return "agent3"
    else:
        return END
    


# Define the function that calls the model
def call_model1(state: MessagesState):
    messages = state['messages'][0].content
    response = gen_model(messages, max_new_tokens=50)
    print("Response from model: ", response)
    return {"messages": [HumanMessage(content=response[0]['generated_text'])]}

def call_model2(state: MessagesState):
    messages = state['messages'][0].content
    response = gen_model1(messages, max_new_tokens=50)
    print("Response from model: ", response)
    return {"messages": [HumanMessage(content=response[0]['generated_text'])]}

def call_model3(state: MessagesState):
    messages = state['messages'][0].content
    response = gen_model2(messages, max_new_tokens=50)
    print("Response from model: ", response)
    return {"messages": [HumanMessage(content=response[0]['generated_text'])]}

def call_orch(state: MessagesState):
    messages = state['messages'][0].content
    response = gen_model_orch(messages, max_new_tokens=10)
    print("Response from model: ", response)
    return {"messages": [HumanMessage(content=response[0]['generated_text'])]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define nodes
workflow.add_node("orch", call_orch)
workflow.add_node("agent1", call_model1)
workflow.add_node("agent2", call_model2)
workflow.add_node("agent3", call_model3)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "orch")

workflow.add_conditional_edges(
    # First, we define the start node. We use `orch`.
    # This means these are the edges taken after the `orch` node is called.
    "orch",
    # Next, we pass in the function that will determine which node is called next.
    next_agent,
)

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent1",
    should_continue,
)

workflow.add_conditional_edges(
    "agent2",
    should_continue,
)

workflow.add_conditional_edges(
    "agent3",
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
#workflow.add_edge("tools", 'orch')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
print("Going to compile")
app = workflow.compile(checkpointer=checkpointer)
print("Done compile")

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="How to register car for DMV? Do not directly answer the question. Have either agent1 or agent2 or agent3 to process information. Invoke an agent.")]},
    config={"configurable": {"thread_id": 42}}
)
print("Done invoking")
print(final_state["messages"][-1].content)