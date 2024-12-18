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
gen_model1.model = model1  # This is a workaround

config2 = PeftConfig.from_pretrained("Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-third-agent")
model2 = PeftModel.from_pretrained(base_model, "Phudish/meta-llama-qa-llama-3.1-8B-Instruct-100-epochs-third-agent", token=token, load_in_4bit=True)

gen_model2 = pipeline("text-generation", model=base_model, tokenizer=tokenizer, torch_dtype=torch.float16)
gen_model2.model = model2  # This is a workaround

config_orch = PeftConfig.from_pretrained("Phudish/meta-llama-qa-llama-3.1-70B-Instruct-10-epochs-orchestrator")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", token=token, load_in_4bit=True)
model_orch = PeftModel.from_pretrained(base_model, "Phudish/meta-llama-qa-llama-3.1-70B-Instruct-10-epochs-orchestrator", token=token, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", token=token)

gen_model_orch = pipeline("text-generation", model=base_model, tokenizer=tokenizer, torch_dtype=torch.float16)
gen_model_orch.model = model_orch

model_orch.gradient_checkpointing_enable()
# print("Start test")
# output = gen_model("How to register car", max_new_tokens=1)
# print(output)
# print("End test")

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # print("last message ", last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if isinstance(last_message, HumanMessage) and "tool_calls" in last_message.content:
        return "tools"
    # Otherwise, we stop (reply to the user)
    # Change this for arch
    return END

def next_agent(state:MessagesState):
    messages = state['messages']
    last_message = messages[-1]
    agent_str = messages[-2]

    # print("Last message", last_message, "\n")
    # print("Agent call", agent_str, "\n")
    # print("All msg", messages, "\n")

    # Replace this with a parser
    if last_message and "agent1" in agent_str.content.lower():
        print("Agent 1 called \n")
        return "agent1"
    elif last_message and "agent2" in agent_str.content.lower():
        print("Agent 2 called \n")
        return "agent2"
    elif last_message and "agent3" in agent_str.content.lower():
        print("Agent 3 called \n")
        return "agent3"
    else:
        print("No agent called \n")
        return END
    


# Define the function that calls the model
def call_model1(state: MessagesState):
    messages = state['messages'][1].content
    response = gen_model(messages, max_new_tokens=100)
    # print("Response from model1 : ", response , "\n")
    return {"messages": [HumanMessage(content=response[0]['generated_text'])]}

def call_model2(state: MessagesState):
    messages = state['messages'][1].content
    response = gen_model1(messages, max_new_tokens=100)
    # print("Response from model2: ", response, "\n")
    return {"messages": [HumanMessage(content=response[0]['generated_text'])]}

def call_model3(state: MessagesState):
    messages = state['messages'][1].content
    response = gen_model2(messages, max_new_tokens=100)
    # print("Response from model3: ", response, "\n")
    return {"messages": [HumanMessage(content=response[0]['generated_text'])]}

# def call_orch(state: MessagesState):
#     messages = state['messages'][0].content + state['messages'][1].content

#     orchestrator_prompt = f"""
#     {messages}

#     Rules for response:
#     - Do not repeat the roles or question.
#     - Only respond with "Invoke agentX" where X is 1, 2, or 3.
#     """

#     response = gen_model_orch(orchestrator_prompt, max_new_tokens=10)

#     print("Response from model orch : ", response, "\n")
#     return {"messages": [HumanMessage(content=response[0]['generated_text']), state['messages'][1].content]}

def call_orch(state: MessagesState):
    messages = state['messages'][0].content + state['messages'][1].content

    orchestrator_prompt = f"""Based on the following roles and question, respond ONLY with "Invoke agent1", "Invoke agent2", or "Invoke agent3": {messages} OUTPUT:"""

    response = gen_model_orch(
        orchestrator_prompt, 
        max_new_tokens=10
    )
    
    # Extract just the agent invocation
    response_text = response[0]['generated_text']
    print(f"Response from model orch : {response_text} \n")
    if "invoke" in response_text.lower():
        agent_call = response_text.split("OUTPUT:")[-1].strip()
    else:
        agent_call = "Invoke agent1"  # fallback

    return {"messages": [HumanMessage(content=agent_call), state['messages'][1].content]}


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
# checkpointer = MemorySaver()

# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
print("Going to compile")
app = workflow.compile()
print("Done compile")

# Use the Runnable
# final_state = app.invoke(
#     {"messages": [HumanMessage(content="How to register car for DMV? Do not directly answer the question. Have either agent1 or agent2 or agent3 to process information. Invoke an agent.")]},
#     config={"configurable": {"thread_id": 42}}
# )

roles_str = "Agent1 is resonsible for General Vehicle Registration and Licensing. Agent2 is responsible for Fees, Taxes, and Financial Management. Agent3 is resposible for Special Vehicles, Plates, and Documentation. Do not answer the question directly. In your response, call the appropriate agent, either agent1, agent2 or agent3, based on the question. Your reponse should only contain Invoke agentx where x can be either 1,2 or 3. Question:"

#questions = ["Do I have to go in person to renew my driver's license?", "How much does it cost to register a car?", "What are the rules for license plates on my car?", "What kind of special vehicles are allowed on the roads?", "What are all the taxes for owning a car?"]
questions = ["What should a buyer do if they believe they paid use tax to a broker?", "What should I do after submitting my DMV 14 form for an address change?"]
# question = "How to register car for DMV?"
# question = "How much does it cost to register a car?"

for question in questions:
    # final_state = app.invoke(
    #     {"messages": [HumanMessage(content=roles_str), HumanMessage(content=question)]},
    #     config={"configurable": {"thread_id": 42}}
    # )
    #checkpointer.clear()
    print(f"\n Processing question: {question}")
    final_state = app.invoke(
        {"messages": [HumanMessage(content=roles_str), HumanMessage(content=question)]},
        config={"configurable": {"thread_id": 42, "callbacks": None}}
    )

    torch.cuda.empty_cache()

    print("Done invoking \n")
    print(final_state["messages"][-1].content)