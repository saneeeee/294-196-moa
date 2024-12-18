# moa-task
Solving real world tasks reliably with mixture of agents 🕵️‍♂️

### Installation
1. Download dependencies: 
    - **TODO:** configure dependencies
    - there is a preliminary dependency list in environment.yml, you might want to change the name if you already have a `llama` conda environment.

2. run `pip install -e .` from root

### Getting datasets
Be careful when running website scraping jobs from Berkeley machines, some clusters explicitly ban this because UC blocks such machines.
You can assemble the dataset yourself by
- running `python3 moatask/data/webscrape.py "./dataset"`
or download pre-assembled dataset from
- `https://sc110kla.s3.us-west-1.amazonaws.com/dataset.zip`

once the web scraping is complete,
- simply download Q&A pairs fromm `https://sc110kla.s3.us-west-1.amazonaws.com/qa_pairs.zip`, or
- run `python3 moatask/sf_qa_generator.py` to generate sft Q&A pairs yourself

It is preferred to download supervised finetuning Q&A pairs from aws, unless you are tuning the generation

### Fintuning 
In order to run finetuning for the small agent you can simply run python finetune_pipeline/ finetune.py --agent[1 or 2 or 3 or 4]. Where 1 is the first agent, 2 is the second agent, 3 is the third agent and 4 is the big orchestrator. Please use the config.py to set the baseline model that you want to use right now it is set to default to be LlaMa 3.1 8B with 100 epoch. However, if you use the orchestrator please make sure that you change the base of the model to 70B in the config.py file in the model name variable. Additionally, that will also contain other hyperparameter that you can changed for the finetuning as well

### Running Eval 
In order to run the baseline (70B) eval simply run python3 finetune_pipeline/eval.py please make sure to change the train, test dataset path and also the model path and running that will give you a json file that store the result including other metric such as the question, target text, response, bleu and rouge score as well 

In order to run our architecture eval then you can run python3 finetune_pipeline/eval_architecture.py that will return the evaluation for our architecture which will hae the other json file that has the same stucutre as the baseline one.

### Getting the metric and inference time
This can be done by running all the cell in python jupyter notebook "eval_analysis.ipynb" that will generate the min,max and mean inference time and also min, max and mean of the METEOR score. 
(Note that the last cell will remove invalid results, run the last cell if you want to remove the invalid results and re-run the second to the last cell again to get the final result)


