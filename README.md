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
