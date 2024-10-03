# moa-task
Solving real world tasks reliably with mixture of agents 🕵️‍♂️

### Installation
1. Download dependencies: 
    - **TODO:** configure dependencies

2. run `pip install -e` from root

### Getting datasets
Be careful when running website scraping jobs from Berkeley machines, some clusters explicitly ban this because UC blocks such machines.
You can assemble the dataset yourself by
- running `python3 moatask/data/webscrape.py "./dataset"`
or download pre-assembled dataset from
- `https://sc110kla.s3.us-west-1.amazonaws.com/dataset.zip`