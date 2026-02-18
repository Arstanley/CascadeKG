import openai
import pandas as pd
import json
import time
from tqdm import tqdm
import torch
from torch_geometric.data import Data

def loadid(datapath):
    ret = {}
    with open(datapath) as f:
        lines = f.readlines()
        for line in lines[1:]:
            cur = line.strip('\n').split("\t")
            token, id = cur[0], int(cur[1])
            ret[id] = token
    return ret

id2rel = loadid('./data/NBATransaction/relation2id.txt')
id2entity = loadid('./data/NBATransaction/entity2id.txt')

def triplets_to_graph(triplets):
    """
        As the name says   
    """
    src = [triplet[0] for triplet in triplets]
    tgt = [triplet[2] for triplet in triplets]
    edge_types = torch.tensor([triplet[1] for triplet in triplets])

    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    data = Data( edge_index=edge_index, edge_type=edge_types) 

    return data

def triplets_to_language(triplets):
    return f"{id2entity[triplets[0]]} {id2rel[triplets[1]]} {id2entity[triplets[2]]}"

# Load JSONL data into a Pandas DataFrame
file_name = "./data/NBATransaction/NBAtransactions_test.json"
with open(file_name, "r") as f:
    jsonObj = json.load(f)

# OpenAI API setup
api_key = "sk-423dc1c07533488b8fee62319b6a083e"
client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# Function to query OpenAI API
def get_direct_results(direct_change_triplets_add, direct_change_triplets_delete, added_triplets, deleted_triplets):
    """Sends a query to the OpenAI API to find direct results of an event."""
    
    # for t in direct_change_triplets_add:
    #     print(triplets_to_language(t))

    # event_str = f'"({trigger_event[0], trigger_event[1], trigger_event[2]})"'

    prompt = f"""Write a short paragraph to describe the following NBA Transaction:
    {{
        "News": Adding {[triplets_to_language(t) for t in direct_change_triplets_add]}; Deleting {[triplets_to_language(t) for t in direct_change_triplets_delete]},
        "Changes to Ensure Consistency": Adding {[triplets_to_language(t) for t in added_triplets]},
        Deleting {[triplets_to_language(t) for t in deleted_triplets]}
    }}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing events."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        # Extract response text and parse the events
        output = response.choices[0].message.content.strip()
        return output  # Returning the raw response for now; can be processed further

    except Exception as e:
        print(f"Error: {e}")
        return []

# Output file path
output_file = "./data/NBATransaction/test_gupdate_paragraphs.jsonl"

# Open the file in append mode to write results line by line
with open(output_file, "a") as f:
    for d in tqdm(jsonObj, total=len(jsonObj)):
        subgraph_before = d['subgraph_before']
        subgraph_before_pyG = triplets_to_graph(subgraph_before)

        subgraph_after = d['subgraph_after']
        subgraph_after_pyG = triplets_to_graph(subgraph_after)

        entities = d['text_mentioned_entities']

        # Convert lists to sets for set operations
        G1_set = set(tuple(triplet) for triplet in subgraph_before)
        G2_set = set(tuple(triplet) for triplet in subgraph_after)

        # Calculate added and deleted triplets
        added_triplets = G2_set - G1_set
        deleted_triplets = G1_set - G2_set
        
        direct_change_triplets_add = []
        direct_change_triplets_del = []

        # Get the direct change triplets (IE-GOLD)
        direct_change_triplets_add = []
        direct_change_triplets_del = []
        for triplet in added_triplets:
            if triplet[0] in entities and triplet[2] in entities:
                direct_change_triplets_add.append(triplet) 
        for triplet in deleted_triplets:
            if triplet[0] in entities and triplet[2] in entities:
                direct_change_triplets_del.append(triplet)
         
        response = get_direct_results(direct_change_triplets_add, direct_change_triplets_del, added_triplets, deleted_triplets)

        # Prepare the new JSON object with the result
        # new_entry = d.to_dict()
        d["paragraph"] = response

        # Write it to the file line by line
        f.write(json.dumps(d) + "\n")

        time.sleep(1)  # To avoid hitting API rate limits

print(f"Results saved to {output_file}")