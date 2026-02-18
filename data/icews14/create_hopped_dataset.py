import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json

split = "test"
data_path = f"./data/icews14"
paragraph_file = f"./data/icews14/{split}_gupdate_rule_paragraphs_finetuned.jsonl"
if split == 'test':
    data_file = f'{data_path}/{split}_gupdate_rule.jsonl' 
else:
    data_file = f'{data_path}/{split}_gupdate_rule_paragraphs.jsonl' 

paragraph_df = pd.read_json(path_or_buf=paragraph_file, lines=True)
data_df = pd.read_json(path_or_buf=data_file, lines=True)
print(len(paragraph_df), len(data_df))

def parse_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except Exception:
        return None

# Precompute triggers
triggers = []
for idx, row in data_df.iterrows():
    trig = row['trigger_event']
    subj, rel, obj, date = trig[:4]
    triggers.append({
        "idx": idx,
        "triple": (subj, rel, obj),
        "nodes": {subj, obj},
        "date": parse_date(date)
    })

def get_neighbors(idx, exclude=None):
    neighbors = set()
    row = data_df.iloc[idx]
    edge_diff = row['edge_diff']
    edges = (edge_diff.get('added', []) or []) + (edge_diff.get('removed', []) or [])
    for edge in edges:
        subj, rel, obj = edge
        edge_nodes = {subj, obj}
        for trg in triggers:
            if trg["idx"] == idx or trg["triple"] is None:
                continue
            if (subj, rel, obj) == trg["triple"] or (edge_nodes & trg["nodes"]):
                if exclude is None or trg["idx"] not in exclude:
                    neighbors.add(trg["idx"])
    return neighbors

def record_to_list(idx):
    row = data_df.iloc[idx]
    print(len(paragraph_df))
    paragraph = paragraph_df.iloc[idx]['generated_paragraph']
    return [row['trigger_event'], row['subgraph_before'], row['subgraph_after'], paragraph]

def build_hops(seed_idx, max_hops=2):
    visited = {seed_idx}
    hops = [[record_to_list(seed_idx)]]

    # Hop 1
    hop1_idxs = list(get_neighbors(seed_idx, exclude=visited))
    visited.update(hop1_idxs)
    hops.append([record_to_list(i) for i in hop1_idxs])

    if max_hops >= 2:
        hop2 = []
        for h1_idx in hop1_idxs:
            hop2_idxs = list(get_neighbors(h1_idx, exclude=visited))
            visited.update(hop2_idxs)
            hop2.append([record_to_list(i) for i in hop2_idxs])
        hops.append(hop2)

    return hops, visited

# -----------------------
# Build cascades without duplicate roots
# -----------------------
all_hops = {}
visited_roots = set()

for idx in tqdm(range(len(data_df))):
    if idx in visited_roots:
        continue  # already part of another cascade

    hops, visited = build_hops(seed_idx=idx, max_hops=2)
    all_hops[idx] = hops

    # mark all nodes found in this cascade as "covered"
    visited_roots.update(visited)

# Example: check the hop2 structure for the first record
print(all_hops[list(all_hops.keys())[0]][2])

# Save cascades
output_file = f"./data/icews14/{split}_gupdate_cascade.json"
with open(output_file, "w") as f:
    json.dump(all_hops, f)
