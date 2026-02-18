import pandas as pd
from tqdm import tqdm
import json
import random
import numpy as np
import orjson
from collections import defaultdict

# --- CONFIGURATION ---
split = "test" 
# Adjust paths as needed
base_path = "./data/YAGO3-10"
input_file = f"{base_path}/event_{split}_paragraphs.jsonl"
output_file = f"{base_path}/{split}_gupdate_cascade.json"
stats_file = f"{base_path}/{split}_gupdate_cascade_stats.json"

# Heuristics to match ICEWS14 density
MAX_BRANCH_PER_HOP = 25    # Max neighbors to traverse per hop
MAX_SUBGRAPH_SIZE = 50     # Max edges in subgraph_before
MAX_NEIGHBORS_SEARCH = 100 # If a node connects to >100 events, sample them (avoids super-hubs)
# Set to None to process ALL data, or an integer to test (e.g., 5000)
MAX_CASCADES = 300        

print(f"Loading data from {input_file}...")
data_df = pd.read_json(path_or_buf=input_file, lines=True)
print(f"Loaded {len(data_df)} records.")

# ---------------------------------------------------------
# 1. OPTIMIZATION: Build Inverted Index
# ---------------------------------------------------------
# Map node_id -> list of event_indices where this node appears
# This turns Neighbor Search from O(N) to O(1)
print("Building inverted index for fast lookup...")
node_to_event_indices = defaultdict(list)

# We also cache the triggers for easy access later
triggers = {} 

for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    subj, rel, obj = row['event'][:3]
    
    # Store trigger info
    triggers[idx] = {
        "idx": idx,
        "triple": (subj, rel, obj),
        "nodes": {subj, obj}
    }
    
    # Update index
    node_to_event_indices[subj].append(idx)
    node_to_event_indices[obj].append(idx)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def get_neighbors_optimized(idx, exclude=None):
    """
    Finds temporally/structurally related events using the inverted index.
    """
    neighbors = set()
    row = data_df.iloc[idx]
    
    # Combine added and deleted edges to find connectivity
    edges = (row.get('added_edges', []) or []) + (row.get('deleted_edges', []) or [])
    
    # Collect potential neighbor indices based on node overlap
    potential_indices = set()
    
    for edge in edges:
        if len(edge) >= 3:
            s, r, o = edge[:3]
            
            # Look up events involving s
            if s in node_to_event_indices:
                s_events = node_to_event_indices[s]
                # If a node is a super-hub (e.g. connects to 1000 events), sample strictly
                if len(s_events) > MAX_NEIGHBORS_SEARCH:
                    potential_indices.update(random.sample(s_events, MAX_NEIGHBORS_SEARCH))
                else:
                    potential_indices.update(s_events)
            
            # Look up events involving o
            if o in node_to_event_indices:
                o_events = node_to_event_indices[o]
                if len(o_events) > MAX_NEIGHBORS_SEARCH:
                    potential_indices.update(random.sample(o_events, MAX_NEIGHBORS_SEARCH))
                else:
                    potential_indices.update(o_events)

    # Filter potential neighbors
    current_triple = triggers[idx]["triple"]
    
    for cand_idx in potential_indices:
        if cand_idx == idx:
            continue
        if exclude and cand_idx in exclude:
            continue
            
        cand = triggers[cand_idx]
        
        # Exact logic: Include if they share nodes or triples match
        # (Since we retrieved them by node lookup, they definitely share nodes)
        neighbors.add(cand_idx)
            
    return list(neighbors)

def record_to_list(idx):
    """
    Formats the data for the model, ensuring input sizes don't blow up memory.
    """
    row = data_df.iloc[idx]
    
    raw_sb = row['subgraph_before'] if row['subgraph_before'] else []
    raw_sa = row['subgraph_after'] if row['subgraph_after'] else []

    # Strategy: 
    # The 'subgraph_before' (context) should prioritize nodes that are actually 
    # modified in 'subgraph_after' (target).
    
    target_nodes = set()
    for s, r, o in raw_sa:
        target_nodes.add(s)
        target_nodes.add(o)
        
    final_sb = []
    other_sb = []
    
    for edge in raw_sb:
        s, r, o = edge[:3]
        if s in target_nodes or o in target_nodes:
            final_sb.append((s,r,o))
        else:
            other_sb.append((s,r,o))
            
    # Fill remaining slots up to limit
    remaining = MAX_SUBGRAPH_SIZE - len(final_sb)
    if remaining > 0:
        final_sb.extend(other_sb[:remaining])
        
    # Final safety clip
    final_sb = final_sb[:MAX_SUBGRAPH_SIZE]
    final_sa = raw_sa[:MAX_SUBGRAPH_SIZE] # Also clip output just in case

    return [row['event'], final_sb, final_sa, row['paragraph']]

def build_hops(seed_idx, max_hops=2):
    visited = {seed_idx}
    # Hop 0: The seed event itself
    hops = [[record_to_list(seed_idx)]]

    # --- Hop 1 ---
    hop1_idxs = get_neighbors_optimized(seed_idx, exclude=visited)
    
    # Downsample if too many neighbors (YAGO density control)
    if len(hop1_idxs) > MAX_BRANCH_PER_HOP:
        hop1_idxs = random.sample(hop1_idxs, MAX_BRANCH_PER_HOP)
    
    visited.update(hop1_idxs)
    hops.append([record_to_list(i) for i in hop1_idxs])

    # --- Hop 2 ---
    if max_hops >= 2:
        hop2_list = []
        for h1_idx in hop1_idxs:
            hop2_idxs = get_neighbors_optimized(h1_idx, exclude=visited)
            
            if len(hop2_idxs) > MAX_BRANCH_PER_HOP:
                hop2_idxs = random.sample(hop2_idxs, MAX_BRANCH_PER_HOP)
                
            visited.update(hop2_idxs)
            hop2_list.append([record_to_list(i) for i in hop2_idxs])
        
        hops.append(hop2_list)

    return hops, visited

# ---------------------------------------------------------
# 3. MAIN GENERATION LOOP
# ---------------------------------------------------------
all_hops = {}
visited_roots = set()
all_indices = list(range(len(data_df)))
random.shuffle(all_indices) # Shuffle to get random distribution if we stop early

print(f"Building cascades (Limit: {MAX_CASCADES if MAX_CASCADES else 'All'})...")

pbar = tqdm(all_indices)
for idx in pbar:
    if MAX_CASCADES and len(all_hops) >= MAX_CASCADES:
        break
        
    if idx in visited_roots:
        continue
        
    cascades, visited_in_cascade = build_hops(seed_idx=idx, max_hops=2)
    
    # Validation: Ensure cascade isn't empty (sanity check)
    if len(cascades) > 0:
        all_hops[str(idx)] = cascades
        visited_roots.update(visited_in_cascade)
        
    pbar.set_description(f"Cascades: {len(all_hops)}")

# ---------------------------------------------------------
# 4. SAVE AND STATS
# ---------------------------------------------------------
print(f"Saving {len(all_hops)} cascades to {output_file}...")
with open(output_file, "wb") as f:
    f.write(orjson.dumps(all_hops))

# Calculate Stats
hop1_lens = [len(x[1]) for x in all_hops.values() if len(x) > 1]
hop2_lens = [sum(len(h) for h in x[2]) for x in all_hops.values() if len(x) > 2]

stats = {
    "dataset": "YAGO3-10-Optimized",
    "total_cascades": len(all_hops),
    "avg_hop1_neighbors": float(np.mean(hop1_lens)) if hop1_lens else 0,
    "avg_hop2_neighbors": float(np.mean(hop2_lens)) if hop2_lens else 0,
}

print("\n--- Final Stats ---")
print(json.dumps(stats, indent=2))
with open(stats_file, "w") as f:
    json.dump(stats, f)