import json
from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm

def load_kg(file_path):
    """Load the ICEWS KG into a dictionary of adjacency lists indexed by timestamp."""
    kg = defaultdict(lambda: defaultdict(list))
    
    with open(file_path, 'r') as f:
        for line in f:
            s, r, tgt, t = line.strip().split('\t')
            t_datetime = datetime.strptime(t, "%Y-%m-%d")
            kg[t_datetime][s].append((r, tgt))  # Store outgoing edges
            kg[t_datetime][tgt].append((r, s))  # Store incoming edges for undirected search
    
    return kg

def extract_2hop_subgraph(kg, s, tgt, t):
    """Extracts the 2-hop subgraph around s and tgt at timestamp t."""
    t_datetime = datetime.strptime(t, "%Y-%m-%d")

    if t_datetime not in kg:
        return set()  # Return an empty set if no data is available for this timestamp

    one_hop_neighbors = set()
    one_hop_edges = set()

    for entity in [s, tgt]:
        if entity in kg[t_datetime]:
            for r, neighbor in kg[t_datetime][entity]:
                one_hop_neighbors.add(neighbor)
                one_hop_edges.add((entity, r, neighbor, t))

    return one_hop_edges

def extract_collapsed_2hop_subgraph(kg, s, tgt, start_t, days=5):
    """Extracts the collapsed 2-hop subgraph over the next 'days' days."""
    start_date = datetime.strptime(start_t, "%Y-%m-%d")
    merged_edges = set()

    for i in range(days):
        t_datetime = start_date + timedelta(days=i)
        subgraph = extract_2hop_subgraph(kg, s, tgt, t_datetime.strftime("%Y-%m-%d"))
        merged_edges.update(subgraph)

    return merged_edges

def compute_delta(before, after):
    """Computes the delta between two subgraphs."""
    added_edges = after - before
    removed_edges = before - after
    return added_edges, removed_edges

def main():
    kg = load_kg("./data/icews14/test.txt")
    output_file = "./data/icews14/added_edges_test.jsonl"
    
    with open(output_file, 'w') as jsonl_file:
        with open("./data/icews14/test.txt", 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing records"):  # Process all records
                s, r, tgt, t = line.strip().split('\t')
                subgraph_before = extract_2hop_subgraph(kg, s, tgt, t)
                subgraph_after = extract_collapsed_2hop_subgraph(kg, s, tgt, t, days=5)

                added_edges, _ = compute_delta(subgraph_before, subgraph_after)
                
                if len(added_edges) > 20:  # Only store if added_edges is larger than 20
                    json_data = {
                        "s": s,
                        "r": r,
                        "tgt": tgt,
                        "t": t,
                        "added_edges": list(added_edges)  # Convert set to list for JSON serialization
                    }

                    jsonl_file.write(json.dumps(json_data) + '\n')
                
if __name__ == "__main__":
    main()