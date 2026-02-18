import json
from collections import Counter

def compute_icews_event_stats(jsonl_path):
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f]

    entity_set = set()
    relation_set = set()
    subgraph_sizes = []
    added_edge_counts = []
    deleted_edge_counts = []

    for entry in lines:
        # Count all entities and relations in subgraphs
        for h, r, t in entry["subgraph_before"] + entry["subgraph_after"]:
            entity_set.update([h, t])
            relation_set.add(r)

        # Count all entities and relations in added/removed edges
        added = entry["edge_diff"].get("added", [])
        removed = entry["edge_diff"].get("removed", [])
        added_edge_counts.append(len(added))
        deleted_edge_counts.append(len(removed))
        for h, r, t in added + removed:
            entity_set.update([h, t])
            relation_set.add(r)

        # Track subgraph size (only using subgraph_before to avoid duplication)
        subgraph_sizes.append(len(entry["subgraph_before"]))

    print("\n=== ICEWS-Event Dataset Statistics ===")
    print(f"Total Events: {len(lines)}")
    print(f"Unique Entities: {len(entity_set)}")
    print(f"Unique Relation Types: {len(relation_set)}")
    print(f"Average Subgraph Size: {sum(subgraph_sizes)/len(subgraph_sizes):.2f}")
    print(f"Average Added Edges: {sum(added_edge_counts)/len(added_edge_counts):.2f}")
    print(f"Average Deleted Edges: {sum(deleted_edge_counts)/len(deleted_edge_counts):.2f}")

# Run
compute_icews_event_stats('./data/icews14/train_gupdate_rule.jsonl')
