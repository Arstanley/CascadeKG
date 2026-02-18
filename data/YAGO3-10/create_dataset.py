from collections import defaultdict, Counter
import json
import copy
from tqdm import tqdm

def load_triplets(path):
    triplets = []
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triplets.append((h, r, t))
            graph[h].append((r, t))
            reverse_graph[t].append((r, h))  # <-- reversed!
    return triplets, graph, reverse_graph

def load_rules(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_1hop_subgraph_for_pair(h, t, graph, reverse_graph):
    subgraph = []

    # Outgoing edges
    subgraph.extend([[h, r, o] for r, o in graph.get(h, [])])
    subgraph.extend([[t, r, o] for r, o in graph.get(t, [])])

    # Incoming edges (now fast)
    subgraph.extend([[src, r, h] for r, src in reverse_graph.get(h, [])])
    subgraph.extend([[src, r, t] for r, src in reverse_graph.get(t, [])])

    return subgraph

def apply_rule(h1, r1, t1, r2, t2, rule):
    new_rel = rule['new_relation']
    mode = rule['mode']
    new_edges = []
    if mode == 'forward':
        new_edges.append([h1, new_rel, t2])
    elif mode == 'cross':
        new_edges.extend([[h1, new_rel, t2], [t2, new_rel, h1]])
    return new_edges

def generate_event_updates(triplets, graph, rules, reverse_graph):
    merged_updates = {}
    all_triplet_set = set(triplets)

    for h1, r1, t1 in tqdm(triplets[:50000]):
        for r2, t2 in graph.get(t1, []):
            # Fire all rules that match (r1, r2)
            fired_rules = [rule for rule in rules if rule['path'] == [r1, r2]]
            if not fired_rules:
                continue

            inferred_edges_set = set()
            for rule in fired_rules:
                inferred_edges_set.update(tuple(edge) for edge in apply_rule(h1, r1, t1, r2, t2, rule))
            inferred_edges = list(inferred_edges_set)

            for base_triple in [[h1, r1, t1], [t1, r2, t2]]:
                h_e, r_e, t_e = base_triple
                base_triple_tuple = tuple(base_triple)

                # Extract local subgraph
                local_subgraph_set = set(
                    tuple(x) for x in get_1hop_subgraph_for_pair(h_e, t_e, graph, reverse_graph)
                )


                # Prepare both add and delete updates
                for event_type in ["add", "delete"]:
                    key = (base_triple_tuple, event_type)

                    if key not in merged_updates:
                        merged_updates[key] = {
                            "event": base_triple,
                            "event_type": event_type,
                            "added_edges": [],
                            "deleted_edges": [],
                            "subgraph_before": set(),
                            "subgraph_after": set()
                        }

                    entry = merged_updates[key]

                    if event_type == "add":
                        entry["added_edges"].extend(inferred_edges)
                        entry["subgraph_before"].update(local_subgraph_set)
                        entry["subgraph_after"].update(local_subgraph_set | {base_triple_tuple} | set(inferred_edges))
                    else:
                        entry["deleted_edges"].extend(inferred_edges)
                        entry["subgraph_before"].update(local_subgraph_set | {base_triple_tuple} | set(inferred_edges))
                        entry["subgraph_after"].update(local_subgraph_set)

    # Finalize merged updates
    updates = []
    for entry in merged_updates.values():
        updates.append({
            "event": entry["event"],
            "event_type": entry["event_type"],
            "added_edges": sorted([list(x) for x in set(entry["added_edges"])]),
            "deleted_edges": sorted([list(x) for x in set(entry["deleted_edges"])]),
            "subgraph_before": sorted([list(x) for x in entry["subgraph_before"]]),
            "subgraph_after": sorted([list(x) for x in entry["subgraph_after"]])
        })

    return updates


def compute_dataset_stats(updates):
    total = len(updates)
    add_events = sum(1 for x in updates if x['event_type'] == 'add')
    delete_events = total - add_events

    avg_added = sum(len(x['added_edges']) for x in updates) / total
    avg_deleted = sum(len(x['deleted_edges']) for x in updates) / total
    avg_subgraph_before = sum(len(x['subgraph_before']) for x in updates) / total
    avg_subgraph_after = sum(len(x['subgraph_after']) for x in updates) / total

    rel_counter = Counter()
    entity_set = set()
    relation_set = set()

    for x in updates:
        for h, r, t in x['added_edges'] + x['deleted_edges'] + x['subgraph_before'] + x['subgraph_after']:
            rel_counter[r] += 1
            relation_set.add(r)
            entity_set.add(h)
            entity_set.add(t)

    print(f"\n==== Dataset Statistics ====")
    print(f"Total event examples: {total}")
    print(f"  Add events: {add_events}")
    print(f"  Delete events: {delete_events}")
    print(f"Average #added_edges: {avg_added:.2f}")
    print(f"Average #deleted_edges: {avg_deleted:.2f}")
    print(f"Average subgraph size BEFORE: {avg_subgraph_before:.2f}")
    print(f"Average subgraph size AFTER:  {avg_subgraph_after:.2f}")
    print(f"Unique relations: {len(relation_set)}")
    print(f"Unique entities:  {len(entity_set)}")
    print(f"\nTop 10 most frequent inferred relations:")
    for rel, count in rel_counter.most_common(10):
        print(f"  {rel}: {count}")


def main():
    input_path = './data/YAGO3-10/train.txt'
    rules_path = './data/YAGO3-10/inference_rules.json'
    output_path = './data/YAGO3-10/event_triggered_kg_updates.jsonl'

    triplets, graph, reverse_graph = load_triplets(input_path)
    rules = load_rules(rules_path)

    updates = generate_event_updates(triplets, graph, rules, reverse_graph)

    # with open(output_path, 'w') as f:
    #     for example in updates:
    #         f.write(json.dumps(example) + '\n')

    # print(f"\nSaved {len(updates)} event examples to: {output_path}")
    compute_dataset_stats(updates)

if __name__ == "__main__":
    main()
