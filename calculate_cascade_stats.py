import json
import os
import argparse
import numpy as np

def is_target_event(item):
    # Check if item is a list of strings (and maybe a timestamp)
    if not isinstance(item, list):
        return False
    if not item:
        return False
    # Check if first element is string
    if isinstance(item[0], str):
        return True
    return False

def extract_samples(data_structure):
    """
    recursively yield samples from the data structure.
    A sample is a list [Target, Context, ...]
    """
    if not isinstance(data_structure, list):
        return

    # Check if this list IS a sample
    # A sample starts with a Target Event (list of strings)
    if len(data_structure) >= 2 and is_target_event(data_structure[0]):
        yield data_structure
        return

    # Otherwise, iterate and recurse
    for item in data_structure:
        if isinstance(item, list):
            yield from extract_samples(item)

def calculate_stats(data_path, dataset_name):
    splits = ['train', 'valid', 'test']
    
    print(f"\n=== {dataset_name} Statistics ===")
    
    global_entity_set = set()
    global_relation_set = set()
    
    for split in splits:
        file_path = os.path.join(data_path, f"{split}_gupdate_cascade.json")
        if not os.path.exists(file_path):
            print(f"Skipping {split}: File not found at {file_path}")
            continue
            
        print(f"Processing {split}...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        sample_count = 0
        subgraph_sizes = []
        split_entity_set = set()
        split_relation_set = set()
        
        # Iterate over all values in the dict
        for key, value in data.items():
            # Value is expected to be a list of samples (or nested lists)
            for sample in extract_samples(value):
                sample_count += 1
                
                target_event = sample[0]
                # Collect entities/relations from target
                try:
                    h, r, t = target_event[0], target_event[1], target_event[2]
                    split_entity_set.add(h)
                    split_entity_set.add(t)
                    split_relation_set.add(r)
                except Exception as e:
                    # print(f"Error parsing target {target_event}: {e}")
                    pass
                
                # Contexts are at index 1 and maybe 2
                # We assume any subsequent element that is a list of lists (triples) is a context
                contexts = []
                for i in range(1, len(sample)):
                    comp = sample[i]
                    # Check if it looks like a subgraph (list of lists)
                    if isinstance(comp, list) and (len(comp) == 0 or isinstance(comp[0], list)):
                         # It's a subgraph
                         contexts.append(comp)
                
                # Calculate stats for contexts
                # For avg size, if there are multiple subgraphs, do we sum them or average them?
                # Previous script summed "subgraph_before" size.
                # Here if we have multiple, we probably sum them as "total context".
                
                context_size = 0
                for ctx in contexts:
                    context_size += len(ctx)
                    for triple in ctx:
                        if len(triple) >= 3:
                            th, tr, tt = triple[0], triple[1], triple[2]
                            split_entity_set.add(th)
                            split_entity_set.add(tt)
                            split_relation_set.add(tr)
                            
                subgraph_sizes.append(context_size)

        global_entity_set.update(split_entity_set)
        global_relation_set.update(split_relation_set)
        
        print(f"  Samples: {sample_count}")
        print(f"  Unique Entities (split): {len(split_entity_set)}")
        print(f"  Unique Relations (split): {len(split_relation_set)}")
        if subgraph_sizes:
            print(f"  Avg Subgraph Size: {np.mean(subgraph_sizes):.2f} +/- {np.std(subgraph_sizes):.2f}")
        else:
            print(f"  Avg Subgraph Size: 0")

    print(f"\nOverall:")
    print(f"  Total Unique Entities: {len(global_entity_set)}")
    print(f"  Total Unique Relations: {len(global_relation_set)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--icews_path", type=str, default="./data/icews14")
    parser.add_argument("--yago_path", type=str, default="./data/YAGO3-10")
    args = parser.parse_args()

    calculate_stats(args.icews_path, "ICEWS14")
    calculate_stats(args.yago_path, "YAGO3-10")
