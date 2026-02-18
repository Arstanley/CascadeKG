import os
import json

def generate_mapping_files():
    # 1. Define your file paths
    base_path = "./data/YAGO3-10/"
    files = [
        os.path.join(base_path, "train_with_inferred.txt"),
        os.path.join(base_path, "valid_with_inferred.txt"),
        os.path.join(base_path, "test_with_inferred.txt")
    ]

    # Sets to store unique entities and relations
    entities = set()
    relations = set()

    print("Reading data files...")

    # 2. Read all files and extract unique items
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace and split by tabs or spaces
                parts = line.strip().split()
                
                # Ensure the line has at least 3 parts (Head, Relation, Tail)
                if len(parts) >= 3:
                    head = parts[0]
                    relation = parts[1]
                    tail = parts[2]
                    
                    entities.add(head)
                    entities.add(tail)
                    relations.add(relation)

    # 2.5 Read inference rules to add new relations
    inference_rules_path = "./data/YAGO3-10/inference_rules.json"
    if os.path.exists(inference_rules_path):
        print(f"Reading {inference_rules_path}...")
        try:
            with open(inference_rules_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
                count_added = 0
                for rule in rules:
                    if "new_relation" in rule:
                        new_rel = rule["new_relation"]
                        print(new_rel)
                        if new_rel not in relations:
                            relations.add(new_rel)
                            count_added += 1
                print(f"Added {count_added} new relations from inference rules.")
        except Exception as e:
            print(f"Error reading inference rules: {e}")
    else:
        print(f"Warning: Inference rules file not found: {inference_rules_path}")

    print(f"Found {len(entities)} unique entities.")
    print(f"Found {len(relations)} unique relations.")

    # 3. Sort them to ensure deterministic IDs (always the same ID for the same data)
    sorted_entities = sorted(list(entities))
    sorted_relations = sorted(list(relations))

    # 4. Write entity2id.txt
    ent_output_path = os.path.join(base_path, "entity2id.txt")
    print(f"Writing {ent_output_path}...")
    with open(ent_output_path, 'w', encoding='utf-8') as f:
        for idx, entity in enumerate(sorted_entities):
            f.write(f"{entity}\t{idx}\n")

    # 5. Write relation2id.txt
    # Changed from rel2id.txt to relation2id.txt to match existing file structure
    rel_output_path = os.path.join(base_path, "relation2id.txt")
    print(f"Writing {rel_output_path}...")
    with open(rel_output_path, 'w', encoding='utf-8') as f:
        for idx, relation in enumerate(sorted_relations):
            f.write(f"{relation}\t{idx}\n")

    print("Done! Files generated.")

if __name__ == "__main__":
    generate_mapping_files()
