from collections import defaultdict, Counter
import json

def main():
    input_path = './data/YAGO3-10/train.txt'
    output_path = './data/YAGO3-10/train_with_inferred.txt'

    # Step 1: Load original triplets
    triplets = []
    rel_set = set()
    graph = defaultdict(list)

    with open(input_path, 'r') as f:
        for line in f:
            src, rel, tgt = line.strip().split('\t')
            triplets.append((src, rel, tgt))
            rel_set.add(rel)
            graph[src].append((rel, tgt))

    print(f"Original triplets: {len(triplets)}")
    print(f"Unique relations in original KG: {len(rel_set)}")

    # Step 2: Define 2-hop inference rules
    with open('./data/YAGO3-10/inference_rules.json') as f:
        rules = json.load(f)
    # rules = json.load('./inference_rules.json') 

    # Step 3: Apply rules to generate inferred triplets
    inferred = set()
    relation_counter = Counter()

    for h1, r1, t1 in triplets:
        if t1 not in graph:
            continue
        for r2, t2 in graph.get(t1, []):
            for rule in rules:
                rel1, rel2 = rule['path'][0], rule['path'][1]
                new_rel, mode = rule['new_relation'], rule['mode']
                if r1 == rel1 and r2 == rel2:
                    if mode == "forward":
                        inferred.add((h1, new_rel, t2))
                        relation_counter[new_rel] += 1
                    elif mode == "cross":
                        inferred.add((h1, new_rel, t2))
                        inferred.add((t2, new_rel, h1))
                        relation_counter[new_rel] += 2

    print(f"Inferred triplets: {len(inferred)}")
    print("\nBreakdown of inferred triplets by relation:")
    for rel, count in relation_counter.items():
        print(f"  {rel}: {count}")

    # Step 4: Write combined output
    with open(output_path, 'w') as f:
        for triplet in triplets:
            f.write('\t'.join(triplet) + '\n')
        for triplet in inferred:
            f.write('\t'.join(triplet) + '\n')

    print(f"\nFinal KG written to: {output_path}")
    print(f"Total triplets in final KG: {len(triplets) + len(inferred)}")

if __name__ == "__main__":
    main()