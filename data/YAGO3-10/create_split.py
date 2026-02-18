import json
import random

input_path = './data/YAGO3-10/event_triggered_kg_updates_paragraph.jsonl'
train_path = './data/YAGO3-10/event_train_paragraphs.jsonl'
valid_path = './data/YAGO3-10/event_valid_paragraphs.jsonl'
test_path  = './data/YAGO3-10/event_test_paragraphs.jsonl'

# Set seed for reproducibility
random.seed(42)

# Load all examples
with open(input_path, 'r') as f:
    examples = [json.loads(line) for line in f]

# Shuffle
random.shuffle(examples)

# Split
n = len(examples)
n_train = int(0.8 * n)
n_valid = int(0.1 * n)
n_test  = n - n_train - n_valid  # ensures total = n

train_set = examples[:n_train]
valid_set = examples[n_train:n_train + n_valid]
test_set  = examples[n_train + n_valid:]

# Save splits
with open(train_path, 'w') as f:
    for ex in train_set:
        f.write(json.dumps(ex) + '\n')

with open(valid_path, 'w') as f:
    for ex in valid_set:
        f.write(json.dumps(ex) + '\n')

with open(test_path, 'w') as f:
    for ex in test_set:
        f.write(json.dumps(ex) + '\n')

print(f"Total: {n} → train: {len(train_set)}, valid: {len(valid_set)}, test: {len(test_set)}")
print(f"Saved to:\n  {train_path}\n  {valid_path}\n  {test_path}")
