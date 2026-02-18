import json

entity2id_path = './data/NBATransaction/entity2id.txt'
relation2id_path = './data/NBATransaction/relation2id.txt'

id2ent = {}
id2rel = {}

with open(entity2id_path, 'r') as f:
    for line in f.readlines()[1:]:
        entity, id = line.strip().split('\t')
        id2ent[int(id)] = entity 

with open(relation2id_path, 'r') as f:
    for line in f.readlines()[1:]:
        relation, id = line.strip().split('\t')
        id2rel[int(id)] = relation 

print(id2ent)
print(id2rel)

with open('./data/NBATransaction/id2entity.txt', 'w') as f:
    # f.write('entity\tid\n')  # Write header
    for id, entity in id2ent.items():
        f.write(f'{id}\t{entity}\n')

with open('./data/NBATransaction/id2relation.txt', 'w') as f:
    # f.write('relation\tid\n')  # Write header
    for relation, id in id2rel.items():
        f.write(f'{id}\t{relation}\n')