import json

data_path = './data/NBATransaction/NBAtransactions_train.json'

if __name__ == '__main__':
    with open(data_path, 'r') as f:
        obj = json.load(f)

    for data in obj:
        
     
    print("First item:", obj[0])
    print("Keys in the first item:", list(obj[0].keys()))
    print(obj[0]['text'])
