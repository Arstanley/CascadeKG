from unsloth import FastLanguageModel, is_bfloat16_supported
from prompts import alpaca_prompt
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json
import torch
from torch_geometric.data import Data

"=================== Helper Functions ==================="
def loadid(datapath):
    ret = {}
    with open(datapath) as f:
        lines = f.readlines()
        for line in lines[1:]:
            cur = line.strip('\n').split("\t")
            token, id = cur[0], int(cur[1])
            ret[id] = token
    return ret

id2rel = loadid('./data/NBATransaction/relation2id.txt')
id2entity = loadid('./data/NBATransaction/entity2id.txt')
def triplets_to_graph(triplets):
    """
        As the name says   
    """
    src = [triplet[0] for triplet in triplets]
    tgt = [triplet[2] for triplet in triplets]
    edge_types = torch.tensor([triplet[1] for triplet in triplets])

    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    data = Data( edge_index=edge_index, edge_type=edge_types) 

    return data

def triplets_to_language(triplets):
    return f"{id2entity[triplets[0]]} {id2rel[triplets[1]]} {id2entity[triplets[2]]}"

def get_direct_change_triplets(datapoint):
    subgraph_before = datapoint['subgraph_before']
    subgraph_after = datapoint['subgraph_after']

    entities = datapoint['text_mentioned_entities']

    # Convert lists to sets for set operations
    G1_set = set(tuple(triplet) for triplet in subgraph_before)
    G2_set = set(tuple(triplet) for triplet in subgraph_after)

    # Calculate added and deleted triplets
    added_triplets = G2_set - G1_set
    deleted_triplets = G1_set - G2_set
    
    direct_change_triplets_add = []
    direct_change_triplets_del = []

    # Get the direct change triplets (IE-GOLD)
    direct_change_triplets_add = []
    direct_change_triplets_del = []
    for triplet in added_triplets:
        if triplet[0] in entities and triplet[2] in entities:
            direct_change_triplets_add.append(triplet) 
    for triplet in deleted_triplets:
        if triplet[0] in entities and triplet[2] in entities:
            direct_change_triplets_del.append(triplet)
    
    return direct_change_triplets_add, direct_change_triplets_del
"======================================================="

INSTRUCTION = "Write a short description of the given NBA transaction."

MODEL_DIR = "lora_model_nbatransaction"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
    # token=...  # only if your checkpoint is private
)

EOS_TOKEN = tokenizer.eos_token
EOS_TOKEN_ID = tokenizer.eos_token_id

# --- Prompt formatting ---
def format_prompt(datapoint):
    # Extract and format the trigger event
    direct_change_triplets_add, direct_change_triplets_del = get_direct_change_triplets(datapoint)
    direct_added_triplets_verbal = []
    direct_deleted_triplets_verbal = []

    for triplet in direct_change_triplets_add:
        direct_added_triplets_verbal.append(triplets_to_language(triplet))
    for triplet in direct_change_triplets_del:
        direct_deleted_triplets_verbal.append(triplets_to_language(triplet))
        
    # print(direct_added_triplets_verbal)
    # print(direct_deleted_triplets_verbal)

    input_text = f"News: Adding {[triplets_to_language(t) for t in direct_change_triplets_add]}; Deleting {[triplets_to_language(t) for t in direct_change_triplets_del]}" 

    # Extract the reference paragraph as the output
    # output_text = datapoint.get("paragraph", None)

    # if output_text is None:
    #     return {"text": None}

    # Format using the Alpaca-style prompt
    prompt = alpaca_prompt().format(INSTRUCTION, input_text, "")
    return {"text": prompt}, direct_change_triplets_add, direct_change_triplets_del

# --- Inference function ---
def generate_description(
    datapoint: list,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    do_sample: bool = True
) -> str:
    # 1) Build the prompt
    prompt, direct_change_triplets_add, direct_change_triplets_del = format_prompt(datapoint)

    # 2) Tokenize
    inputs = tokenizer(
        prompt["text"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    # 3) Generate token IDs
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        eos_token_id=EOS_TOKEN_ID,
    )

    # 4) Decode to text
    full = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

    # 5) Strip off the prompt prefix
    if "### Response:" in full:
        resp = full.split("### Response:")[-1]
    else:
        resp = full[len(prompt):]

    # 6) Remove trailing EOS
    return resp.split(EOS_TOKEN)[0].strip(), direct_change_triplets_add, direct_change_triplets_del

def generate_and_save(dataset, split_name, output_path):
    """
    Generate descriptions for a dataset split and save them to JSONL.
    """
    output_file = f'./data/NBATransaction/{split_name}_gupdate_rule_paragraphs_finetuned.jsonl'
    # output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in tqdm(dataset["train"], desc=f"Generating {split_name}"):
            prediction, direct_change_triplets_add, direct_change_triplets_del = generate_description(entry)
            print(prediction)
            print(direct_change_triplets_add)
            print(direct_change_triplets_del)
            result = {
                "direct_added": direct_change_triplets_add,
                "direct_deleted": direct_change_triplets_del,
                "generated_paragraph": prediction
            }
            f_out.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    OUTPUT_DIR = "outputs_ft_NBATransaction/generation"
    train_dataset = load_dataset("json", data_files="./data/NBATransaction/train_gupdate_paragraphs.jsonl")
    eval_dataset = load_dataset("json", data_files="./data/NBATransaction/valid_gupdate_paragraphs.jsonl")
    test_dataset = load_dataset("json", data_files="./data/NBATransaction/test_gupdate_paragraphs.jsonl")
    generate_and_save(train_dataset, "train", OUTPUT_DIR)
    generate_and_save(eval_dataset, "valid", OUTPUT_DIR)  # 'train' is key in both cases from HuggingFace dataset
    generate_and_save(test_dataset, "test", OUTPUT_DIR)

