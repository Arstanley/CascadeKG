from unsloth import FastLanguageModel, is_bfloat16_supported
from prompts import alpaca_prompt
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json

# Constants
INSTRUCTION = "Write a short description of the given event."
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True
MODEL_DIR = "lora_model_icews"  # path where you saved the finetuned LoRA model

# --- Load the finetuned model & tokenizer ---
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
def format_prompt(trigger_event: list) -> str:
    """
    Build an Alpaca-style prompt from the trigger_event list,
    which is assumed to be [subject, verb, object, ...].
    """
    # Join the first three elements into a simple sentence
    subj, verb, obj = trigger_event[:3]
    input_text = f"{subj} {verb} {obj}"
    # Leave the "output" slot empty for the model to fill in
    return alpaca_prompt().format(INSTRUCTION, input_text, "")

# --- Inference function ---
def generate_description(
    trigger_event: list,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    do_sample: bool = True
) -> str:
    # 1) Build the prompt
    prompt = format_prompt(trigger_event)

    # 2) Tokenize
    inputs = tokenizer(
        prompt,
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
    return resp.split(EOS_TOKEN)[0].strip()

def generate_and_save(dataset, split_name, output_path):
    """
    Generate descriptions for a dataset split and save them to JSONL.
    """
    output_file = f'./data/icews14/{split_name}_gupdate_rule_paragraphs_finetuned.jsonl'
    # output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in tqdm(dataset["train"], desc=f"Generating {split_name}"):
            trigger_event = entry.get("trigger_event")
            if not trigger_event or len(trigger_event) < 3:
                continue
            try:
                prediction = generate_description(trigger_event)
                result = {
                    "trigger_event": trigger_event,
                    "generated_paragraph": prediction
                }
                f_out.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"Error generating for event {trigger_event}: {e}")
                continue

if __name__ == "__main__":
    OUTPUT_DIR = "outputs_ft_icews/generation"
    train_dataset = load_dataset("json", data_files="./data/icews14/train_gupdate_rule_paragraphs.jsonl")
    eval_dataset = load_dataset("json", data_files="./data/icews14/valid_gupdate_rule_paragraphs.jsonl")
    test_dataset = load_dataset("json", data_files="./data/icews14/test_gupdate_rule.jsonl")
    generate_and_save(train_dataset, "train", OUTPUT_DIR)
    generate_and_save(eval_dataset, "valid", OUTPUT_DIR)  # 'train' is key in both cases from HuggingFace dataset
    generate_and_save(test_dataset, "test", OUTPUT_DIR)