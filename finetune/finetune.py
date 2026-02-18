import os
from unsloth import FastLanguageModel
from tqdm import tqdm
import json
from prompts import alpaca_prompt
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset

INSTRUCTION = "Write a short description of the given event."

# Reference: https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=2eSvM9zX_2d3
max_seq_length = 4096
dtype = None
load_in_4bit = True

fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit"            # Gemma 2x faster!
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token=os.environ.get('HF_TOKEN', '')
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

EOS_TOKEN = tokenizer.eos_token

# Prepare dataset 
train_dataset = load_dataset("json", data_files="./data/icews14/train_gupdate_rule_paragraphs.jsonl")
eval_dataset = load_dataset("json", data_files="./data/icews14/valid_gupdate_rule_paragraphs.jsonl")

def formatting_prompts_func(datapoint):
    # Extract and format the trigger event
    trigger = datapoint.get("trigger_event", None)
    if trigger is None:
        return {"text": None}

    # trigger is expected to be a list like [subject, verb, object, date]
    if isinstance(trigger, list) and len(trigger) == 4:
        subj, verb, obj, date = trigger
        input_text = f"{subj} {verb} {obj}"
    else:
        # Fallback: join whatever is there
        input_text = " ".join(map(str, trigger))

    # Extract the reference paragraph as the output
    output_text = datapoint.get("paragraph", None)
    if output_text is None:
        return {"text": None}

    # Format using the Alpaca-style prompt
    prompt = alpaca_prompt().format(INSTRUCTION, input_text, output_text)
    return {"text": prompt}

# Map and filter
texts_train = train_dataset.map(formatting_prompts_func).filter(lambda x: x["text"] is not None)
texts_eval = eval_dataset.map(formatting_prompts_func).filter(lambda x: x["text"] is not None)

# train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = texts_train['train'],
    eval_dataset=texts_eval['train'], # TODO: NEED TO UPDATE
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 50, # Set this for 1 full training run.
        max_steps = 500,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_ft_implicit",
        do_eval=True
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("lora_model_icews") # Local saving
tokenizer.save_pretrained("lora_model_icews")