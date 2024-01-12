import os
import torch
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Replace with your file path
file_path = '/home/mc9432/train.json'

dataset = []

with open(file_path, 'r') as file:
    for line in file:
        try:
            dataset.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")



def process_data(data):
    # Extract 'src' and 'tgt' texts
    sources = [item['src'] for item in data]
    targets = [item['tgt'] for item in data]
    return sources, targets

# Load and process the data
sources, targets = process_data(dataset)

# Assuming 'sources' and 'targets' are lists extracted from your dataset
hf_dataset = Dataset.from_dict({'src': sources, 'tgt': targets})

# Replace with your file path
file_path2 = '/home/mc9432/val.json'

dataset_val = []

with open(file_path2, 'r') as file:
    for line in file:
        try:
            dataset_val.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")



def process_data(data):
    # Extract 'src' and 'tgt' texts
    sources = [item['src'] for item in data]
    targets = [item['tgt'] for item in data]
    return sources, targets

# Load and process the data
sources, targets = process_data(dataset_val)

# Assuming 'sources' and 'targets' are lists extracted from your dataset
hf_val_dataset = Dataset.from_dict({'src': sources, 'tgt': targets})

model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-bbcnews-summary"
model_v5 = "llama-2-7b-v5"

torch.cuda.empty_cache()

# Load tokenizer and model with QLoRA configuration
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

max_seq_length = None
packing = False
device_map = {"": 0}
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

OUTPUT_DIR = "/home/mc9432/results"
LOGS_DIR = "/home/mc9432/log"

num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        #per_device_train_batch_size=per_device_train_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
        logging_dir=LOGS_DIR,
        save_safetensors=True,
    )

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    eval_dataset=hf_val_dataset,
    peft_config=peft_config,
    dataset_text_field="src",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save trained model
trainer.save_model(OUTPUT_DIR)
#trainer.model.save_pretrained(OUTPUT_DIR)
print("successfully saved the model")


## Reload model in FP16 and merge it with LoRA weights
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    low_cpu_mem_usage=True,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")




     

