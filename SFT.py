from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig, TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import default_data_collator
import torch

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_sample(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}<|endoftext|>"
    }

dataset = dataset.map(format_sample)

# Tokenizer and model
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Tokenization + label prep
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt, interval_steps=100, max_new_tokens=100):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.interval_steps = interval_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval_steps == 0:
            model = kwargs["model"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()

            input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.7
                )
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"\n[Step {state.global_step}] Sample Output:\n{decoded}\n")
            model.train()

tokenized_dataset = dataset.map(tokenize, batched=True)
eval_dataset = tokenized_dataset.select(range(64))

# Custom callback
sample_prompt = "### Instruction:\nWhy is the sky blue?\n\n### Response:\n"
sample_callback = SampleGenerationCallback(tokenizer, sample_prompt, interval_steps=100)

# Training args
training_args = TrainingArguments(
    output_dir="./sft-neo-qlora",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=1,
    remove_unused_columns=False,
    report_to="tensorboard"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    #tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    callbacks=[sample_callback],
)

# Train
trainer.train()
trainer.save_model("./sft-neo-qlora-final")
