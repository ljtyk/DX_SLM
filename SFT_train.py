import ast
import re

import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, get_scheduler, PreTrainedTokenizerFast, AutoTokenizer
import json
import os
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp import autocast, GradScaler

# --- Config ---
MODEL_PATH = "checkpoints/latest.pt"  # Your .pt file
TOKENIZER_PATH = "aligned-tokenizer"  # Directory containing tokenizer files
DATA_PATH = "train-00000-of-00001.parquet"
CONTEXT_LEN = 1024
EPOCHS = 10
BATCH_SIZE = 12
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Tokenizer ---
tokenizer_2 = Tokenizer.from_file("bpe_tokenizer.json")

# Wrap it
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_2,
    pad_token="<PAD>",
    bos_token="<BOS>",
    eos_token="<EOS>",
)

# --- Load Model ---
from transformers import GPT2Config
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=CONTEXT_LEN,
    n_ctx=CONTEXT_LEN,
    n_embd=768,
    n_layer=12,
    n_head=12,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    pad_token_id=tokenizer.pad_token_id,
    scale_attn_by_inverse_layer_idx=True,
    reorder_and_upcast_attn=True
)
model = GPT2LMHeadModel(config)
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)

# In dataset loader

# --- Dataset ---
class HH_RLHF_Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.samples = []
        eos = tokenizer.eos_token_id

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                dialogue = obj["chosen"]

                # Extract all Human–Assistant pairs
                pairs = re.findall(r'(Human:.*?)(Assistant:.*?)(?=Human:|$)', dialogue, re.DOTALL)

                for human, assistant in pairs:
                    text = (human + assistant).strip()

                    tokens = tokenizer(
                        text,
                        truncation=True,
                        max_length=max_len - 1,  # Reserve space for EOS
                        return_tensors='pt',
                        add_special_tokens=False
                    )

                    input_ids = tokens["input_ids"].squeeze(0)
                    attention_mask = tokens["attention_mask"].squeeze(0)

                    # Append EOS
                    input_ids = torch.cat([input_ids, torch.tensor([eos])])
                    attention_mask = torch.cat([attention_mask, torch.tensor([1])])

                    # Pad if needed
                    pad_len = max_len - input_ids.shape[0]
                    if pad_len > 0:
                        input_ids = torch.cat([input_ids, torch.full((pad_len,), eos)])
                        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])

                    if input_ids.shape[0] == max_len:
                        self.samples.append({
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["input_ids"].clone()
        }

class ParquetChatDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=CONTEXT_LEN):
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        df = pd.read_parquet(parquet_path)
        self.examples = self._parse_conversations(df)

    def _parse_conversations(self, df):
        examples = []
        #print(df["conversations"])
        for convo in df["conversations"]:
            if convo is None:
                continue
            #print(convo)
            # If stored as string instead of list, parse it
            if isinstance(convo, str):
                convo = ast.literal_eval(convo)

            dialog = []
            human = 0
            gpt = 0
            for turn in convo:
                role = turn.get("from")
                text = turn.get("value")
                if role == "human":
                    dialog.append(f"Human: {text}")
                    human = 1
                elif role == "gpt":
                    dialog.append(f"Assistant: {text}")
                    gpt = 1
                if human and gpt:
                    human = 0
                    gpt = 0
                    examples.append("\n\n".join(dialog))
                    dialog = []
        return examples
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        #print(text)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=self.max_length - 1,  # Reserve space for EOS
            return_tensors='pt',
            add_special_tokens=False
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Append EOS
        input_ids = torch.cat([input_ids, torch.tensor([self.eos])])
        attention_mask = torch.cat([attention_mask, torch.tensor([1])])

        # Pad if needed
        pad_len = self.max_length - input_ids.shape[0]
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])

        #if input_ids.shape[0] == self.max_length:
        #    self.samples.append({
        #        "input_ids": input_ids,
        #        "attention_mask": attention_mask
        #    })

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For causal LM loss
        }



#dataset = HH_RLHF_Dataset(DATA_PATH, tokenizer, CONTEXT_LEN)
#loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dataset = ParquetChatDataset(DATA_PATH, tokenizer=tokenizer, max_length=CONTEXT_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Optimizer & Scheduler ---
# --- Optimizer, FP16, Accum ---
decay = set()
no_decay = set()
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.endswith(".bias") or "ln_" in name.lower() or "layernorm" in name.lower() or "wpe" in name.lower():
        no_decay.add(name)
    else:
        decay.add(name)

param_groups = [
    {
        "params": [param for name, param in model.named_parameters() if name in decay],
        "weight_decay":
            1e-3,
    },
    {
        "params": [param for name, param in model.named_parameters() if name in no_decay],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW8bit(param_groups, lr=LR)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=int(EPOCHS * len(loader) * 0.03),
                             num_training_steps=EPOCHS * len(loader))

scaler = GradScaler()


def add_gradient_fuzzing_uniform(model, fuzz_pct=0.10):
    with torch.no_grad():
        for (name, param) in model.named_parameters():
            if param.grad is None or "wpe" in name.lower():
                continue

            # Uniform noise in [-fuzz_pct * |grad|, +fuzz_pct * |grad|]
            scale = fuzz_pct * param.grad.abs()
            noise = (torch.rand_like(param.grad) * 2 - 1) * scale
            param.grad += noise


# --- Training Loop ---
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    for i, batch in enumerate(loop):
        #print(batch)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss


        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Fuzz gradients uniformly ±5%
        add_gradient_fuzzing_uniform(model, fuzz_pct=0.1)

        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        #optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())
        if i % 500 == 0:
            model.eval()
            prompt = "Human: Why is the sky blue?\n\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                sample_output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_length=1024,
                    do_sample=True,
                    temperature=0.8,
                    top_k=20,
                    top_p=0.95,
                )

            generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
            print(f"\n📝 Sample at step {i}:\n{generated_text}\n")
            model.train()
            # --- Save ---
            os.makedirs("aligned-model", exist_ok=True)
            model.save_pretrained("aligned-model")
            tokenizer.save_pretrained("aligned-model")

os.makedirs("aligned-model", exist_ok=True)
model.save_pretrained("aligned-model")
tokenizer.save_pretrained("aligned-model")