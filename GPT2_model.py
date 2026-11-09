import os
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.amp import autocast, GradScaler
from datetime import datetime
from bitsandbytes.optim import AdamW8bit
import matplotlib as plt
import random
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


class ChunkedStreamingTextDataset(IterableDataset):
    def __init__(self, root_dir, tokenizer, chunk_size=1024, encoding='utf-8'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.encoding = encoding
        self.file_list = self._get_all_files()

    def _get_all_files(self):
        files = []
        for root, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(".txt"):
                    files.append(os.path.join(root, fname))
        random.shuffle(files)
        return files

    def _file_chunk_generator(self, file_paths):
        for path in file_paths:
            try:
                with open(path, 'r', encoding=self.encoding, errors='ignore') as f:
                    text = f.read()  # Optional char limit
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    for i in range(0, len(tokens), self.chunk_size):
                        chunk = tokens[i:i + self.chunk_size]
                        if not chunk:
                            continue
                        input_ids = chunk + [self.tokenizer.pad_token_id] * (self.chunk_size - len(chunk))
                        attention_mask = [1] * len(chunk) + [0] * (self.chunk_size - len(chunk))
                        yield {
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
                        }
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            assigned_files = self.file_list
        else:
            assigned_files = self.file_list[worker_info.id::worker_info.num_workers]
        return self._file_chunk_generator(assigned_files)

def add_gradient_fuzzing_uniform(model, fuzz_pct=0.05):
    with torch.no_grad():
        for (name, param) in model.named_parameters():
            if param.grad is None or "wpe" in name.lower():
                continue

            # Uniform noise in [-fuzz_pct * |grad|, +fuzz_pct * |grad|]
            scale = fuzz_pct * param.grad.abs()
            noise = (torch.rand_like(param.grad) * 2 - 1) * scale
            param.grad += noise

@torch.no_grad()
def run_validation(model, val_dataloader, tokenizer, writer, global_step):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, batch in enumerate(val_dataloader):
        if i >= 20:  # limit to 20 batches for speed, adjust as needed
            break
        input_ids = batch["input_ids"].to("cuda", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)

        with autocast(device_type='cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.masked_fill(attention_mask == 0, -100)
            )
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    writer.add_scalar("Loss/val", avg_loss, global_step)
    print(f"📊 Validation | Loss: {avg_loss:.4f} | Perplexity: {perplexity.item():.2f}")
    model.train()

# --- Main Training ---
def main():
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Tokenizer ---
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #tokenizer.pad_token = tokenizer.eos_token
    tokenizer_2 = Tokenizer.from_file("bpe_tokenizer.json")

    # Wrap it
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_2,
        pad_token="<PAD>",
        bos_token="<BOS>",
        eos_token="<EOS>",
    )

    # --- Model Config ---
    config = {
        "vocab_size": tokenizer.vocab_size,
        "n_positions": 1024,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "pad_token_id": tokenizer.pad_token_id,
        "scale_attn_by_inverse_layer_idx": True,
        "reorder_and_upcast_attn": True
    }
    model = GPT2LMHeadModel(GPT2Config(**config)).cuda()
    folder = "cache/cleaned"

    batchsize = 25
    # --- Dataset & Dataloader ---
    dataset = ChunkedStreamingTextDataset(folder, tokenizer, chunk_size=config["n_ctx"])
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=0, pin_memory=True)

    val_folder = "cache/validation"  # Point to your ~180MB validation set
    val_dataset = ChunkedStreamingTextDataset(val_folder, tokenizer, chunk_size=config["n_ctx"])
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, num_workers=0, pin_memory=False)

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
            "weight_decay": 1e-2,
        },
        {
            "params": [param for name, param in model.named_parameters() if name in no_decay],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW8bit(param_groups, lr=3e-4)
    #optimizer = Adafactor(
    #    model.parameters(),
    #    scale_parameter=True,
    #    relative_step=True,
    #    warmup_init=True,
    #    lr=None  # Important: Let the scheduler handle learning rate
    #)
    #lr_scheduler = AdafactorSchedule(optimizer)

    torch.set_float32_matmul_precision('high')

    scaler = GradScaler('cuda')
    accumulation_steps = 512 // batchsize + 1
    print(f"{accumulation_steps = }")

    # --- TensorBoard Logging ---
    log_dir = f"runs/gpt2_gutenberg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # --- Training Loop ---
    model.train()
    global_step = 0

    if os.path.exists(checkpoint_path):
        print(f"🔁 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = checkpoint["global_step"]
        print(f"✅ Resumed from step {global_step}")

    model.compile()
    #torch.autograd.set_detect_anomaly(True)

    for i in range(10):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)

            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids.masked_fill(attention_mask == 0, -100)
                    #,output_attentions = True  # 👈 Add this
                )
                #attentions = outputs.attentions  # This is a tuple of (layers x batch x heads x seq x seq)
                loss = outputs.loss / accumulation_steps


            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                # 👇 Clip gradients before stepping
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Fuzz gradients uniformly ±5%
                #add_gradient_fuzzing_uniform(model, fuzz_pct=0.1)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                #lr_scheduler.step()  # 👈 Scheduler step

            if global_step % ((100 // accumulation_steps + 1) * accumulation_steps) == 0:
                perplexity = torch.exp(loss * accumulation_steps)
                writer.add_scalar("Loss/train", loss.item() * accumulation_steps, global_step)
                print(f"Step {global_step:,} | Loss: {loss.item() * accumulation_steps:.4f} | Perplexity: {perplexity.item():.2f}")

            if global_step % ((1000 // accumulation_steps + 1) * accumulation_steps) == 0:
                #print("Sample Input:", tokenizer.decode(input_ids[0][:100]))
                attentions = []
                for layer_idx, layer_attn in enumerate(attentions):
                    for head_idx in range(layer_attn.shape[1]):  # (batch_size, num_heads, seq_len, seq_len)
                        fig, ax = plt.subplots(figsize=(4, 4))
                        heatmap = layer_attn[0, head_idx].detach().cpu().numpy()
                        ax.imshow(heatmap, cmap='viridis', aspect='auto')
                        ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
                        writer.add_figure(f"Attention/L{layer_idx}_H{head_idx}", fig, global_step)
                        plt.close(fig)

                for name, param in model.named_parameters():
                    param = param.detach().cpu()
                    writer.add_histogram(f"Weights/{name}", param, global_step)
                    writer.add_scalar(f"WtStats/{name}_mean", param.mean(), global_step)
                    writer.add_scalar(f"WtStats/{name}_std", param.std(), global_step)

            if global_step % ((5000 // accumulation_steps + 1) * accumulation_steps) == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "global_step": global_step,
                }, checkpoint_path)
                print(f"💾 Saved checkpoint at step {global_step}")

                run_validation(model, val_dataloader, tokenizer, writer, global_step)

                model.eval()
                prompt = "Once upon a time"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    sample_output = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=100,
                        do_sample=True,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.95,
                        #pad_token_id=tokenizer.eos_token_id
                    )

                generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                print(f"\n📝 Sample at step {global_step}:\n{generated_text}\n")
                writer.add_text("Samples/OnceUponATime", generated_text, global_step)
                model.train()

                # Log token embeddings
                #embedding_weights = model.transformer.wte.weight.detach().cpu()[:500]
                #metadata = [tokenizer.decode([i]) for i in range(embedding_weights.shape[0])]
                #writer.add_embedding(embedding_weights, metadata=metadata, tag="TokenEmbeddings", global_step=global_step)

            global_step += 1

    writer.close()
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
    }, os.path.join(checkpoint_dir, "final.pt"))

if __name__ == "__main__":
    main()