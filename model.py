import copy
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from lion_pytorch import Lion
from torch.nn import utils
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from transformers import Adafactor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from transformers import GPT2Tokenizer
from torch.utils.data import IterableDataset, get_worker_info
from transformers import GPT2Config, GPT2LMHeadModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the pretrained GPT-2 tokenizer
sp = GPT2Tokenizer.from_pretrained("gpt2")
sp.pad_token = sp.eos_token

folder = "cache/epub"

max_len = 1024

config = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 512,
    "n_layer": 6,
    "n_head": 8,
    "n_inner": 2048,  # optional
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
    "initializer_range": 0.02
}

model_config = GPT2Config(**config)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1, max_seq_len=max_len):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # Precompute upper triangular causal mask (not registered as a parameter)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x, key_padding_mask=None, return_attn=False):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Optional mask of shape (batch_size, seq_len), where True indicates padding tokens.
        """
        seq_len = x.size(1)
        # Shape (seq_len, seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len].to(dtype=torch.bool, device=x.device)
        attn_mask = attn_mask.to(dtype=torch.bool, device=x.device)

        # Multihead attention with causal mask and optional padding mask
        attn_output, attn_weights = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False  # Return separate heads
        )

        # Apply residual connection + LayerNorm
        x = self.ln1(x + self.dropout(attn_output))

        # Feed-forward network with residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))

        return (x, attn_weights) if return_attn else x


class SmallGPT(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=256, num_heads=4, num_layers=4, ffn_hidden_dim=1024, max_seq_len=256,
                 dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # === Weight tying ===
        self.head.weight = self.token_embedding.weight

    def forward(self, x, key_padding_mask=None, return_attn=False):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
            key_padding_mask: Optional mask of shape (batch_size, seq_len)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)

        # Token + positional embeddings
        x = self.token_embedding(x) + self.pos_embedding(pos)

        # Pass through each transformer block
        attentions = []
        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, key_padding_mask=key_padding_mask, return_attn=True)
                attentions.append(attn)  # Shape: (batch_size, num_heads, seq_len, seq_len)
            else:
                x = layer(x, key_padding_mask=key_padding_mask)

        # Final normalization and projection
        x = self.ln_final(x)
        logits = self.head(x)
        return (logits, attentions) if return_attn else logits

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

# Tokenizer setup
def train_tokenizer(text_file, vocab_size=30000, model_prefix="tokenizer"):
    spm.SentencePieceTrainer.Train(
        input=text_file, model_prefix=model_prefix, vocab_size=vocab_size,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

def encode_text(sp, text):
    tokens = sp.encode(text, out_type=int)
    return torch.tensor(tokens, dtype=torch.long)  # No truncation

def decode_text(sp, token_ids):
    return sp.decode(token_ids)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, tokenized_data, seq_len=256):
        self.data = tokenized_data
        self.seq_len = seq_len

        # Debugging: Print dataset size
        if debug:
            print(f"Dataset size: {len(self.data)} tokens, Expected min: {self.seq_len + 1}")

            if len(self.data) < self.seq_len + 1:
                raise ValueError("Dataset is too small to form sequences!")

    def __len__(self):
        return len(self.data) - self.seq_len  # Number of training samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x.clone().detach(), y.clone().detach()

class ChunkedStreamingTextDataset(IterableDataset):
    def __init__(self, root_dir, chunk_size=1024, encoding='utf-8'):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.encoding = encoding

    def _get_all_files(self):
        files = []
        for root, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(".txt"):
                    files.append(os.path.join(root, fname))
        return sorted(files)  # consistent order

    def _file_chunk_generator(self, file_paths):
        for path in file_paths:
            with open(path, 'r', encoding=self.encoding) as f:
                buffer = ""
                for line in f:
                    buffer += line
                    while len(buffer) >= self.chunk_size:
                        yield sp(buffer[:self.chunk_size], return_tensors="pt", truncation=True, padding="max_length")
                        buffer = buffer[self.chunk_size:]
                # Optionally yield remaining buffer
                if buffer:
                    yield sp(buffer, return_tensors="pt", truncation=True, padding="max_length")

    def __iter__(self):
        worker_info = get_worker_info()
        all_files = self._get_all_files()

        # Multi-worker: split file list by worker id
        if worker_info is None:
            # Single worker
            assigned_files = all_files
        else:
            # Split file list among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            assigned_files = all_files[worker_id::num_workers]

        return self._file_chunk_generator(assigned_files)

# Load dataset
def load_dataset(text_file, sp, seq_len=256, batch_size=32):
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    tokenized_data = encode_text(sp, text)

    if debug:
        # Debugging: Print token count
        print(f"Total tokens in dataset: {len(tokenized_data)}")

    dataset = TextDataset(tokenized_data, seq_len)

    if debug:
        # Check dataset length before passing it to DataLoader
        print(f"Total samples in dataset: {len(dataset)}")

        if len(dataset) == 0:
            raise ValueError("Dataset is too small, no samples available for training.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def lr_finder(model, dataloader, device, min_lr=1e-6, max_lr=1, steps=100):
    use_autocast = 0
    model.train()

    # Save the initial state of the model to reset later
    initial_state = copy.deepcopy(model.state_dict())

    # Exponential LR schedule
    def lr_lambda(step):
        return min_lr * ((max_lr / min_lr) ** (step / steps))

    loss_values, lr_values = [], []

    for step, (x, y) in enumerate(dataloader):
        if step > steps:
            break
        x, y = x.to(device), y.to(device)

        # Reset model to initial state at each step
        #model.load_state_dict(initial_state)

        # Create a new optimizer each time with current LR
        lr = lr_lambda(step)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        optimizer.zero_grad()

        if use_autocast:
            with torch.amp.autocast(device_type=device if device == "cuda" else "cpu"):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        optimizer.step()

        lr_values.append(lr)
        loss_values.append(loss.item())
        print(f"Step {step+1}/{steps}, LR: {lr:.6f}, Loss: {loss.item():.4f}")

    # Plot loss vs learning rate
    plt.figure(figsize=(8, 6))
    plt.plot(lr_values, loss_values, marker="o", linestyle="-", color="b")
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Loss vs Learning Rate (Fixed Model Parameters)")
    plt.grid(True)
    plt.show()

    return lr_values, loss_values

def visualize_attention(attn_weights, step, layer_idx=0, head_idx=0):
    # attn_weights: (batch, num_heads, seq_len, seq_len)
    attn = attn_weights[layer_idx][0, head_idx].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(attn, cmap="viridis")
    plt.title(f"Layer {layer_idx}, Head {head_idx} Attention (Step {step})")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"attention_step{step}_layer{layer_idx}_head{head_idx}.png")
    plt.close()

def visualize_attention_tb(attn_weights, step, writer, layer_idx=0, head_idx=0):
    import numpy as np
    attn = attn_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    writer.add_image(
        f"Attention/Layer{layer_idx}_Head{head_idx}",
        attn[np.newaxis, :, :],  # Add channel dimension
        global_step=step,
        dataformats='CHW'
    )

# Training function
def train_model(model, dataloader, epochs=2, lr=3e-4, weight_decay=1e-2,
                accumulation_steps=4, device=None, log_dir="runs/small_gpt",
                log_attn_every=1000, log_param_histograms=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = Lion(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)  # Using Lion instead of AdamW
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=False, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(init_scale=512.0, growth_interval=2000, backoff_factor=0.5)

    writer = SummaryWriter(log_dir="runs/small_gpt")

    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        running_loss = 0.0
        global_steps = 0
        lowest_loss = float('inf')
        optimizer.zero_grad()

        for i, (x, y) in enumerate(dataloader):
            global_steps += 1
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)) / accumulation_steps

            # Check for NaN in loss (optional safeguard)
            if not torch.isfinite(loss):
                print(f"NaN detected at step {i + 1}, skipping step")
                scaler.update()
                continue

            scaler.scale(loss).backward()

            # Optional: log parameter histograms
            if log_param_histograms and global_steps % 500 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f"Weights/{name}", param, global_steps)
                        writer.add_histogram(f"Gradients/{name}", param.grad, global_steps)
                        writer.add_scalar(f"GradStats/{name}_mean", param.grad.mean(), global_steps)
                        writer.add_scalar(f"GradStats/{name}_std", param.grad.std(), global_steps)
                        writer.add_scalar(f"WtStats/{name}_mean", param.mean(), global_steps)
                        writer.add_scalar(f"WtStats/{name}_std", param.std(), global_steps)

            running_loss += loss.item()
            total_loss += loss.item()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                # Gradient clipping
                #scaler.unscale_(optimizer)  # Important: unscale before clipping
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # ✅ Log to TensorBoard
                writer.add_scalar("Loss/train", loss.item() * accumulation_steps, i + 1)
                #writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], i + 1)

            if global_steps % 100 == 0:
                avg_loss = running_loss * accumulation_steps / 100
                logger.info(f"Step {i + 1:,}/{len(dataloader):,}, Epoch {epoch + 1}/{epochs}, "
                            f"Loss: {avg_loss:.4f}")
                running_loss = 0.0

                if avg_loss < lowest_loss:
                    lowest_loss = avg_loss

            if global_steps % log_attn_every == 0:
                # Log attention maps for every layer & head
                with torch.no_grad():
                    logits, attns = model(x, return_attn=True)
                    for layer_idx, layer_attn in enumerate(attns):  # (batch, heads, seq, seq)
                        for head_idx in range(layer_attn.size(1)):
                            attn = layer_attn[0, head_idx].detach().cpu().numpy()
                            writer.add_image(
                                f"Attention/Layer{layer_idx}_Head{head_idx}",
                                attn[np.newaxis, :, :],  # (1, seq, seq)
                                global_step=i + 1,
                                dataformats='CHW'
                            )


            if global_steps % 5000 == 0:
                avg_loss = total_loss * accumulation_steps / (i + 1)
                if avg_loss < lowest_loss * 1.1:
                    print(f"Saving model, lowest_loss = {lowest_loss:.4f}")
                    torch.save(model.state_dict(), "small_gpt.pth")
                else:
                    print(f"Training loss did not improve significantly, lowest_loss = {lowest_loss:.4f}")

                with torch.no_grad():
                    model.eval()
                    prompt2 = 'Once upon a time'
                    text = generate_text(model, sp, prompt2, max_length=100, temperature=0.8, top_k=10)
                    writer.add_text("Sampled text", text, global_steps)
                    logger.info(text)
                    model.train()
                model.train()
                optimizer.zero_grad()

            if global_steps % 50_000 == 0:
                with torch.no_grad():
                    # Token embeddings
                    token_emb = model.token_embedding.weight.detach().cpu()
                    writer.add_embedding(token_emb, metadata=vocab,
                                         tag="TokenEmbedding", global_step=global_steps)

                    # Positional embeddings
                    pos_emb = model.pos_embedding.weight.detach().cpu()
                    writer.add_embedding(pos_emb, metadata=[f"pos_{i}" for i in range(pos_emb.size(0))],
                                         tag="PosEmbedding", global_step=global_steps)

                    writer.flush()

        avg_epoch_loss = total_loss * accumulation_steps / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar("Loss/epoch_avg", avg_epoch_loss, epoch)

    writer.close()
    print("Training complete. TensorBoard logs saved to:", log_dir)


# Text generation function
def generate_text(model, sp, prompt, max_length=50, temperature=1.0, top_k=10, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokens = encode_text(sp, prompt).unsqueeze(0).to(device)  # Add batch dimension
    generated_tokens = tokens.tolist()[0]

    for _ in range(max_length):
        logits = model(tokens)[:, -1, :]  # Get last token's logits
        logits = logits / temperature  # Apply temperature
        top_k_logits, top_k_indices = torch.topk(logits, top_k)  # Get top-k logits
        probs = F.softmax(top_k_logits, dim=-1)  # Convert to probabilities
        next_token = top_k_indices[0, torch.multinomial(probs, 1)]  # Sample from top-k
        generated_tokens.append(next_token.item())

        if next_token.item() == 3:  # EOS token
            break

        tokens = torch.tensor(generated_tokens, dtype=torch.long, device=device).unsqueeze(0)

    return decode_text(sp, generated_tokens)

def save_model(model, opt, scaler, path="small_gpt.pth"):
    checkpoint = {"model": model.state_dict(),
                  "optimizer": opt.state_dict(),
                  "scaler": scaler.state_dict()}
    # Write checkpoint as desired, e.g.,
    torch.save(checkpoint, "filename")
    print(f"Model saved to {path}")

def load_model(path="small_gpt.pth", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda(device))
    #net =
    #net.load_state_dict(checkpoint["model"])
    #opt.load_state_dict(checkpoint["optimizer"])
    #scaler.load_state_dict(checkpoint["scaler"])

    #return (net, opt, scaler)

load_tokenzier = 1
debug = 1
training_LR = 0
training_model = 0
text_generation = 0
#sp = spm.SentencePieceProcessor()

if __name__ == "__main__":
    dataset = ChunkedStreamingTextDataset(folder, chunk_size=1024)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4)

    if load_tokenzier:
        #sp = spm.SentencePieceProcessor()
        #sp.load("tokenizer.model")

        if debug:
            text = "Hello, how are you?"
            tokens = sp.tokenize(text)
            input_ids = sp.encode(text, return_tensors="pt")  # Returns PyTorch tensor

            print("Tokens:", tokens)
            print("Input IDs:", input_ids)

            for batch in dataloader:
                print(batch)  # list of text chunks
                input_ids = batch["input_ids"]
                for ids in input_ids:
                    print(sp.decode(ids.squeeze(), skip_special_tokens=True))
                    print("========================================================")
                input()

    if training_LR:
        #dataloader = load_dataset("tokenizer_input.txt", sp, seq_len=256, batch_size=84)
        #dataloader = load_dataset("tokenizer_input.txt", sp, seq_len=256, batch_size=24)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SmallGPT(vocab_size=30000).to(device)
        if os.path.exists("small_gpt.pth"):
            model.load_state_dict(torch.load("small_gpt.pth"))
        lr_finder(model, dataloader, device)

    if training_model:
        #dataloader = load_dataset("tokenizer_input.txt", sp, seq_len=256, batch_size=45)
        #dataloader = load_dataset("tokenizer_input.txt", sp, seq_len=128, batch_size=128)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SmallGPT(vocab_size=sp.vocab_size, max_seq_len=256).to(device)
        if os.path.exists("small_gpt.pth"):
            model.load_state_dict(torch.load("small_gpt.pth"))
        else:
            model.apply(initialize_weights)
        model = torch.compile(model, backend="inductor")
        #train_model(model, dataloader, epochs=10, lr=3e-4, accumulation_steps=2)
        train_model(model, dataloader, epochs=5, lr=3e-4, accumulation_steps=4)
        torch.save(model.state_dict(), "small_gpt.pth")  # Save model weights
        print("Model saved as small_gpt.pth")

    if text_generation:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SmallGPT(vocab_size=30000).to(device)
        model.load_state_dict(torch.load("small_gpt.pth"))
        prompt = 'Once upon a time'
        generated_text = generate_text(model, sp, prompt, max_length=100, temperature=0.8, top_k=10)
        print(generated_text)
