"""Score pretraining checkpoints on the same fixed validation sample.

Run from reproduction/work/:  python ../eval_checkpoints.py NAME=PATH [NAME=PATH ...]
Uses GPT2_model.py's dataset and the token-weighted loss from its run_validation,
over a seeded sample of N_BATCHES batches so every checkpoint sees identical text.
Appends results to eval_results.json.
"""
import json
import math
import os
import random
import sys

import torch
import torch.utils._triton

# bitsandbytes (imported by GPT2_model) only needs Triton for training kernels.
torch.utils._triton.has_triton = lambda: False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GPT2_model import ChunkedStreamingTextDataset  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from torch.amp import autocast  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # noqa: E402

N_BATCHES = 200
BATCH_SIZE = 25

tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file("bpe_tokenizer.json"),
                                    pad_token="<PAD>", bos_token="<BOS>", eos_token="<EOS>")
config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=1024, n_ctx=1024, n_embd=768,
                    n_layer=12, n_head=12, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
                    layer_norm_epsilon=1e-5, initializer_range=0.02,
                    pad_token_id=tokenizer.pad_token_id,
                    scale_attn_by_inverse_layer_idx=True, reorder_and_upcast_attn=True)


def batches():
    random.seed(0)  # the dataset shuffles its file list with `random`
    ds = ChunkedStreamingTextDataset("cache/validation", tokenizer, chunk_size=1024)
    for i, batch in enumerate(DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)):
        if i >= N_BATCHES:
            break
        yield batch


@torch.no_grad()
def score(path):
    model = GPT2LMHeadModel(config)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda().eval()
    total_loss = total_tokens = 0
    for batch in batches():
        ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        with autocast(device_type="cuda"):
            loss = model(input_ids=ids, attention_mask=mask, labels=ids.masked_fill(mask == 0, -100)).loss
        n = mask.sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    avg = total_loss / total_tokens
    return {"global_step": ckpt["global_step"], "val_loss": avg, "perplexity": math.exp(avg),
            "tokens": total_tokens}


def main():
    results = json.load(open("eval_results.json")) if os.path.exists("eval_results.json") else {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        results[name] = score(path)
        print(name, results[name], flush=True)
        json.dump(results, open("eval_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
