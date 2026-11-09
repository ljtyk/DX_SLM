# DX_SLM — learning language models by building one

This is my first project learning how language models work. I started by writing a
small transformer by hand, then scaled up to pretraining a GPT-2-sized model on
Project Gutenberg, fine-tuned it into a chat model, and finally tried QLoRA on a
larger pretrained model.

The code is published as I left it: a learning log, not a library. It has rough
edges, and I have listed the ones I know about [below](#known-issues) rather than
tidying them away.

**On AI assistance:** I wrote most of the early code myself and gradually introduced
a chatbot assistant as the project went on. The later scripts have more assisted
code than the earlier ones.

## The four stages

| Stage | Files | What it does |
|---|---|---|
| 1. Transformer from scratch | `model.py`, `train_tokenizer.py` | A hand-written `SmallGPT` in PyTorch: causal self-attention, learned positional embeddings, weight tying. Includes a learning-rate finder, attention-map logging to TensorBoard, and experiments with AdamW, Lion and Adafactor. `train_tokenizer.py` cleans Project Gutenberg text, splits off a validation set, and trains a SentencePiece tokenizer. |
| 2. Pretraining a GPT-2-sized model | `tokenizer.py`, `GPT2_model.py` | Trains a custom 12,000-token byte-level BPE tokenizer, then pretrains a 12-layer, 768-wide GPT-2 (about 95M parameters) from random initialisation on Gutenberg books. Uses mixed precision, 8-bit AdamW, gradient accumulation to an effective batch of about 512, gradient clipping, checkpoint resume, and periodic validation perplexity and sample generation. |
| 3. Chat fine-tuning | `SFT_train.py`, `run_interact.py` | Fine-tunes the pretrained model on `Human:` / `Assistant:` dialogues, and adds a small experiment I call "gradient fuzzing" (uniform noise proportional to each gradient's magnitude). `run_interact.py` is a terminal chat loop for the result. |
| 4. QLoRA on a larger model | `SFT.py`, `SFT_run.py` | 4-bit QLoRA fine-tune of GPT-Neo 1.3B on the Alpaca instruction dataset using PEFT and TRL, run for 3 epochs (39,003 steps). `SFT_run.py` runs inference with the adapter. |

## What I learned along the way

- How a decoder-only transformer is put together, by writing the blocks myself
  before switching to the Hugging Face implementation.
- Why tokenizer choice matters: I moved from SentencePiece to the GPT-2 tokenizer to
  a custom 12k BPE vocabulary sized for the data.
- Fitting training on a single consumer GPU: mixed precision, 8-bit optimizers,
  gradient accumulation, and streaming the dataset from disk instead of loading it.
- The difference between pretraining, full fine-tuning, and parameter-efficient
  fine-tuning (LoRA on a quantised model).

## Running it

The scripts expect to be run from the repository root, because they load files by
relative path. A CUDA GPU is assumed.

```bash
pip install -r requirements.txt
```

Rough order:

1. `train_tokenizer.py` — clean the Gutenberg text under `cache/` (toggle the flags near the bottom of the file).
2. `tokenizer.py` — train the BPE tokenizer, producing `bpe_tokenizer.json` (included in this repo).
3. `GPT2_model.py` — pretrain; writes checkpoints to `checkpoints/`.
4. `SFT_train.py` — chat fine-tune; writes the model to `aligned-model/`.
5. `run_interact.py` — chat with the fine-tuned model.
6. `SFT.py` then `SFT_run.py` — the separate QLoRA experiment.

Model weights and datasets are not in this repository (see below), so steps 3 and 4
need to be run before step 5 will work.

## Data

No datasets are redistributed here. The scripts expect:

- **Project Gutenberg** plain-text books under `cache/` (pretraining and tokenizer training).
- **[Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)** as `train.jsonl` (first chat fine-tuning attempt).
- A ShareGPT-format conversation dataset as `train-00000-of-00001.parquet` (later chat fine-tuning).
- **[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)**, downloaded automatically by `SFT.py`.

## Known issues

Things I would fix or do differently now:

- `model.py` was left mid-refactor: it loads the GPT-2 tokenizer but parts of the
  training and generation code still call SentencePiece methods, and `save_model`
  writes to a literal file named `filename`.
- `GPT2_model.py` has `import matplotlib as plt` (it should be `matplotlib.pyplot`);
  it only goes unnoticed because the attention-plotting loop never runs. Training
  also uses a constant learning rate with no warmup or decay.
- `SFT_train.py` computes loss on padding tokens instead of masking them, and clips
  gradients without unscaling them first under mixed precision. The saved model
  config keeps GPT-2's default end-of-sequence id (50256) even though the
  vocabulary has 12,000 tokens; `run_interact.py` works around this by passing the
  id explicitly.
- `SFT.py` evaluates on a 64-example subset of its own training data, so the
  evaluation loss is not a held-out measure. It also computes loss on padding.
- Configuration is done with flags and constants inside each file rather than
  command-line arguments.

## Licence

MIT — see [LICENSE](LICENSE).
