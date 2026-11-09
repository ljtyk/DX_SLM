import string

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
import os


def collect_txt_files_recursively(root_dir):
    txt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(dirpath, f))
    return txt_files

# --- Setup ---
data_dir = "cache/"
files = collect_txt_files_recursively(data_dir)

# --- Init tokenizer ---
tokenizer = Tokenizer(BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

# --- Trainer ---
trainer = BpeTrainer(
    vocab_size=12000,  # change as needed
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

# --- Train ---
tokenizer.train(files, trainer)

# --- Save ---
tokenizer.save("bpe_tokenizer.json")
