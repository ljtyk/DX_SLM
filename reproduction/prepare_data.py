"""Clean the downloaded books and split off a validation set, using train_tokenizer.py's own functions.

Run from reproduction/work/ after download_gutenberg.py:  python ../prepare_data.py
  cache/epub/      raw downloads (English only, already filtered by the catalog)
  cache/cleaned/   headers/footers stripped, whitespace normalised   -> training data
  cache/validation 180 MB of cleaned books moved out of cache/cleaned -> validation data
"""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_tokenizer import clean_books_in_directory, collect_txt_files, move_validation_set  # noqa: E402

random.seed(0)  # make the validation split reproducible

if not os.path.isdir("cache/cleaned"):
    clean_books_in_directory("cache/epub", "cache/cleaned")

if not os.path.isdir("cache/validation"):
    # Same 180 MB target as the validation step in train_tokenizer.py
    move_validation_set(collect_txt_files("cache/cleaned"), 180 * 1024 * 1024, "cache/validation")


def summarise(path):
    files = collect_txt_files(path)
    size = sum(os.path.getsize(f) for f in files)
    print(f"{path}: {len(files):,} books, {size / 1e9:.2f} GB")


summarise("cache/cleaned")
summarise("cache/validation")
