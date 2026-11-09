import os
import random
import re
import sentencepiece as spm
import csv
import shutil

def clean_text(text):
    """Removes Gutenberg headers, footers, and excessive whitespace."""
    text = re.sub(r"\*\*\* START OF (.*?) \*\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*\* END OF (.*?) \*\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"[\r\n]+", "\n", text)  # Normalize newlines
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text.strip()

def prepare_tokenizer_data(folder_path, output_file="tokenizer_input.txt"):
    """Iterates through all .txt files and writes cleaned text to a single file."""
    with open(output_file, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    print(f"{os.path.getsize(output_file):,}")
                    if os.path.getsize(output_file) > 2 ** 30 / 2:
                        return
                    with open(file_path, "r", encoding="utf-8") as in_f:
                        text = in_f.read()
                        out_f.write(clean_text(text) + "\n")

def train_tokenizer(text_file="tokenizer_input.txt", vocab_size=30000, model_prefix="tokenizer"):
    """Trains a SentencePiece tokenizer using the prepared text file."""
    spm.SentencePieceTrainer.Train(
        input=text_file, model_prefix=model_prefix, vocab_size=vocab_size,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        max_sentence_length=10_000_000
    )
    print("Tokenizer training completed.")

def old_start():
    # Prepare text and train tokenizer
    folder_path = "epub/"
    if not os.path.exists("tokenizer_input.txt"):
        prepare_tokenizer_data(folder_path)  # Create tokenizer_input.txt
    train_tokenizer()  # Train tokenizer

SOURCE_DIR = "./cache/validation"
DEST_DIR = "./cache/validation2"

def clean_gutenberg_text(text):
    """
    Removes Project Gutenberg headers/footers and cleans up the text.
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Match the core content
    start_re = re.compile(r"\*\*\* *START OF.*?\*\*\*", re.IGNORECASE)
    end_re = re.compile(r"\*\*\* *END OF.*?\*\*\*", re.IGNORECASE)

    start_match = start_re.search(text)
    end_match = end_re.search(text)

    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    elif start_match:
        text = text[start_match.end():]
    elif end_match:
        text = text[:end_match.start()]

    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse large gaps
    text = re.sub(r"[\r\n]+", "\n", text)  # Normalize newlines
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    text = text.strip()

    return text


def clean_books_in_directory(source_dir, dest_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".txt"):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_dir)
                dest_path = os.path.join(dest_dir, relative_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

                cleaned_text = clean_gutenberg_text(raw_text)

                with open(dest_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

                print(f"Processed: {relative_path}")


def collect_txt_files(root_dir):
    txt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                full_path = os.path.join(dirpath, filename)
                txt_files.append(full_path)
    return txt_files


def move_validation_set(files, target_size_bytes, val_dir):
    os.makedirs(val_dir, exist_ok=True)
    random.shuffle(files)

    total_size = 0
    val_files = []

    for file in files:
        file_size = os.path.getsize(file)
        if total_size + file_size > target_size_bytes:
            continue
        total_size += file_size
        val_files.append(file)

    for file in val_files:
        rel_path = os.path.relpath(file, start=os.path.commonpath([file, val_dir]))
        dest_path = os.path.join(val_dir, os.path.basename(file))
        shutil.move(file, dest_path)
        print(f"Moved {file} -> {dest_path}")

    print(f"Validation set size: {total_size / (1024 * 1024):.2f} MB from {len(val_files)} files")





remove_non_english = 0
clean_books = 1
validation = 0

if __name__ == "__main__":
    if remove_non_english:
        with open("pg_catalog.csv", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[1] != "Text" or row[4] != "en":
                    if not os.path.exists(f"cache/epub/{row[0]}"):
                        continue
                    if not os.path.exists(f"excluded/{row[0]}"):
                        os.makedirs(f"excluded/{row[0]}")
                    if os.path.exists(f"cache/epub/{row[0]}/pg{row[0]}.txt"):
                        shutil.move(src=f"cache/epub/{row[0]}/pg{row[0]}.txt", dst=f"excluded/{row[0]}")
                    os.rmdir(f"cache/epub/{row[0]}")
                    #print(row)

    if clean_books:
        clean_books_in_directory(SOURCE_DIR, DEST_DIR)

    if validation:

        # === USAGE ===
        source_dir = 'cache/cleaned'
        val_dir = 'cache/validation'
        target_size_mb = 180
        target_size_bytes = target_size_mb * 1024 * 1024

        txt_files = collect_txt_files(source_dir)
        move_validation_set(txt_files, target_size_bytes, val_dir)