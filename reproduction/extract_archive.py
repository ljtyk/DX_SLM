"""Extract English texts from Gutenberg's bulk archive into cache/epub/{id}/pg{id}.txt.

Much faster than download_gutenberg.py: fetch the whole plain-text collection once,
  https://www.gutenberg.org/cache/epub/feeds/txt-files.tar.zip  (~11 GB, all languages)
then run from reproduction/work/:  python ../extract_archive.py txt-files.tar.zip
Only books passing the same filter as train_tokenizer.py (Type == "Text",
Language == "en", from pg_catalog.csv) are extracted. Existing files are skipped, and
download_gutenberg.py can be run afterwards to fill in anything the archive lacks.
"""
import csv
import os
import re
import shutil
import sys
import tarfile
import zipfile

ARCHIVE = sys.argv[1] if len(sys.argv) > 1 else "txt-files.tar.zip"
NAME_RE = re.compile(r"(?:^|/)(\d+)/pg\1\.txt$")


def english_ids():
    with open("pg_catalog.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return {row[0] for row in reader if row[1] == "Text" and row[4] == "en"}


def extract_tar(fileobj, wanted):
    counts = {"ok": 0, "skip": 0, "other": 0}
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:  # streaming, no temp tar on disk
        for member in tar:
            m = NAME_RE.search(member.name) if member.isfile() else None
            if not m or m.group(1) not in wanted:
                counts["other"] += 1
                continue
            book_id = m.group(1)
            path = f"cache/epub/{book_id}/pg{book_id}.txt"
            if os.path.exists(path):
                counts["skip"] += 1
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with tar.extractfile(member) as src, open(path + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(path + ".part", path)
            counts["ok"] += 1
            if counts["ok"] % 2000 == 0:
                print(counts, flush=True)
    return counts


def main():
    wanted = english_ids()
    print(f"{len(wanted):,} English texts in catalog", flush=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        names = zf.namelist()
        print("archive contains:", names[:5], flush=True)
        for name in names:
            if name.endswith(".tar"):
                with zf.open(name) as f:
                    print(name, extract_tar(f, wanted), flush=True)


if __name__ == "__main__":
    main()
