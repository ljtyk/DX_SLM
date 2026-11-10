"""Download English Project Gutenberg texts into the layout train_tokenizer.py expects.

Run from reproduction/work/:  python ../download_gutenberg.py
Creates pg_catalog.csv and cache/epub/{id}/pg{id}.txt. Applies the same filter as
the remove_non_english step in train_tokenizer.py (Type == "Text", Language == "en"),
so non-English books are never downloaded. Resumable: existing files are skipped.

This fetches books one at a time (~1 MB/s, many hours). The fast route is the bulk
archive plus extract_archive.py; this script then only fills in what it lacks.
"""
import csv
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
MIRROR = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"  # PG mirror for bulk access
HEADERS = {"User-Agent": "gutenberg-corpus-fetch/1.0"}
WORKERS = 8  # the mirror serves ~150-200 KB/s per connection


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def download(book_id):
    path = f"cache/epub/{book_id}/pg{book_id}.txt"
    if os.path.exists(path):
        return "skip", 0
    data = fetch(MIRROR.format(id=book_id))
    if data is None:
        return "missing", 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return "ok", len(data)


def main():
    if not os.path.exists("pg_catalog.csv"):
        with open("pg_catalog.csv", "wb") as f:
            f.write(fetch(CATALOG_URL))

    with open("pg_catalog.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        ids = [row[0] for row in reader if row[1] == "Text" and row[4] == "en"]
    print(f"{len(ids):,} English texts in catalog", flush=True)

    counts, total_bytes, start = {"ok": 0, "skip": 0, "missing": 0}, 0, time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, (status, n) in enumerate(pool.map(download, ids), 1):
            counts[status] += 1
            total_bytes += n
            if i % 500 == 0 or i == len(ids):
                rate = total_bytes / max(time.time() - start, 1) / 1e6
                print(f"{i:,}/{len(ids):,}  {counts}  {total_bytes / 1e9:.2f} GB new  {rate:.1f} MB/s",
                      flush=True)


if __name__ == "__main__":
    sys.exit(main())
