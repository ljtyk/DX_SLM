"""Keep a copy of GPT2_model.py's rolling checkpoint every few hours.

GPT2_model.py overwrites checkpoints/latest.pt every ~16 minutes, and its own validation
uses whichever 5 books a random shuffle puts first at start-up, so the logged validation
curve jumps whenever training is resumed. Snapshots let eval_checkpoints.py score the
whole run afterwards on one fixed sample (the same one used for the original checkpoints).

Run from reproduction/work/:  python ../snapshot_checkpoints.py [HOURS]
"""
import os
import shutil
import sys
import time

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
SRC = "checkpoints/latest.pt"
DST_DIR = "snapshots"


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    last_copy, last_seen = 0.0, None
    while True:
        if os.path.exists(SRC):
            mtime = os.path.getmtime(SRC)
            # Copy once the file has changed, has been stable for a minute (save finished),
            # and the previous snapshot is old enough.
            if mtime != last_seen and time.time() - mtime > 60 and time.time() - last_copy >= HOURS * 3600:
                name = time.strftime("%Y%m%d_%H%M", time.localtime(mtime))
                dst = os.path.join(DST_DIR, f"ckpt_{name}.pt")
                shutil.copyfile(SRC, dst + ".part")
                if os.path.getmtime(SRC) == mtime:  # not overwritten mid-copy
                    os.replace(dst + ".part", dst)
                    last_copy, last_seen = time.time(), mtime
                    print(time.strftime("%H:%M"), "saved", dst, flush=True)
                else:
                    os.remove(dst + ".part")
        time.sleep(30)


if __name__ == "__main__":
    main()
