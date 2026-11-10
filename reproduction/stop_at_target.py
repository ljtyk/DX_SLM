"""Score each new snapshot on the fixed validation sample; stop training at a target.

Run from reproduction/work/:  python ../stop_at_target.py [TARGET_LOSS] [MAX_STEP]
Defaults: the original final.pt's score in eval_results.json, and its global step.
Every few minutes, snapshots/ckpt_*.pt files not yet scored are passed to
eval_checkpoints.py (results go to eval_results.json as snap_*). Training stops when
  - a snapshot scores at or below the target loss, or
  - GPT2_model.py saves a checkpoint at or past MAX_STEP; that checkpoint is copied to
    snapshots/ as the final snapshot and scored.
The GPT2_model.py and snapshot_checkpoints.py processes are stopped either way.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def results():
    return json.load(open("eval_results.json")) if os.path.exists("eval_results.json") else {}


def stop_training():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { "
          "$_.CommandLine -match 'GPT2_model.py|snapshot_checkpoints.py' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; 'stopped ' + $_.CommandLine }")
    print(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout,
          flush=True)


def last_saved_step():
    if not os.path.exists("train.log"):
        return 0
    with open("train.log", encoding="utf-8", errors="ignore") as f:
        steps = re.findall(r"Saved checkpoint at step (\d+)", f.read())
    return int(steps[-1]) if steps else 0


def score(path):
    name = "snap_" + os.path.basename(path)[5:-3]
    subprocess.run([sys.executable, os.path.join(HERE, "eval_checkpoints.py"), f"{name}={path}"],
                   capture_output=True)
    r = results().get(name)
    if r is None:
        print(time.strftime("%H:%M"), name, "eval failed", flush=True)
    else:
        print(time.strftime("%H:%M"), name, f"step {r['global_step']:,}  val {r['val_loss']:.4f}", flush=True)
    return r


def main():
    orig = results().get("original_final", {})
    target = float(sys.argv[1]) if len(sys.argv) > 1 else orig["val_loss"]
    max_step = int(sys.argv[2]) if len(sys.argv) > 2 else orig["global_step"]
    print(f"target val loss {target:.4f}, step cap {max_step:,}", flush=True)
    while True:
        step = last_saved_step()
        if step >= max_step:
            time.sleep(60)  # let the save finish
            dst = f"snapshots/ckpt_step{step}.pt"
            shutil.copyfile("checkpoints/latest.pt", dst)
            print(time.strftime("%H:%M"), f"step cap reached at {step:,}, stopping training", flush=True)
            stop_training()
            score(dst)
            return

        done = results()
        for path in sorted(glob.glob("snapshots/ckpt_*.pt")):
            if "snap_" + os.path.basename(path)[5:-3] in done:
                continue
            r = score(path)
            if r is not None and r["val_loss"] <= target:
                print(time.strftime("%H:%M"), "target loss reached, stopping training", flush=True)
                stop_training()
                return
        time.sleep(300)


if __name__ == "__main__":
    main()
