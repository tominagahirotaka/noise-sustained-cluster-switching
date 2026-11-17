#!/usr/bin/env python3
"""Run simulator tasks received as JSON lines on stdin; write results as JSON lines on stdout (stdlib only).

Used by scripts/eps_cluster_ensemble.py, locally and on remote hosts over ssh. Each input line is
{"id": int, "exe": name in --bin-dir, "args": [str, ...], "stdin": str}; each output line is
{"id": int, "rc": int, "stdout": str, "stderr": str}. Up to -j tasks run at once; the worker exits after stdin
closes and all tasks have finished.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import subprocess
import sys
import threading

ap = argparse.ArgumentParser()
ap.add_argument("--bin-dir", required=True)
ap.add_argument("-j", type=int, required=True)
a = ap.parse_args()
bin_dir = Path(a.bin_dir).expanduser()
lock = threading.Lock()


def run(task):
    try:
        p = subprocess.run([str(bin_dir / task["exe"]), *task["args"]], input=task["stdin"], text=True,
                           capture_output=True)
        rc, out, err = p.returncode, p.stdout, p.stderr[-2000:]
    except Exception as e:  # report instead of dropping the task, so the dispatcher never waits forever
        rc, out, err = -1, "", repr(e)
    line = json.dumps(dict(id=task["id"], rc=rc, stdout=out, stderr=err))
    with lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


with ThreadPoolExecutor(a.j) as pool:
    for raw in sys.stdin:
        if raw.strip():
            pool.submit(run, json.loads(raw))
