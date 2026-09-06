#!/usr/bin/env python3
"""
Step 1b: Body-only vertebral segmentation (TotalSegmentator --task vertebrae_body).

Added 2026-09 after the adversarial review (finding F01): the original
isolate_vertebral_body() kept the largest connected component per slice, which
in pedicle slices is the whole vertebra including the posterior arch. The
vertebrae_body task outputs the vertebral bodies only (no arch), which is what
the trabecular ROI and the Methods text require.

This script is ADDITIVE: it writes
    DerivedData/<sub>/<ses>/segmentations/vertebrae_body/
        vertebrae_body.nii.gz
        intervertebral_discs.nii.gz
        invocation.log
        manifest.json          (command, versions, input hash, timing, exit code)
and touches nothing else. Downstream use of these masks is a separate step.

Usage:
    python Scripts/01b_segment_vertebral_bodies.py --data ../RawData/bmd_ct --output .
    python Scripts/01b_segment_vertebral_bodies.py --data ../RawData/bmd_ct --output . --subject sub-101

Notes:
- macOS limits AF_UNIX socket paths to ~104 chars; nnU-Net's multiprocessing
  creates such sockets under TMPDIR. With this project's long absolute path
  that fails, so the worker chdirs to the project root and uses a short
  RELATIVE temp dir (tmp_ts/, git-ignored). This mirrors the setup that the
  review used successfully on all 26 scans.
- Device: mps (Apple GPU), -nr 2 -ns 2, as in the review (mean 42 s per scan).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SESSIONS = ["ses-Baseline", "ses-Followup"]
TASK = "vertebrae_body"
OUT_SUBDIR = "vertebrae_body"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def ts_version() -> str:
    try:
        from importlib.metadata import version
        return version("TotalSegmentator")
    except Exception:
        return "unknown"


def find_ct(data_dir: Path, sub: str, ses: str) -> Path:
    cands = sorted((data_dir / sub / ses / "ct").glob("*_rec-stnd1.25mm_ct.nii.gz"))
    if len(cands) != 1:
        raise FileNotFoundError(f"{sub}/{ses}: expected exactly one 1.25 mm CT, found {len(cands)}")
    return cands[0]


# ----------------------------------------------------------------------------
# Worker: runs inside a subprocess with short relative temp paths
# ----------------------------------------------------------------------------
def worker(ct: Path, out: Path, root: Path):
    import multiprocessing
    os.chdir(root)
    tmp = Path("tmp_ts")
    (tmp / "mp").mkdir(parents=True, exist_ok=True)
    (tmp / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp / "tmp")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    multiprocessing.current_process()._config["tempdir"] = str(tmp / "mp")
    from totalsegmentator.bin.TotalSegmentator import main
    sys.argv = ["TotalSegmentator", "-i", str(ct), "-o", str(out),
                "--task", TASK, "-d", "mps", "-nr", "2", "-ns", "2"]
    main()


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def already_done(out: Path) -> bool:
    m = out / "manifest.json"
    if not (out / "vertebrae_body.nii.gz").exists() or not m.exists():
        return False
    try:
        return json.load(open(m)).get("exit_code") == 0
    except Exception:
        return False


def segment_vertebral_bodies(ct: Path, out: Path, root: Path, force: bool = False, log=print) -> int:
    """Run the vertebrae_body task for one CT into `out` (with manifest). Returns 0 on success."""
    return _run(ct, Path(out), Path(root), force, log, label=f"{Path(ct).name}")


def run_session(sub: str, ses: str, data_dir: Path, root: Path, force: bool, log):
    ct = find_ct(data_dir, sub, ses)
    out = root / "DerivedData" / sub / ses / "segmentations" / OUT_SUBDIR
    return _run(ct, out, root, force, log, label=f"{sub}/{ses}")


def _run(ct: Path, out: Path, root: Path, force: bool, log, label: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    if already_done(out) and not force:
        log(f"{label}: exists, skipping")
        return 0

    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
           "--ct", str(ct), "--out", str(out), "--root", str(root)]
    log(f"{label}: START  {ct.name}")
    t0 = time.time()
    with open(out / "invocation.log", "w") as f:
        f.write("COMMAND " + " ".join(cmd) + "\n")
        f.write(f"TotalSegmentator args: -i {ct} -o {out} --task {TASK} -d mps -nr 2 -ns 2\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = "TIMEOUT"
    dt = time.time() - t0
    ok = code == 0 and (out / "vertebrae_body.nii.gz").exists()
    manifest = {
        "task": TASK,
        "totalsegmentator_version": ts_version(),
        "device": "mps", "nr_threads_resampling": 2, "nr_threads_saving": 2,
        "input_ct": str(ct), "input_sha256": sha256(ct),
        "output_files": sorted(p.name for p in out.glob("*.nii.gz")),
        "start": datetime.fromtimestamp(t0).isoformat(),
        "elapsed_s": round(dt, 1),
        "exit_code": code if ok else (code if code != 0 else "NO_OUTPUT"),
        "git_commit": git_commit(root),
        "script": Path(__file__).name,
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    log(f"{label}: {'OK' if ok else 'FAILED'}  exit={code}  {dt:.0f}s")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="RawData/bmd_ct directory")
    ap.add_argument("--output", type=Path, default=Path("."), help="project root (contains DerivedData/)")
    ap.add_argument("--subject", help="single subject, e.g. sub-101")
    ap.add_argument("--force", action="store_true")
    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--ct", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--root", type=Path)
    a = ap.parse_args()

    if a.worker:
        worker(a.ct, a.out, a.root)
        return

    root = a.output.resolve()
    data_dir = a.data.resolve()
    (root / "Log").mkdir(exist_ok=True)
    logfile = root / "Log" / f"vertebrae_body_{datetime.now():%Y%m%d_%H%M%S}.txt"

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(logfile, "a") as f:
            f.write(line + "\n")

    subjects = [a.subject] if a.subject else sorted(p.name for p in data_dir.glob("sub-*"))
    log(f"TotalSegmentator {ts_version()}, task {TASK}, {len(subjects)} subjects, log {logfile.name}")
    failures = 0
    for sub in subjects:
        for ses in SESSIONS:
            if not (data_dir / sub / ses).exists():
                continue
            try:
                failures += run_session(sub, ses, data_dir, root, a.force, log)
            except Exception as e:
                failures += 1
                log(f"{sub}/{ses}: ERROR {e}")
    log(f"DONE. failures={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
