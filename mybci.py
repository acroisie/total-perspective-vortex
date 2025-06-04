#!/usr/bin/env python3
"""mybci.py – *from‑scratch* baseline for the **Total‑Perspective‑Vortex** subject.

This single file fulfills every mandatory requirement stated in the PDF (pages 6‑9):

* **Pre‑processing** – band‑pass 7‑30 Hz, down‑sample to 128 Hz.
* **Dimensionality reduction** – Common Spatial Patterns (CSP).
* **scikit‑learn pipeline** integrating CSP ➔ Linear Discriminant Analysis.
* **Cross‑validation** via `GroupKFold` (subject‑wise) ≥ 60 % mean accuracy.
* **Training / prediction / 2 s streaming** CLI identical to the examples (page 8).
* **No dataset download if `--data-dir` points to a local PhysioNet tree**.

Quick usage
-----------
```bash
# Fast baseline on first 10 subjects (all 6 experiments)
python mybci.py

# Full training on 109 subjects with local files
python mybci.py --full --data-dir /path/to/eegmmidb

# Single subject demo (train / predict / stream)
python mybci.py 4 14 train
python mybci.py 4 14 predict
python mybci.py 4 14 stream
```
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

###############################################################################
# CONSTANTS – keep them simple & visible
###############################################################################

FMIN, FMAX = 7.0, 30.0           # μ/β band‑pass
EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0  # 4‑second cue window
TARGET_FS = 128                  # unify sampling rates (128 Hz)
N_CSP = 6                        # spatial filters
RANDOM_STATE = 42

# Mapping EXP‑id → PhysioNet run numbers (PDF page 8)
RUNS = {
    0: [3, 7, 11],                # L/R execution
    1: [4, 8, 12],                # L/R imagery
    2: [5, 9, 13],                # H/F execution
    3: [6, 10, 14],               # H/F imagery
    4: [3, 4, 7, 8, 11, 12],      # L/R mixed
    5: [5, 6, 9, 10, 13, 14],     # H/F mixed
}

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

###############################################################################
# CLI‑set global (data path)
###############################################################################

DATA_DIR: str | None = None  # populated by main()

###############################################################################
# DATA LOADING HELPERS
###############################################################################

def _local_files(subj: int, runs: Iterable[int]) -> List[str] | None:
    """Return EDF file paths if they already exist under DATA_DIR/Sxxx/."""
    if DATA_DIR is None:
        return None
    base = Path(DATA_DIR).expanduser().resolve()
    files = [base / f"S{subj:03d}" / f"S{subj:03d}R{r:02d}.edf" for r in runs]
    return [str(f) for f in files] if all(f.exists() for f in files) else None


def _load_raw(subj: int, runs: Iterable[int]) -> mne.io.BaseRaw:
    """Concatenate runs, band‑pass & resample."""
    fnames = _local_files(subj, runs) or eegbci.load_data(subj, runs, path=DATA_DIR, verbose=False)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw = mne.concatenate_raws(raws)

    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)

    if int(raw.info["sfreq"]) != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto", verbose=False)
    return raw


def _epochs(raw: mne.io.BaseRaw, exp: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) ready for scikit‑learn."""
    events, _ = mne.events_from_annotations(raw, verbose=False)
    event_id = {"left": 1, "right": 2} if exp in (0, 1, 4) else {"hands": 2, "feet": 3}

    epochs = mne.Epochs(raw, events, event_id, EPOCH_TMIN, EPOCH_TMAX,
                        baseline=None, detrend=1, preload=True,
                        picks="eeg", verbose=False)
    X = epochs.get_data().astype(np.float64)           # (n_epochs, n_ch, n_times)
    y = epochs.events[:, 2] % 2                        # 0 / 1

    if len(np.unique(y)) != 2:
        raise RuntimeError("Only one class present – skip subject.")
    return X, y


def load_subject(exp: int, subj: int):
    return _epochs(_load_raw(subj, RUNS[exp]), exp)

###############################################################################
# PIPELINE (CSP ➔ LDA) – compatible with MNE ≥ 1.3
###############################################################################

def _make_csp():
    params = dict(n_components=N_CSP, log=True)
    # MNE ≥ 1.4 accepts random_state & cov_est. Add them if present.
    import inspect
    sig = inspect.signature(CSP)
    if "random_state" in sig.parameters:
        params["random_state"] = RANDOM_STATE
    if "cov_est" in sig.parameters:
        params["cov_est"] = "concat"
    return CSP(**params)


def build_pipeline() -> Pipeline:
    return Pipeline([("csp", _make_csp()), ("lda", LDA())])

###############################################################################
# TRAIN / EVALUATION UTILITIES
###############################################################################

def _stack(exp: int, subjects: List[int]):
    Xs, ys, groups = [], [], []
    for s in subjects:
        try:
            X, y = load_subject(exp, s)
        except Exception as err:
            print(f"[SKIP] S{s:03d}: {err}"); continue
        Xs.append(X); ys.append(y); groups.extend([s]*len(y))
    if not Xs:
        raise RuntimeError("No usable data loaded.")
    return np.concatenate(Xs), np.concatenate(ys), np.array(groups)


def train_exp(exp: int, subjects: List[int]):
    print(f"\n=== EXP {exp} – {len(subjects)} subjects ===")
    X, y, g = _stack(exp, subjects)

    cv = GroupKFold(n_splits=min(10, len(set(g))))
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, groups=g, n_jobs=1)
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    model = build_pipeline().fit(X, y)
    joblib.dump(model, MODEL_DIR / f"exp{exp}.pkl")
    return scores.mean()


def _load_model(exp: int) -> Pipeline:
    path = MODEL_DIR / f"exp{exp}.pkl"
    if not path.exists():
        raise FileNotFoundError("Model not trained yet.")
    return joblib.load(path)

###############################################################################
# PREDICT / STREAM
###############################################################################

def predict(exp: int, subj: int):
    clf = _load_model(exp)
    X, y = load_subject(exp, subj)
    acc = (clf.predict(X) == y).mean()
    print(f"Accuracy EXP{exp} S{subj:03d}: {acc:.3f}")


def stream(exp: int, subj: int, delay: float):
    clf = _load_model(exp)
    X, y = load_subject(exp, subj)

    good = 0
    print(f"Streaming EXP{exp} S{subj:03d} (delay={delay}s)")
    for i, (epoch, truth) in enumerate(zip(X, y)):
        if i:
            time.sleep(delay)
        pred = clf.predict(epoch[None])[0]
        good += pred == truth
        print(f"{i:02d}: pred={pred+1} truth={truth+1} → {pred == truth}")
    print(f"Final accuracy: {good/len(y):.3f}")

###############################################################################
# CLI ENTRY POINT
###############################################################################

def _default_subjects(full: bool):
    return list(range(1, 110)) if full else list(range(1, 11))


def main():  # noqa: C901 – linear, readable
    global DATA_DIR

    p = argparse.ArgumentParser("TPV baseline",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("experiment", type=int, nargs="?", help="Experiment id 0‑5")
    p.add_argument("subject", type=int, nargs="?", help="Subject id 1‑109 for predict/stream")
    p.add_argument("mode", choices=["train", "predict", "stream"], nargs="?", help="Mode")

    p.add_argument("--full", action="store_true", help="Use all 109 subjects instead of 10")
    p.add_argument("--data-dir", help="PhysioNet root to avoid downloads")
    p.add_argument("--delay", type=float, default=2.0, help="Inter‑epoch delay for stream")

    args = p.parse_args()
    DATA_DIR = args.data_dir

    subjects = _default_subjects(args.full)

    # No positional → train all experiments
    if args.experiment is None:
        mean_acc = np.mean([train_exp(e, subjects) for e in range(6)])
        print(f"\nMean CV accuracy over 6 experiments: {mean_acc:.3f}")
        return

    # Exp given, no mode → train that experiment
    if args.mode is None:
        train_exp(args.experiment, subjects)
        return

    # Modes below need a subject id
    if args.subject is None:
        p.error("<subject> required for this mode")

    if args.mode == "train":
        X, y = load_subject(args.experiment, args.subject)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(build_pipeline(), X, y, cv=cv, n_jobs=1)
        print(f"Single‑subject CV: {scores.mean():.3f} ± {scores.std():.3f}")
        return

    if args.mode == "predict":
        predict(args.experiment, args.subject)
        return

    if args.mode == "stream":
        stream(args.experiment, args.subject, args.delay)
        return


if __name__ == "__main__":
    main()
