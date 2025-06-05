#!/usr/bin/env python3
import argparse, logging, time, warnings
from pathlib import Path

import joblib
import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from custom_csp import CustomCSP
from filterbank_csp import FilterBankCSP

SEED = 42
FREQ = (8, 32)
WIN = (0.7, 3.9)
EEG_CH = ["C3", "C4", "Cz"]
FAST = 10
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

RUNS = {
    0: [3, 7, 11],
    1: [4, 8, 12],
    2: [5, 9, 13],
    3: [6, 10, 14],
    4: [3, 4, 7, 8, 11, 12],
    5: [5, 6, 9, 10, 13, 14],
}

# mute lib spam
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
for n in ("mne", "joblib", ""):
    logging.getLogger(n).setLevel(logging.ERROR)


class FBCSP(FilterBankCSP):
    """Patch pour forcer float64"""

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        self.csp_list = []
        for lo, hi in self.freq_bands:
            Xf = self._bandpass_filter(X, lo, hi)
            self.csp_list.append(CustomCSP(n_components=self.n_csp, log=True).fit(Xf, y))
        return self

    def transform(self, X):
        return super().transform(np.asarray(X, np.float64))


# ---------- data ---------- #

def _get_raw(files):
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"), verbose=False)
    raw.filter(*FREQ, fir_design="firwin", verbose=False)
    raw.pick(mne.pick_channels(raw.info["ch_names"], include=EEG_CH))
    return raw


def load(exp: int, subj: int, data_dir: Path | None = None):
    """Retourne X, y."""
    if data_dir:
        files = [
            str(data_dir / f"S{subj:03d}" / f"S{subj:03d}R{r:02d}.edf")
            for r in RUNS[exp]
            if (data_dir / f"S{subj:03d}" / f"S{subj:03d}R{r:02d}.edf").exists()
        ]
        if not files:
            raise FileNotFoundError(f"pas de fichier EDF pour S{subj:03d}")
    else:
        files = eegbci.load_data(subj, RUNS[exp], verbose=False)
    raw = _get_raw(files)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    event_id = {"left": 1, "right": 2} if exp in (0, 1, 4) else {"hands": 2, "feet": 3}
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        *WIN,
        baseline=None,
        detrend=1,
        preload=True,
        picks="eeg",
        verbose=False,
    )
    X = epochs.get_data().astype(np.float64)
    y = (epochs.events[:, 2] % 2).astype(int)
    return X, y


# ---------- model ---------- #

def make_pipe(n_csp=3, fb=True, sfreq=160):
    step = ("FBCSP", FBCSP(n_csp=n_csp, sfreq=sfreq)) if fb else ("CSP", CustomCSP(n_components=n_csp, log=True))
    return Pipeline([step, ("LDA", LDA())])


# ---------- helpers ---------- #

def _aggregate(dirs, exp, data_dir):
    X_all, y_all = [], []
    lens = []
    valids = []
    for d in dirs:
        s = int(d.name[1:])
        try:
            X, y = load(exp, s, data_dir)
            X_all.append(X)
            y_all.append(y)
            lens.append(X.shape[2])
            valids.append(s)
        except Exception:
            pass
    if not X_all:
        raise ValueError("aucune donnée")
    L = min(lens)
    X_all = [x[:, :, :L] for x in X_all]
    return np.concatenate(X_all), np.concatenate(y_all), valids


# ---------- train ---------- #

def train_split(exp: int, data_dir: Path, full=False):
    dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("S")])
    if not full:
        dirs = dirs[:FAST]
    tr, tmp = train_test_split(dirs, test_size=0.4, random_state=SEED)
    te, ho = train_test_split(tmp, test_size=0.5, random_state=SEED)

    Xtr, ytr, _ = _aggregate(tr, exp, data_dir)
    Xte, yte, _ = _aggregate(te, exp, data_dir)

    pipe = make_pipe()
    cv = cross_val_score(pipe, Xtr, ytr, cv=5)
    pipe.fit(Xtr, ytr)
    acc = pipe.score(Xte, yte)

    joblib.dump({"model": pipe, "cv": cv, "test": acc}, MODEL_DIR / f"bci_exp{exp}.pkl")
    print(f"exp{exp} ▶ test={acc:.3f} cv={cv.mean():.3f}")

    try:
        Xh, yh, _ = _aggregate(ho, exp, data_dir)
        print(f"hold={pipe.score(Xh, yh):.3f}")
    except Exception:
        pass


# ---------- predict ---------- #

def _find_model(exp, subj):
    cands = [
        MODEL_DIR / f"bci_exp{exp}_subj{subj:03d}.pkl",
        MODEL_DIR / f"bci_exp{exp}.pkl",
        MODEL_DIR / f"bci_exp{exp}_split.pkl",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("aucun modèle")


def predict(exp: int, subj: int, data: Path | None = None, stream=False, delay=0.1):
    mdl = joblib.load(_find_model(exp, subj))
    pipe = mdl["model"] if isinstance(mdl, dict) else mdl
    X, y = load(exp, subj, data)
    ok, t0 = 0, time.time()
    for i, (e, t) in enumerate(zip(X, y)):
        if stream and i:
            time.sleep(delay)
        p = pipe.predict(e.reshape(1, *e.shape))[0]
        ok += p == t
        print(f"{i:02d}: [{p+1}] [{t+1}] {p == t}")
    print(f"acc={ok/len(y):.3f} in {time.time()-t0:.1f}s")


# ---------- CLI ---------- #

def main():
    pa = argparse.ArgumentParser(description="BCI MI – refacto")
    pa.add_argument("exp", type=int, nargs="?", help="exp 0‑5")
    pa.add_argument("subj", type=int, nargs="?", help="subject 1‑109")
    pa.add_argument("mode", choices=["train", "predict", "stream"], nargs="?")
    pa.add_argument("--data", type=Path)
    pa.add_argument("--fast", action="store_true")
    args = pa.parse_args()

    if args.exp is None:
        if not args.data:
            return pa.error("--data obligatoire pour l'entraînement global")
        for e in range(6):
            train_split(e, args.data, full=not args.fast)
        return

    if args.subj is None:
        if not args.data:
            return pa.error("--data requis pour exp seule")
        train_split(args.exp, args.data, full=not args.fast)
        return

    if args.mode == "train":
        X, y = load(args.exp, args.subj, args.data)
        pipe = make_pipe()
        cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=SEED))
        pipe.fit(X, y)
        joblib.dump({"model": pipe, "cv": cv}, MODEL_DIR / f"bci_exp{args.exp}_subj{args.subj:03d}.pkl")
        print(f"cv={cv.mean():.3f}")
    elif args.mode == "predict":
        predict(args.exp, args.subj, args.data)
    else:
        predict(args.exp, args.subj, args.data, stream=True)


if __name__ == "__main__":
    main()
