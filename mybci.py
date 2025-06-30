import argparse
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import mne
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.io import read_raw_edf, concatenate_raws
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from filterbank_csp import FilterBankCSP
from custom_csp import CustomCSP

warnings.filterwarnings("ignore")
for mod in ("mne", "joblib"):
    logging.getLogger(mod).setLevel(logging.ERROR)
    logging.getLogger(mod).propagate = False
mne.set_log_level("ERROR")

FMIN, FMAX = 8.0, 32.0
TMIN, TMAX = 0.7, 3.9
SEED = 42
FAST_SUBJECTS = 10
PREDICT_DELAY = 0.1
EEG_CHANNELS = ["C3", "C4", "Cz"]
N_CSP = 3
USE_FILTERBANK = True
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

EXPERIMENT_RUNS = {
    0: [3, 7, 11],  # left vs right hand
    1: [4, 8, 12],  # left vs right foot
    2: [5, 9, 13],  # right hand vs foot
    3: [6, 10, 14],  # right hand vs foot
    4: [3, 4, 7, 8, 11, 12],  # all left
    5: [5, 6, 9, 10, 13, 14],  # all right
}


class FBCSP(FilterBankCSP):
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.csp_list = []
        for fmin, fmax in self.freq_bands:
            X_f = self._bandpass_filter(X, fmin, fmax)
            csp = CustomCSP(n_components=self.n_csp, log=True)
            csp.fit(X_f, y)
            self.csp_list.append(csp)
        return self

    def transform(self, X):
        return super().transform(np.asarray(X, dtype=np.float64))


def make_pipe():
    if USE_FILTERBANK:
        return Pipeline(
            [
                ("FBCSP", FBCSP(n_csp=N_CSP, sfreq=160)),
                ("LDA", LDA()),
            ]
        )
    return Pipeline(
        [
            ("CSP", CustomCSP(n_components=N_CSP, log=True)),
            ("LDA", LDA()),
        ]
    )


def _raw_from_files(exp: int, subj: int, root: Path):
    runs = EXPERIMENT_RUNS[exp]
    p = root / f"S{subj:03d}"
    files = [p / f"S{subj:03d}R{r:02d}.edf" for r in runs]
    files = [str(f) for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"Missing EDF for S{subj:03d}, exp {exp}")
    return concatenate_raws(
        [read_raw_edf(f, preload=True, verbose=False) for f in files]
    )


def _prep_raw(raw):
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"), verbose=False)
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    raw.pick(mne.pick_channels(raw.info["ch_names"], include=EEG_CHANNELS))
    return raw


def load_data(exp: int, subj: int, data_dir: Path):
    raw = _raw_from_files(exp, subj, data_dir)
    raw = _prep_raw(raw)
    events, _ = mne.events_from_annotations(raw, verbose=False)

    if exp in {0, 1, 4}:
        event_id = {"left": 1, "right": 2}
    else:
        event_id = {"hands": 2, "feet": 3}

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        TMIN,
        TMAX,
        baseline=None,
        detrend=1,
        preload=True,
        verbose=False,
        picks="eeg",
    )
    X = epochs.get_data().astype(np.float64)
    y = (epochs.events[:, 2] % 2).astype(int)
    min_len = min(X.shape[2], *(e.shape[2] for e in [X]))
    return X[:, :, :min_len], y


def _pick_model(exp: int, subj: int | None = None):
    cands = [
        MODEL_DIR / f"bci_exp{exp}_subj{subj:03d}.pkl" if subj else None,
        MODEL_DIR / f"bci_exp{exp}.pkl",
        MODEL_DIR / f"bci_exp{exp}_split.pkl",
    ]
    for p in filter(None, cands):
        if p.exists():
            return p
    raise FileNotFoundError("Model not found; train first")


def _load_pipe(path: Path):
    data = joblib.load(path)
    return data["model"] if isinstance(data, dict) and "model" in data else data


def predict_subject(
    exp: int, subj: int, data_dir: Path, playback: bool = False
):
    model_path = _pick_model(exp, subj)
    pipe = _load_pipe(model_path)
    X, y = load_data(exp, subj, data_dir)
    if playback:
        start = time.time()
        ok = 0
        print(f"epoch nb: [prediction] [truth] equal?")
        for i, (e, t) in enumerate(zip(X, y)):
            time.sleep(0.01)
            pred = pipe.predict(e[None])[0]
            ok += pred == t
            print(
                f"epoch {i:02d}: [{pred+1}] [{t+1}] {pred==t} (t={time.time()-start:.1f}s)"
            )
        print(f"Accuracy: {ok/len(y):.4f}")
    else:
        preds = pipe.predict(X)
        report(preds, y)


def stream_subject(exp: int, subj: int, data_dir: Path, delay: float = 2.0):
    model_path = _pick_model(exp, subj)
    pipe = _load_pipe(model_path)
    X, y = load_data(exp, subj, data_dir)
    start = time.time()
    ok = 0
    print(f"epoch nb: [prediction] [truth] equal?")
    for i, (e, t) in enumerate(zip(X, y)):
        if i:
            time.sleep(delay)
        else:
            time.sleep(0.06)
        pred = pipe.predict(e[None])[0]
        ok += pred == t
        print(
            f"epoch {i:02d}: [{pred+1}] [{t+1}] {pred==t} (t={time.time()-start:.1f}s)"
        )
    print(f"Accuracy: {ok/len(y):.4f}")


def report(preds, truth):
    mapped_pred, mapped_truth = preds + 1, truth + 1
    ok = mapped_pred == mapped_truth
    for i, (p, t, c) in enumerate(zip(mapped_pred, mapped_truth, ok)):
        print(f"epoch {i:02d}: [{p}] [{t}] {c}")
    print(f"Accuracy: {ok.mean():.4f}")


def _cv(pipe, X, y):
    cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1)
    print(scores.round(4))
    print(f"cross_val_score: {scores.mean():.4f}")
    return scores


def train_subject(exp: int, subj: int, data_dir: Path):
    X, y = load_data(exp, subj, data_dir)
    pipe = make_pipe()
    scores = _cv(pipe, X, y)
    pipe.fit(X, y)
    model_path = MODEL_DIR / f"bci_exp{exp}_subj{subj:03d}.pkl"
    joblib.dump({"model": pipe, "cv_scores": scores}, model_path)
    print(f"Model saved to {model_path}")


def _aggregate(subject_dirs, exp, data_dir):
    from utils_multi_subject import aggregate_multi_subject_data

    return aggregate_multi_subject_data(
        subject_dirs, exp, lambda e, s: load_data(e, s, data_dir)
    )


def train_split(exp: int | None, data_dir: Path, full: bool):
    from utils_multi_subject import list_subject_dirs

    start_time = time.time()
    subs = list_subject_dirs(data_dir)
    if not full:
        subs = subs[:FAST_SUBJECTS]
        print(f"[INFO] Mode --fast : {len(subs)} subjects")
    else:
        print(f"[INFO] {len(subs)} subjects found")
    train_dirs, tmp_dirs = train_test_split(
        subs, test_size=0.4, random_state=SEED
    )
    test_dirs, holdout_dirs = train_test_split(
        tmp_dirs, test_size=0.5, random_state=SEED
    )
    print(
        f"[INFO] Train: {len(train_dirs)} Test: {len(test_dirs)} Holdout: {len(holdout_dirs)}"
    )
    exps = range(6) if exp is None else [exp]
    for e in exps:
        print(f"\n[INFO] ==== Exp {e} ====")
        try:
            X_tr, y_tr, _ = _aggregate(train_dirs, e, data_dir)
            X_te, y_te, _ = _aggregate(test_dirs, e, data_dir)
        except Exception as err:
            print(f"[WARN] Skip exp {e}: {err}")
            continue
        if len(np.unique(y_tr)) < 2:
            print(f"[WARN] Exp {e} has one class only, skip")
            continue
        pipe = make_pipe()
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy")
        print(f"[INFO] cv_scores: {cv_scores.round(4)}")
        print(f"[INFO] cv mean={cv_scores.mean():.3f}")
        pipe.fit(X_tr, y_tr)
        test_acc = pipe.score(X_te, y_te)
        print(f"[INFO] Test acc={test_acc:.3f}")
        model_path = MODEL_DIR / f"bci_exp{e}.pkl"
        joblib.dump(
            {"model": pipe, "cv_scores": cv_scores, "test_acc": test_acc},
            model_path,
        )
        print(f"[INFO] Saved => {model_path}")
    holdout_accs = []
    for e in exps:
        try:
            from joblib import load

            model_path = MODEL_DIR / f"bci_exp{e}.pkl"
            pipe = load(model_path)["model"]
            X_ho, y_ho, _ = _aggregate(holdout_dirs, e, data_dir)
            hold_acc = pipe.score(X_ho, y_ho)
            holdout_accs.append((e, hold_acc))
            print(f"[INFO] Holdout acc exp {e}: {hold_acc:.3f}")
        except Exception as err:
            print(f"[INFO] No holdout for exp {e}: {err}")
    if holdout_accs:
        print("\n[INFO] Holdout accuracies by experiment:")
        for e, acc in holdout_accs:
            print(f"  Exp {e}: {acc:.3f}")
        mean_acc = np.mean([acc for _, acc in holdout_accs])
        print(f"[INFO] Holdout mean accuracy: {mean_acc:.3f}")
    else:
        print("[INFO] No holdout accuracies computed.")
    elapsed = time.time() - start_time
    print(f"[INFO] Total execution time: {elapsed:.1f} seconds")


def main():
    ap = argparse.ArgumentParser("Total Perspective Vortex – BCI")
    ap.add_argument(
        "data_path",
        type=Path,
        help="Local data root (containing S001/, S002/, etc.)",
    )
    ap.add_argument("experiment", type=int, nargs="?", help="Exp id (0‑5)")
    ap.add_argument("subject", type=int, nargs="?", help="Subject id (1‑109)")
    ap.add_argument("mode", choices=["train", "predict", "stream"], nargs="?")
    ap.add_argument("--fast", action="store_true", help="Use 10 subjects only")
    args = ap.parse_args()

    if not args.data_path.exists():
        ap.error(f"Data dir {args.data_path} not found")

    if args.experiment is None:
        train_split(None, args.data_path, not args.fast)
        return

    if args.subject is None:
        train_split(args.experiment, args.data_path, not args.fast)
        return

    if args.mode is None:
        ap.error("Need a mode (train/predict/stream) for single subject action")

    if args.mode == "train":
        train_subject(args.experiment, args.subject, args.data_path)
    elif args.mode == "predict":
        predict_subject(
            args.experiment, args.subject, args.data_path, playback=True
        )
    else:
        stream_subject(args.experiment, args.subject, args.data_path)


if __name__ == "__main__":
    main()
