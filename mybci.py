import argparse
from pathlib import Path
import numpy as np
import joblib
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)

FMIN, FMAX = 7.0, 30.0
TMIN, TMAX = 0.0, 4.0
N_CSP = 3
HOLD_OUT = 0.2
SEED = 42

EXPERIMENT_RUNS = {
    0: [3, 7, 11],  # L/R execution
    1: [4, 8, 12],  # L/R imagery
    2: [5, 9, 13],  # Hand/Feet execution
    3: [6, 10, 14],  # Hand/Feet imagery
    4: [3, 4, 7, 8, 11, 12],  # L/R mixed (exec + imag)
    5: [5, 6, 9, 10, 13, 14],  # H/F mixed (exec + imag)
}

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------


def load_data(exp: int, subj: int):
    runs = EXPERIMENT_RUNS[exp]
    files = eegbci.load_data(subj, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    events, _ = mne.events_from_annotations(raw)
    event_id = {"hands": 2, "feet": 3, "left": 1, "right": 2}
    # For L/R runs T1=left hand, T2=right; for H/F runs T1=hands, T2=feet
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
    y = epochs.events[:, 2]
    # Map left/right to 0/1 and hands/feet to 0/1 so classes always 0/1
    y = (y % 2).astype(int)  # 1→1%2=1;2→0 ;3→1
    if len(X) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Dataset has <2 classes")
    return X, y


def build_pipeline(n_csp=N_CSP):
    return Pipeline(
        [
            (
                "CSP",
                CSP(n_components=n_csp, reg=None, log=True, norm_trace=False),
            ),
            ("LDA", LDA()),
        ]
    )


# -------------------------------------------------------------------------


def train(exp: int, subj: int):
    X, y = load_data(exp, subj)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=HOLD_OUT, random_state=SEED
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=1)
    print(
        f"Exp {exp} Sub {subj}: CV scores {scores.round(3)} mean={scores.mean():.3f}"
    )

    pipe.fit(X_train, y_train)
    path = MODEL_DIR / f"bci_exp{exp}_sub{subj}.pkl"
    joblib.dump({"model": pipe, "hold_idx": test_idx}, path)
    print(f"Model saved → {path} (hold-out {len(test_idx)} epochs)")


def predict(exp: int, subj: int):
    path = MODEL_DIR / f"bci_exp{exp}_sub{subj}.pkl"
    obj = joblib.load(path)
    pipe, test_idx = obj["model"], obj["hold_idx"]
    X, y = load_data(exp, subj)
    preds = pipe.predict(X[test_idx])
    acc = (preds == y[test_idx]).mean()
    print(f"Exp {exp} Sub {subj}: hold-out accuracy = {acc:.3f}")


# -------------------------------------------------------------------------


def train_all(subj: int):
    for exp in range(6):
        try:
            train(exp, subj)
        except Exception as e:
            print(f"Skip exp {exp}: {e}")


def predict_all(subj: int):
    accs = []
    for exp in range(6):
        try:
            path = MODEL_DIR / f"bci_exp{exp}_sub{subj}.pkl"
            if path.exists():
                obj = joblib.load(path)
                pipe, test_idx = obj["model"], obj["hold_idx"]
                X, y = load_data(exp, subj)
                acc = (pipe.predict(X[test_idx]) == y[test_idx]).mean()
                accs.append(acc)
                print(f"Exp {exp}: {acc:.3f}")
            else:
                print(f"Model for exp {exp} missing")
        except Exception as e:
            print(f"Exp {exp} skipped: {e}")
    if accs:
        print(f"Mean over 6 experiments: {np.mean(accs):.3f}")


# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("TPV 6‑model trainer")
    parser.add_argument(
        "mode", choices=["train", "predict"], nargs="?", default=None
    )
    parser.add_argument(
        "experiment", type=int, nargs="?", help="0‑5 (optional)"
    )
    parser.add_argument("subject", type=int, nargs="?", default=1)
    args = parser.parse_args()

    if args.mode is None:  # no arg → run full loop on subject 1
        print("No arguments → training + evaluating 6 models for subject 1")
        train_all(args.subject)
        predict_all(args.subject)
        return

    if args.experiment is None:
        if args.mode == "train":
            train_all(args.subject)
        else:
            predict_all(args.subject)
    else:
        if args.mode == "train":
            train(args.experiment, args.subject)
        else:
            predict(args.experiment, args.subject)


if __name__ == "__main__":
    main()
