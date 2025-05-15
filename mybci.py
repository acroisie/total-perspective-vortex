import argparse
from pathlib import Path

import joblib
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

# ----------------------------- hyper‑params -------------------------------
FMIN, FMAX = 7.0, 30.0
TMIN, TMAX = 0.0, 4.0
N_CSP = 3
TEST_FRAC = 0.2  # 20 % hold‑out
RANDOM_STATE = 42

EXPERIMENT_RUNS = {
    0: [1],
    1: [2],
    2: [3, 7, 11],
    3: [4, 8, 12],
    4: [5, 9, 13],
    5: [6, 10, 14],
}
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------


def load_data(exp: int, subj: int):
    runs = EXPERIMENT_RUNS.get(exp)
    files = eegbci.load_data(subj, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw,
        events,
        {"hands": 2, "feet": 3},
        TMIN,
        TMAX,
        baseline=None,
        detrend=1,
        preload=True,
        verbose=False,
    )
    X = epochs.get_data().astype(np.float64)
    y = epochs.events[:, 2]
    if len(X) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Not enough epochs or only one class present.")
    return X, y


def build_pipeline(n_csp: int = N_CSP):
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

    # --- split 80/20 hold‑out -------------------------------------------
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRAC, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = (
        X[test_idx],
        y[test_idx],
    )  # réservé, jamais vu pendant le fit

    # --- 5‑fold CV sur les 80 % -----------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline()
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=1)
    print("CV scores:", cv_scores, "| mean=", cv_scores.mean())

    # --- entraînement final sur les 80 % --------------------------------
    pipe.fit(X_train, y_train)
    model_path = MODEL_DIR / f"bci_exp{exp}_sub{subj}.pkl"
    joblib.dump({"model": pipe, "holdout_idx": test_idx}, model_path)
    print(f"Model saved → {model_path} (hold‑out size={len(test_idx)})")


def predict(exp: int, subj: int):
    model_path = MODEL_DIR / f"bci_exp{exp}_sub{subj}.pkl"
    obj = joblib.load(model_path)
    pipe, test_idx = obj["model"], obj["holdout_idx"]

    X, y = load_data(exp, subj)
    X_test, y_test = X[test_idx], y[test_idx]

    preds = pipe.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Hold‑out accuracy on {len(test_idx)} epochs: {acc:.4f}")

    for i, (p, t) in enumerate(zip(preds, y_test)):
        print(f"epoch {i:02d}: pred={p} true={t} ok={p==t}")


# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("TPV train / predict (80‑20 split + CV)")
    parser.add_argument("experiment", type=int)
    parser.add_argument("subject", type=int)
    parser.add_argument("mode", choices=["train", "predict"])
    args = parser.parse_args()

    if args.mode == "train":
        train(args.experiment, args.subject)
    else:
        predict(args.experiment, args.subject)


if __name__ == "__main__":
    main()
