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
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ----------------------------- constants ---------------------------------
FMIN, FMAX = 7.0, 30.0    # band‑pass limits (Hz)
TMIN, TMAX = 0.0, 4.0      # epoch window (s)
N_COMPONENTS = 3           # CSP components to keep
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# mapping experiment id → PhysioNet runs
EXPERIMENT_RUNS = {
    0: [1],              # baseline open eyes
    1: [2],              # baseline closed eyes
    2: [3, 7, 11],       # motor execution hands
    3: [4, 8, 12],       # motor imagery hands
    4: [5, 9, 13],       # motor execution hands vs feet
    5: [6, 10, 14],      # motor imagery hands vs feet
}

# -------------------------------------------------------------------------
def load_data(experiment: int, subject: int):
    """
    Download & return (X, y) for one (experiment, subject).
    Raises ValueError if not enough epochs or single class.
    """
    if experiment not in EXPERIMENT_RUNS:
        raise ValueError("experiment must be in 0-5")

    runs = EXPERIMENT_RUNS[experiment]
    # By default, MNE will look into MNE_DATA folder; if you extracted the zip
    # put it under MNE_DATA/eegbci to reuse your local copy.
    file_paths = eegbci.load_data(subject, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in file_paths]
    raw = concatenate_raws(raws)

    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)

    events, _ = mne.events_from_annotations(raw)
    event_id = {"hands": 2, "feet": 3}
    epochs = mne.Epochs(
        raw, events, event_id,
        TMIN, TMAX,
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

# -------------------------------------------------------------------------
def build_pipeline():
    csp = CSP(n_components=N_COMPONENTS, reg=None, log=True, norm_trace=False)
    lda = LDA()
    return Pipeline([("CSP", csp), ("LDA", lda)])

# -------------------------------------------------------------------------
def cross_validate(experiment: int, subject: int):
    """Perform 5‑fold cross‑validation and print scores + mean."""
    X, y = load_data(experiment, subject)
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1, error_score="raise")
    print("CV scores:", scores)
    print(f"Mean CV accuracy: {scores.mean():.4f}")

# -------------------------------------------------------------------------
def train(experiment: int, subject: int, model_path: Path):
    X, y = load_data(experiment, subject)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

# -------------------------------------------------------------------------
def predict_stream(experiment: int, subject: int, model_path: Path):
    pipeline = joblib.load(model_path)
    X, y_true = load_data(experiment, subject)
    correct = 0
    for i, epoch in enumerate(X):
        pred = pipeline.predict(epoch[None, ...])[0]
        correct += int(pred == y_true[i])
        print(f"epoch {i:02d}: pred={pred} true={y_true[i]} ok={pred==y_true[i]}")
    print(f"Accuracy: {correct/len(X):.4f}")

# -------------------------------------------------------------------------
def benchmark_all():
    all_scores = []
    for exp in range(6):
        exp_scores = []
        for sub in range(1, 110):
            try:
                X, y = load_data(exp, sub)
            except ValueError:
                continue
            pipeline = build_pipeline()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1).mean()
            exp_scores.append(score)
            print(f"exp {exp} sub{sub:03d}: acc={score:.3f}")
        if exp_scores:
            print(f"Experiment {exp}: mean acc={np.mean(exp_scores):.4f}")
            all_scores.extend(exp_scores)
    if all_scores:
        print(f"\nGlobal mean accuracy: {np.mean(all_scores):.4f}")
    else:
        print("\nNo valid epochs found.")

# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BCI CSP/LDA script")
    parser.add_argument("experiment", type=int, nargs="?", help="experiment id 0-5")
    parser.add_argument("subject", type=int, nargs="?", help="subject id 1-109")
    parser.add_argument("mode", type=str, nargs="?", choices=["train", "predict", "crossval"], help="mode")
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    if args.experiment is None or args.subject is None or args.mode is None:
        benchmark_all()
        return

    model_path = args.model_path or MODEL_DIR / f"bci_exp{args.experiment}_sub{args.subject}.pkl"
    if args.mode == "train":
        train(args.experiment, args.subject, model_path)
    elif args.mode == "crossval":
        cross_validate(args.experiment, args.subject)
    elif args.mode == "predict":
        if not model_path.exists():
            raise FileNotFoundError("Model not found; run train first")
        predict_stream(args.experiment, args.subject, model_path)

if __name__ == "__main__":
    main()
