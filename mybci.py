import argparse
import os
import numpy as np
import mne

from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


FMIN, FMAX = 7, 30  # Filter band
TMIN, TMAX = 0, 4
N_COMPONENTS = 3  # cZ C3 C4
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def _load_data(exp: int, subj: int):
    runs = eegbci.runs_for_experiment(exp)
    files = eegbci.load_data(exp, subj, runs, verbose=False)
    raws = [read_raw(f, preload=True, verbose=False) for f in files]
    raws = concatenate_raws(raws)

    eegbci.standardize(raws)
    raw.set_montage(make_standard_montage("standard_1005"))

    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)

    events = mne.events_from_annotations(raw)
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
    )
    x = epochs.get_data()
    y = epochs.events[:, 2]
    return x, y


def _build__pipeline():
    csp = CSP(
        n_components=N_COMPONENTS,
        reg=None,
        log=True,
        norm_trace=False,
    )
    lda = LDA()
    return Pipeline(["CSP", csp], ["LDA", lda])


def predict_stream(exp: int, subj: int, model_path: str):
    pipe = joblib.load(model_path)
    X, y_true = _load_data(exp, subj)

    correct = 0
    print("epoch nb: [prediction] [truth] equal?")
    for i in range(len(X)):
        pred = pipe.predict(X[i : i + 1])[0]
        truth = y_true[i]
        equal = pred == truth
        correct += int(equal)
        print(f"epoch {i:02d}: [{pred}] [{truth}] {equal}")
    accuracy = correct / len(X)
    print(f"Accuracy: {accuracy:.4f}")


def train(exp: int, subj: int, model_path: str):
    X, y = _load_data(exp, subj)
    pipe = _build__pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1)
    print(scores)
    print(f"cross_val_score: {scores.mean():.4f}")


def run_all():
    experiments = range(6)
    subjects = range(110)
    accuracies = {e: [] for e in experiments}

    for exp in experiments:
        for subj in subjects:
            try:
                X, y = _load_data(exp, subj)
            except Exception:
                continue
            pipe = _build_pipeline()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1).mean()
            accuracies[exp].append(score)
            print(
                f"experiment {exp}: subject {subj:03d}: accuracy = {score:.3f}"
            )

    for exp in experiments:
        if accuracies[exp]:
            mean_acc = np.mean(accuracies[exp])
            print(f"experiment {exp}: accuracy = {mean_acc:.4f}")
    global_mean = np.mean([np.mean(v) for v in accuracies.values() if v])
    print(f"\nMean accuracy of 6 experiments: {global_mean:.4f}")

    experiments = range(6)
    accuracies = []

    for i in experiments:
        if accuracies:
            mean_accuracy = np.nanmean(accuracies)
            print(f"Mean accuracy of {len(accuracies)} experiments: {mean_accuracy:.2f}")
        else:
            print("No valid accuracies to compute mean.")


def main():
    parser = argparse.ArgumentParser(description="Simple CSP-LDA model for BCI")
    parser.add_argument(
        "experiment", type=int, nargs="?", help="Experiment id (0-5)"
    )
    parser.add_argument(
        "subject", type=int, nargs="?", help="Subject id (0-109)"
    )
    parser.add_argument(
        "mode",
        type=str,
        nargs="?",
        choices=["train", "predict", "crossval"],
        help="Operation mode (train, predict, crossval)",
    )

    args = parser.parse_args()

    if args.experiment is None or args.subject is None or args.mode is None:
        run_all()
        return

    model_path = args.model_path or os.path.join(
        MODEL_DIR,
        f"bci_experiment_{args.experiment}_subject_{args.subject}.pkl",
    )

    if args.mode == "train":
        train(args.experiment, args.subject, model_path)
    elif args.mode == "crossval":
        evaluate_crossval(args.experiment, args.subject, model_path)
    elif args.mode == "predict":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train it first with 'train' mode."
            )
        predict_stream(args.experiment, args.subject, model_path)


if __name__ == "__main__":
    main()
