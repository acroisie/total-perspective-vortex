import argparse
from pathlib import Path
import numpy as np
import joblib
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from custom_csp import CustomCSP

FMIN, FMAX = 7.0, 30.0
TMIN, TMAX = 0.0, 4.0
N_CSP = 3
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

# Global variable to store data path
DATA_PATH = None

# -------------------------------------------------------------------------


def load_data_from_physionet(exp: int, subj: int):
    """Load data from PhysioNet using MNE's eegbci dataset"""
    runs = EXPERIMENT_RUNS[exp]
    files = eegbci.load_data(subj, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    events, _ = mne.events_from_annotations(raw)
    event_id = {"hands": 2, "feet": 3, "left": 1, "right": 2}
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
    y = (y % 2).astype(int)
    if len(X) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Dataset has <2 classes")
    return X, y


def load_data_from_files(exp: int, subj: int):
    """Load data from local files"""
    if DATA_PATH is None:
        raise ValueError("Data path not set. Use --data option.")
    
    runs = EXPERIMENT_RUNS[exp]
    subj_dir = DATA_PATH / f"S{subj:03d}"
    
    if not subj_dir.exists():
        raise ValueError(f"Subject directory {subj_dir} not found")
    
    files = []
    for run in runs:
        edf_file = subj_dir / f"S{subj:03d}R{run:02d}.edf"
        if edf_file.exists():
            files.append(str(edf_file))
        else:
            print(f"Warning: {edf_file} not found, skipping")
    
    if not files:
        raise ValueError(f"No EDF files found for subject {subj} experiment {exp}")
    
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    
    # Apply standard preprocessing
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    events, _ = mne.events_from_annotations(raw)
    
    # Determine event_id based on experiment type
    if exp in [0, 1, 4]:  # L/R experiments
        event_id = {"left": 1, "right": 2}
    else:  # H/F experiments
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
    y = epochs.events[:, 2]
    # Map to binary classes
    y = (y % 2).astype(int)
    
    if len(X) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Dataset has <2 classes")
    return X, y


def load_data(exp: int, subj: int):
    """Load data either from PhysioNet or local files"""
    if DATA_PATH is not None:
        return load_data_from_files(exp, subj)
    else:
        return load_data_from_physionet(exp, subj)


def build_pipeline(n_csp=N_CSP):
    return Pipeline(
        [
            (
                "CSP",
                CustomCSP(n_components=n_csp, reg=None, log=True, norm_trace=False),
            ),
            ("LDA", LDA()),
        ]
    )


def collect_all_data(exp: int, subjects=None, use_full_dataset=False):
    """Collect data from all subjects for training a single model"""
    if subjects is None:
        if use_full_dataset:
            # Use all 109 subjects for full dataset
            subjects = range(1, 110)
            print(f"Using full dataset: {len(subjects)} subjects")
        else:
            # Use first 10 subjects for quick testing
            subjects = range(1, 11)
            print(f"Using subset: {len(subjects)} subjects")  # Quick test with 10 subjects
    
    all_X, all_y = [], []
    valid_subjects = []
    
    for subj in subjects:
        try:
            X, y = load_data(exp, subj)
            all_X.append(X)
            all_y.append(y)
            valid_subjects.append(subj)
        except Exception as e:
            print(f"Skip subject {subj}: {e}")
    
    if not all_X:
        raise ValueError(f"No valid data found for experiment {exp}")
    
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    print(f"Experiment {exp}: collected {len(X_combined)} epochs from {len(valid_subjects)} subjects")
    return X_combined, y_combined, valid_subjects


# -------------------------------------------------------------------------


def train_experiment(exp: int, use_full_dataset=False):
    """Train a single model for one experiment using all subjects"""
    print(f"\n=== Training experiment {exp} ===")
    
    X, y, valid_subjects = collect_all_data(exp, use_full_dataset=use_full_dataset)
    
    # Cross-validation
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=1)
    
    print(f"Cross-validation scores: {scores}")
    print(f"cross_val_score: {scores.mean():.4f}")
    
    # Train final model
    pipe.fit(X, y)
    
    # Save model
    model_path = MODEL_DIR / f"bci_exp{exp}.pkl"
    joblib.dump({
        "model": pipe, 
        "valid_subjects": valid_subjects,
        "cv_scores": scores,
        "mean_cv_score": scores.mean()
    }, model_path)
    
    print(f"Model saved to {model_path}")
    return scores.mean()


def predict_subject(exp: int, subj: int):
    """Predict on a specific subject using the trained model"""
    model_path = MODEL_DIR / f"bci_exp{exp}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model for experiment {exp} not found. Train first.")
    
    # Load model
    data = joblib.load(model_path)
    pipe = data["model"]
    
    # Load subject data
    X, y = load_data(exp, subj)
    
    # Predict
    predictions = pipe.predict(X)
    
    # Output format as required (map 0/1 back to 1/2)
    predictions_mapped = predictions + 1
    y_mapped = y + 1
    
    print(f"epoch nb: [prediction] [truth] equal?")
    correct = 0
    for i, (pred, truth) in enumerate(zip(predictions_mapped, y_mapped)):
        is_correct = pred == truth
        if is_correct:
            correct += 1
        print(f"epoch {i:02d}: [{pred}] [{truth}] {is_correct}")
    
    accuracy = correct / len(y)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def train_all_experiments(use_full_dataset=False):
    """Train models for all 6 experiments"""
    dataset_info = "full dataset (109 subjects)" if use_full_dataset else "subset (10 subjects)"
    print(f"Training all 6 experiments on {dataset_info}...")
    accuracies = []
    
    for exp in range(6):
        try:
            acc = train_experiment(exp, use_full_dataset=use_full_dataset)
            accuracies.append(acc)
        except Exception as e:
            print(f"Failed to train experiment {exp}: {e}")
    
    if accuracies:
        print(f"\nMean cross-validation accuracy across experiments: {np.mean(accuracies):.4f}")


def evaluate_all_experiments():
    """Evaluate all experiments on all subjects"""
    print("Evaluating all experiments on all subjects...")
    
    exp_accuracies = {}
    
    for exp in range(6):
        model_path = MODEL_DIR / f"bci_exp{exp}.pkl"
        if not model_path.exists():
            print(f"experiment {exp}: model not found")
            continue
            
        data = joblib.load(model_path)
        pipe = data["model"]
        valid_subjects = data.get("valid_subjects", range(1, 110))
        
        subject_accs = []
        for subj in valid_subjects:
            try:
                X, y = load_data(exp, subj)
                predictions = pipe.predict(X)
                acc = (predictions == y).mean()
                subject_accs.append(acc)
                print(f"experiment {exp}: subject {subj:03d}: accuracy = {acc:.1f}")
            except Exception as e:
                print(f"experiment {exp}: subject {subj:03d}: error = {e}")
        
        if subject_accs:
            exp_mean = np.mean(subject_accs)
            exp_accuracies[exp] = exp_mean
            print(f"experiment {exp}: accuracy = {exp_mean:.4f}")
        else:
            print(f"experiment {exp}: no valid subjects")
    
    if exp_accuracies:
        overall_mean = np.mean(list(exp_accuracies.values()))
        print(f"Mean accuracy of {len(exp_accuracies)} experiments: {overall_mean:.4f}")


def predict_subject_stream(exp: int, subj: int, delay=2.0):
    """
    Predict on a specific subject with stream simulation (playback with delay).
    Simulates real-time BCI by introducing a delay between epochs.
    """
    import time
    
    model_path = MODEL_DIR / f"bci_exp{exp}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model for experiment {exp} not found. Train first.")
    
    # Load model
    data = joblib.load(model_path)
    pipe = data["model"]
    
    # Load subject data
    X, y = load_data(exp, subj)
    
    print(f"Starting stream simulation for experiment {exp}, subject {subj}")
    print(f"Processing {len(X)} epochs with {delay}s delay between predictions...")
    print(f"epoch nb: [prediction] [truth] equal?")
    
    correct = 0
    start_time = time.time()
    
    for i, (epoch_data, truth) in enumerate(zip(X, y)):
        # Simulate processing time delay
        if i > 0:  # No delay for first epoch
            time.sleep(delay)
        
        # Predict on single epoch (reshape to add batch dimension)
        epoch_reshaped = epoch_data.reshape(1, *epoch_data.shape)
        pred = pipe.predict(epoch_reshaped)[0]
        
        # Map 0/1 back to 1/2 for output
        pred_mapped = pred + 1
        truth_mapped = truth + 1
        
        is_correct = pred == truth
        if is_correct:
            correct += 1
            
        elapsed = time.time() - start_time
        print(f"epoch {i:02d}: [{pred_mapped}] [{truth_mapped}] {is_correct} (t={elapsed:.1f}s)")
    
    accuracy = correct / len(y)
    total_time = time.time() - start_time
    print(f"Stream simulation completed in {total_time:.1f}s")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


# -------------------------------------------------------------------------


def main():
    global DATA_PATH
    
    parser = argparse.ArgumentParser(
        description="Total Perspective Vortex - BCI Motor Imagery Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mybci.py                     # Train all 6 models then evaluate
  python mybci.py --full              # Train all 6 models on full dataset (109 subjects)
  python mybci.py 4 14 train         # Train experiment 4, test on subject 14
  python mybci.py 4 14 predict       # Predict experiment 4 on subject 14
  python mybci.py 4 14 stream        # Stream simulation for experiment 4, subject 14
  python mybci.py --data files       # Use local data files
        """
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        help="Path to local data directory (containing S001/, S002/, etc.)"
    )
    
    parser.add_argument(
        "experiment", 
        type=int, 
        nargs="?", 
        help="Experiment number (0-5)"
    )
    
    parser.add_argument(
        "subject", 
        type=int, 
        nargs="?", 
        help="Subject number (1-109)"
    )
    
    parser.add_argument(
        "mode", 
        choices=["train", "predict", "stream"], 
        nargs="?", 
        help="Mode: train, predict, or stream (real-time simulation)"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset (all 109 subjects) instead of subset"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds for stream simulation (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Set data path if provided
    if args.data:
        DATA_PATH = Path(args.data)
        if not DATA_PATH.exists():
            print(f"Error: Data path {DATA_PATH} does not exist")
            return
        print(f"Using local data from: {DATA_PATH}")
    
    # Case 1: No arguments - train all then evaluate
    if args.experiment is None:
        if args.full:
            print("Training all 6 models on full dataset (109 subjects)...")
        else:
            print("Training all 6 models on subset (10 subjects)...")
        train_all_experiments(use_full_dataset=args.full)
        evaluate_all_experiments()
        return
    
    # Case 2: Only experiment provided
    if args.subject is None:
        print(f"Training experiment {args.experiment} on all subjects...")
        train_experiment(args.experiment)
        return
    
    # Case 3: Experiment and subject provided
    if args.mode is None:
        print("Mode not specified. Use 'train' or 'predict'")
        return
    
    if args.mode == "train":
        # For compatibility: train on single subject and test immediately
        print(f"Training experiment {args.experiment} on subject {args.subject}...")
        try:
            X, y = load_data(args.experiment, args.subject)
            pipe = build_pipeline()
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
            scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=1)
            print(scores.round(4))
            print(f"cross_val_score: {scores.mean():.4f}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.mode == "predict":
        try:
            predict_subject(args.experiment, args.subject)
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.mode == "stream":
        try:
            predict_subject_stream(args.experiment, args.subject, delay=args.delay)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
