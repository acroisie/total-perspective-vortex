import argparse
from pathlib import Path
import numpy as np
import joblib
import time
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from custom_csp import CustomCSP
import warnings
import logging

# Désactivation des warnings Python
def disable_all_logs():
    warnings.filterwarnings("ignore")
    mne.set_log_level('ERROR')
    logging.getLogger('mne').setLevel(logging.ERROR)
    logging.getLogger('mne').propagate = False
    logging.getLogger().setLevel(logging.ERROR)
    try:
        joblib_logger = logging.getLogger('joblib')
        joblib_logger.setLevel(logging.ERROR)
        joblib_logger.propagate = False
    except Exception:
        pass

# Appel dès le début du script
disable_all_logs()

FMIN, FMAX = 8, 32.0
TMIN, TMAX = 0.7, 3.9
SEED = 42

FAST_SUBJECTS = 10  # Nombre de sujets utilisés en mode --fast
PREDICT_DELAY = 0.1  # Délai en secondes entre chaque époque en mode predict (100ms)


EXPERIMENT_RUNS = {
    0: [3, 7, 11],  # L/R execution
    1: [4, 8, 12],  # L/R imagery
    2: [5, 9, 13],  # Hand/Feet execution
    3: [6, 10, 14],  # Hand/Feet imagery
    4: [3, 4, 7, 8, 11, 12],  # L/R mixed (exec + imag)
    5: [5, 6, 9, 10, 13, 14],  # H/F mixed (exec + imag)
}

# Constante pour les canaux EEG utilisés (zones motrices étendues)
# # Fast : 0.66 -- 2min45 -- OK
EEG_CHANNELS = ["C3", "C4", "Cz"]
N_CSP = 3

# EEG_CHANNELS = [
#     "C3", "C4", "Cz",     
#     "FC3", "FC4", 
#     "CP3", "CP4" ]
# N_CSP = 6

USE_FILTERBANK = True  # Utiliser FilterBankCSP ou CustomCSP

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Global variable to store data path
DATA_PATH = None

# Define FilterBankCSPOfficial at module level to avoid pickle issues
from filterbank_csp import FilterBankCSP
from custom_csp import CustomCSP

from mne.decoding import CSP
class FilterBankCSPCustom(FilterBankCSP):
    def fit(self, X, y):
        # Assurer que X est en float64 pour éviter les erreurs
        X = np.asarray(X, dtype=np.float64)
        self.csp_list = []
        for fmin, fmax in self.freq_bands:
            X_f = self._bandpass_filter(X, fmin, fmax)
            csp = CustomCSP(n_components=self.n_csp, log=True)
            csp.fit(X_f, y)
            self.csp_list.append(csp)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return super().transform(X)

# -------------------------------------------------------------------------



def load_data_from_physionet(exp: int, subj: int):
    """Load data from PhysioNet using MNE's eegbci dataset"""
    runs = EXPERIMENT_RUNS[exp]
    files = eegbci.load_data(subj, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"), verbose=False)
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    picks = mne.pick_channels(raw.info["ch_names"], include=EEG_CHANNELS)
    raw.pick(picks)
    events, _ = mne.events_from_annotations(raw, verbose=False)
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
    # Harmonize epoch length
    min_len = X.shape[2]
    if X.shape[0] > 0:
        min_len = min([X.shape[2]] + [X.shape[2] for X in [X]])
    X = X[:, :, :min_len]
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
        raise ValueError(
            f"No EDF files found for subject {subj} experiment {exp}"
        )

    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)

    # Apply standard preprocessing
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    picks = mne.pick_channels(raw.info["ch_names"], include=EEG_CHANNELS)
    raw.pick(picks)
    events, _ = mne.events_from_annotations(raw, verbose=False)

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
    # Harmonize epoch length
    min_len = X.shape[2]
    if X.shape[0] > 0:
        min_len = min([X.shape[2]] + [X.shape[2] for X in [X]])
    X = X[:, :, :min_len]
    return X, y


def load_data(exp: int, subj: int):
    """Load data either from PhysioNet or local files"""
    if DATA_PATH is not None:
        return load_data_from_files(exp, subj)
    else:
        return load_data_from_physionet(exp, subj)


def build_pipeline(n_csp=N_CSP, use_filterbank=USE_FILTERBANK, sfreq=160):
    if use_filterbank:
        print(f"Using FilterBankCSP with n_csp={n_csp}, sfreq={sfreq}")
        return Pipeline([
            ("FBCSP", FilterBankCSPCustom(n_csp=n_csp, sfreq=sfreq)),
            ("LDA", LDA()),
        ])
    else:
        print(f"Using CustomCSP with n_components={n_csp}")
        return Pipeline([
            ("CSP", CustomCSP(n_components=n_csp, log=True)),
            ("LDA", LDA()),
        ])


def collect_all_data(exp: int, subjects=None, use_full_dataset=False):
    """Collect data from all subjects for training a single model (epochs harmonized in time)"""
    if subjects is None:
        if not use_full_dataset:
            subjects = range(1, 11)
            print(f"Using subset: {len(subjects)} subjects")
        else:
            subjects = range(1, 110)
            print(f"Using full dataset: {len(subjects)} subjects")

    all_X, all_y = [], []
    valid_subjects = []
    epoch_lengths = []

    # First pass: get all data and record epoch lengths
    for subj in subjects:
        try:
            X, y = load_data(exp, subj)
            if X.shape[0] > 0:
                all_X.append(X)
                all_y.append(y)
                valid_subjects.append(subj)
                epoch_lengths.append(X.shape[2])
        except Exception as e:
            print(f"Skip subject {subj}: {e}")

    if not all_X:
        raise ValueError(f"No valid data found for experiment {exp}")

    # Harmonize all epochs to the minimal length across all subjects
    min_len = min(epoch_lengths)
    all_X = [X[:, :, :min_len] for X in all_X]

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    print(f"Experiment {exp}: collected {len(X_combined)} epochs from {len(valid_subjects)} subjects (epoch length: {min_len})")
    return X_combined, y_combined, valid_subjects


# -------------------------------------------------------------------------


def predict_subject(exp: int, subj: int):
    """Predict on a specific subject using the trained model (subject-specific first, then global)"""
    # Try subject-specific model first
    subject_model_path = MODEL_DIR / f"bci_exp{exp}_subj{subj:03d}.pkl"
    global_model_path = MODEL_DIR / f"bci_exp{exp}.pkl"
    split_model_path = MODEL_DIR / f"bci_exp{exp}_split.pkl"
    
    model_path = None
    model_type = None
    
    if subject_model_path.exists():
        model_path = subject_model_path
        model_type = "subject-specific"
    elif global_model_path.exists():
        model_path = global_model_path
        model_type = "global"
    elif split_model_path.exists():
        model_path = split_model_path
        model_type = "split"
    else:
        raise FileNotFoundError(
            f"No model found for experiment {exp} and subject {subj}. "
            f"Looked for:\n- Subject-specific: {subject_model_path}\n- Global: {global_model_path}\n- Split: {split_model_path}\n"
            f"Train first with: python mybci.py {exp} {subj} train"
        )

    print(f"Using {model_type} model: {model_path}")

    # Load model
    data = joblib.load(model_path)
    if isinstance(data, dict) and "model" in data:
        pipe = data["model"]
    else:
        # For compatibility with older model files
        pipe = data

    # Load subject data
    X, y = load_data(exp, subj)

    # Predict
    predictions = pipe.predict(X)

    # Output format as required (map 0/1 back to 1/2)
    predictions_mapped = predictions + 1
    y_mapped = y + 1

    print(f"epoch nb: [prediction] [truth] equal?")
    correct = 0
    start_time = time.time()
    
    for i, (pred, truth) in enumerate(zip(predictions_mapped, y_mapped)):
        # Add delay between epochs (except for the first one)
        if i > 0:
            time.sleep(PREDICT_DELAY)
        
        is_correct = pred == truth
        if is_correct:
            correct += 1
        
        elapsed = time.time() - start_time
        print(f"epoch {i:02d}: [{pred}] [{truth}] {is_correct} (t={elapsed:.1f}s)")

    accuracy = correct / len(y)
    total_time = time.time() - start_time
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total prediction time: {total_time:.1f}s")
    return accuracy

def evaluate_all_experiments():
    """Evaluate all experiments on all subjects (format conforme au sujet)"""
    start_time = time.time()
    
    print("Evaluating all experiments on all subjects...")

    exp_accuracies = {}
    all_exp_subject_accs = {}
    cross_val_scores = {}

    for exp in range(6):
        model_path = MODEL_DIR / f"bci_exp{exp}.pkl"
        if not model_path.exists():
            print(f"experiment {exp}: model not found")
            continue

        data = joblib.load(model_path)
        pipe = data["model"]
        valid_subjects = data.get("valid_subjects", range(1, 110))
        
        # Récupérer et stocker le cross-validation score
        if "mean_cv_score" in data:
            cross_val_scores[exp] = data["mean_cv_score"]
        elif "cv_scores" in data:
            cross_val_scores[exp] = np.mean(data["cv_scores"])

        subject_accs = []
        for subj in valid_subjects:
            try:
                X, y = load_data(exp, subj)
                predictions = pipe.predict(X)
                acc = (predictions == y).mean()
                subject_accs.append(acc)
                # Format plus simple comme demandé dans le sujet
                print(f"experiment {exp}: subject {subj:03d}: accuracy = {acc:.1f}")
            except Exception as e:
                print(f"experiment {exp}: subject {subj:03d}: error = {e}")

        if subject_accs:
            exp_mean = np.mean(subject_accs)
            exp_accuracies[exp] = exp_mean
            all_exp_subject_accs[exp] = subject_accs
            print(f"experiment {exp}: accuracy = {exp_mean:.4f}")
        else:
            print(f"experiment {exp}: no valid subjects")

    if exp_accuracies:
        print("\nMean accuracy of the six different experiments for all 109 subjects:")
        for exp in range(6):
            if exp in exp_accuracies:
                print(f"experiment {exp}: accuracy = {exp_accuracies[exp]:.4f}")
                if exp in cross_val_scores:
                    print(f"experiment {exp}: cross_val_score = {cross_val_scores[exp]:.4f}")
        
        overall_mean = np.mean(list(exp_accuracies.values()))
        print(f"Mean accuracy of 6 experiments: {overall_mean:.4f}")
        
        if cross_val_scores:
            cv_mean = np.mean(list(cross_val_scores.values()))
            print(f"Mean cross_val_score of experiments: {cv_mean:.4f}")
    
    # Afficher le temps total d'exécution
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nTemps total d'évaluation: {minutes}:{seconds:02d} (min:ss)")


def predict_subject_stream(exp: int, subj: int, delay=2.0):
    """
    Predict on a specific subject with stream simulation (playback with delay).
    Simulates real-time BCI by introducing a delay between epochs.
    """

    # Try subject-specific model first, then global models
    subject_model_path = MODEL_DIR / f"bci_exp{exp}_subj{subj:03d}.pkl"
    global_model_path = MODEL_DIR / f"bci_exp{exp}.pkl"
    split_model_path = MODEL_DIR / f"bci_exp{exp}_split.pkl"
    
    model_path = None
    model_type = None
    
    if subject_model_path.exists():
        model_path = subject_model_path
        model_type = "subject-specific"
    elif global_model_path.exists():
        model_path = global_model_path
        model_type = "global"
    elif split_model_path.exists():
        model_path = split_model_path
        model_type = "split"
    else:
        raise FileNotFoundError(
            f"No model found for experiment {exp} and subject {subj}. "
            f"Train first with: python mybci.py {exp} {subj} train"
        )

    print(f"Using {model_type} model: {model_path}")

    # Load model
    data = joblib.load(model_path)
    if isinstance(data, dict) and "model" in data:
        pipe = data["model"]
    else:
        # For compatibility with older model files
        pipe = data

    # Load subject data
    X, y = load_data(exp, subj)

    print(f"Starting stream simulation for experiment {exp}, subject {subj}")
    print(
        f"Processing {len(X)} epochs with {delay}s delay between predictions..."
    )
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
        print(
            f"epoch {i:02d}: [{pred_mapped}] [{truth_mapped}] {is_correct} (t={elapsed:.1f}s)"
        )

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
  python mybci.py --data /path/to/data             # Train all 6 models with subject split approach
  python mybci.py --data /path/to/data --fast     # Train all 6 models with subject split on reduced dataset (10 subjects)
  python mybci.py 4 14 train                      # Train experiment 4, test on subject 14 (single subject)
  python mybci.py 4 14 predict                    # Predict experiment 4 on subject 14
  python mybci.py 4 14 stream                     # Stream simulation for experiment 4, subject 14
  python mybci.py 4 --data /path/to/data          # Train only experiment 4 with subject split
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        help="Path to local data directory (containing S001/, S002/, etc.)",
    )

    parser.add_argument(
        "experiment", type=int, nargs="?", help="Experiment number (0-5)"
    )

    parser.add_argument(
        "subject", type=int, nargs="?", help="Subject number (1-109)"
    )

    parser.add_argument(
        "mode",
        choices=["train", "predict", "stream"],
        nargs="?",
        help="Mode: train, predict, or stream (real-time simulation)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a reduced subset (10 subjects) for faster training/evaluation (default: use all subjects)",
    )

    args = parser.parse_args()

    # Set data path if provided
    if args.data:
        DATA_PATH = Path(args.data)
        if not DATA_PATH.exists():
            print(f"Error: Data path {DATA_PATH} does not exist")
            return
        print(f"Using local data from: {DATA_PATH}")

    # Case 1: No arguments - train all then evaluate (TOUJOURS avec split)
    if args.experiment is None:
        if DATA_PATH is None:
            print("Error: --data path is required for split training approach")
            print("Usage: python mybci.py --data /path/to/data [--fast]")
            return
        
        if args.fast:
            print("Training all experiments with subject split approach on reduced subset (10 subjects)...")
        else:
            print("Training all experiments with subject split approach on full dataset...")
        train_all_experiments_split(data_path=DATA_PATH, use_full_dataset=not args.fast)
        return

    # Case 2: Only experiment provided (TOUJOURS avec split)
    if args.subject is None:
        if DATA_PATH is None:
            print("Error: --data path is required for split training approach")
            print("Usage: python mybci.py --experiment exp_id --data /path/to/data [--fast]")
            return
        
        if args.fast:
            print(f"Training experiment {args.experiment} with subject split approach on reduced subset (10 subjects)...")
        else:
            print(f"Training experiment {args.experiment} with subject split approach on full dataset...")
        train_single_experiment_split(args.experiment, data_path=DATA_PATH, use_full_dataset=not args.fast)
        return

    # Case 3: Experiment and subject provided
    if args.mode is None:
        print("Mode not specified. Use 'train' or 'predict'")
        return

    if args.mode == "train":
        # Train on single subject and save subject-specific model
        print(
            f"Training experiment {args.experiment} on subject {args.subject}..."
        )
        try:
            X, y = load_data(args.experiment, args.subject)
            pipe = build_pipeline()
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
            scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=14)
            print(scores.round(4))
            print(f"cross_val_score: {scores.mean():.4f}")
            
            # Train final model on all data and save it
            pipe.fit(X, y)
            
            # Save subject-specific model with explicit naming
            model_path = MODEL_DIR / f"bci_exp{args.experiment}_subj{args.subject:03d}.pkl"
            joblib.dump({
                "model": pipe,
                "experiment": args.experiment,
                "subject": args.subject,
                "cv_scores": scores,
                "mean_cv_score": scores.mean(),
                "n_epochs": len(X),
                "data_shape": X.shape
            }, model_path)
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error: {e}")

    elif args.mode == "predict":
        try:
            predict_subject(args.experiment, args.subject)
        except Exception as e:
            print(f"Error: {e}")

    elif args.mode == "stream":
        try:
            predict_subject_stream(
                args.experiment, args.subject, delay=args.delay
            )
        except Exception as e:
            print(f"Error: {e}")


def train_single_experiment_split(exp: int, data_path=None, use_full_dataset=False):
    """
    Entraîne une seule expérience avec l'approche de split multi-sujets.
    """
    from sklearn.model_selection import train_test_split
    from utils_multi_subject import list_subject_dirs, aggregate_multi_subject_data

    if data_path is None:
        raise ValueError("data_path doit être spécifié pour le mode multi-sujets.")
    
    data_path = Path(data_path)
    subject_dirs = list_subject_dirs(data_path)
    # Limiter à FAST_SUBJECTS sujets si use_full_dataset == False (donc mode --fast)
    if not use_full_dataset:
        subject_dirs = subject_dirs[:FAST_SUBJECTS]
        print(f"[INFO] Mode rapide (--fast) : {len(subject_dirs)} sujets utilisés.")
    else:
        print(f"[INFO] {len(subject_dirs)} sujets trouvés.")

    # Split train/test/holdout : 60% / 20% / 20%
    train_dirs, tmp_dirs = train_test_split(subject_dirs, test_size=0.4, random_state=SEED)
    test_dirs, holdout_dirs = train_test_split(tmp_dirs, test_size=0.5, random_state=SEED)
    print(f"[INFO] Train: {len(train_dirs)} sujets, Test: {len(test_dirs)}, Holdout: {len(holdout_dirs)}")

    start_time = time.time()
    print(f"\n[INFO] ============ Expérience {exp} ============")
    
    try:
        X_train, y_train, valid_train = aggregate_multi_subject_data(train_dirs, exp, load_data)
        X_test, y_test, valid_test = aggregate_multi_subject_data(test_dirs, exp, load_data)
    except Exception as e:
        print(f"[ERROR] Pas de data pour exp={exp}: {e}")
        return
    
    if len(np.unique(y_train)) < 2:
        print(f"[ERROR] Expérience {exp} n'a qu'une seule classe en train.")
        return

    pipe = build_pipeline()
    
    try:
        # Cross-validation sur le train set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        print(f"[INFO] cross_val_scores={cv_scores}, mean={cv_scores.mean():.3f}")
    except Exception as e:
        print(f"[ERROR] Erreur cross_val pour exp={exp}: {e}")
        return
    
    # Entraînement final sur le train set
    pipe.fit(X_train, y_train)
    
    # Test sur le test set
    test_acc = pipe.score(X_test, y_test)
    print(f"[INFO] Test accuracy={test_acc:.3f}")
    
    # Sauvegarde du modèle
    model_filename = MODEL_DIR / f"bci_exp{exp}.pkl"
    joblib.dump({
        "model": pipe,
        "cv_scores": cv_scores,
        "test_acc": test_acc,
        "valid_train_subjects": valid_train,
        "valid_test_subjects": valid_test
    }, model_filename)
    print(f"[INFO] Modèle sauvegardé => {model_filename}")
    
    # Holdout evaluation
    try:
        X_hold, y_hold, _ = aggregate_multi_subject_data(holdout_dirs, exp, load_data)
        preds = pipe.predict(X_hold)
        hold_acc = np.mean(preds == y_hold)
        print(f"[INFO] Holdout accuracy exp={exp}: {hold_acc:.3f}")
    except Exception as e:
        print(f"[INFO] exp={exp}, pas de data holdout => skip. ({e})")
    
    elapsed = time.time() - start_time
    print(f"\n[INFO] Temps d'exécution pour exp {exp} : {int(elapsed//60)}:{int(elapsed%60):02d} (min:ss)")


def train_all_experiments_split(data_path=None, use_full_dataset=False):
    """
    Nouvelle version : split multi-sujets (train/test/holdout), aggregation, entraînement, validation, test, sauvegarde, reporting.
    """
    from sklearn.model_selection import train_test_split
    from utils_multi_subject import list_subject_dirs, aggregate_multi_subject_data

    if data_path is None:
        raise ValueError("data_path doit être spécifié pour le mode multi-sujets.")
    data_path = Path(data_path)
    subject_dirs = list_subject_dirs(data_path)
    # Limiter à FAST_SUBJECTS sujets si use_full_dataset == False (donc mode --fast)
    if not use_full_dataset:
        subject_dirs = subject_dirs[:FAST_SUBJECTS]
        print(f"[INFO] Mode rapide (--fast) : {len(subject_dirs)} sujets utilisés.")
    else:
        print(f"[INFO] {len(subject_dirs)} sujets trouvés.")

    # Split train/test/holdout : 60% / 20% / 20%
    train_dirs, tmp_dirs = train_test_split(subject_dirs, test_size=0.4, random_state=42)
    test_dirs, holdout_dirs = train_test_split(tmp_dirs, test_size=0.5, random_state=42)
    print(f"[INFO] Train: {len(train_dirs)} sujets, Test: {len(test_dirs)}, Holdout: {len(holdout_dirs)}")

    categories = list(range(6))
    models = {}
    results = {}
    start_time = time.time()
    for exp in categories:
        print(f"\n[INFO] ============ Expérience {exp} ============")
        try:
            X_train, y_train, valid_train = aggregate_multi_subject_data(train_dirs, exp, load_data)
            X_test, y_test, valid_test = aggregate_multi_subject_data(test_dirs, exp, load_data)
        except Exception as e:
            print(f"[WARN] Pas de data pour exp={exp}: {e}")
            continue
        if len(np.unique(y_train)) < 2:
            print(f"[WARN] Expérience {exp} n'a qu'une seule classe en train, skip.")
            continue
        pipe = build_pipeline()
        try:
            # Réduire la verbosité pendant le cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
            print(f"[INFO] cross_val_scores={cv_scores}, mean={cv_scores.mean():.3f}")
        except Exception as e:
            print(f"[ERROR] Erreur cross_val pour exp={exp}: {e}")
            continue
        pipe.fit(X_train, y_train)
        test_acc = pipe.score(X_test, y_test)
        print(f"[INFO] Test accuracy={test_acc:.3f}")
        models[exp] = pipe
        results[exp] = {"cv_scores": cv_scores, "test_acc": test_acc}
        model_filename = MODEL_DIR / f"bci_exp{exp}.pkl"
        joblib.dump(pipe, model_filename)
        print(f"[INFO] Modèle sauvegardé => {model_filename}")
    # Holdout
    holdout_accuracies = []
    for exp, pipe in models.items():
        try:
            X_hold, y_hold, _ = aggregate_multi_subject_data(holdout_dirs, exp, load_data)
        except Exception as e:
            print(f"[INFO] exp={exp}, pas de data holdout => skip. ({e})")
            continue
        preds = pipe.predict(X_hold)
        hold_acc = np.mean(preds == y_hold)
        print(f"[INFO] Holdout accuracy exp={exp}: {hold_acc:.3f}")
        holdout_accuracies.append(hold_acc)
    if holdout_accuracies:
        moyenne = np.mean(holdout_accuracies)
        print(f"\n[INFO] Moyenne holdout accuracy des modèles : {moyenne:.3f}")
    elapsed = time.time() - start_time
    print(f"\n[INFO] Temps total d'exécution : {int(elapsed//60)}:{int(elapsed%60):02d} (min:ss)")


if __name__ == "__main__":
    main()
