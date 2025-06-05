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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

MOTOR_CHANS = [
    "Cz","C3","C4","FC1","FC2","CP1","CP2",
    "FC3","FC4","CP3","CP4","C1","C2","CPz","FCz",
    "P3","P4","Pz","F3","F4","Fz","Oz",
]

###############################################################################
# CONSTANTS – keep them simple & visible
###############################################################################

FMIN, FMAX = 7.0, 30.0           # μ/β band‑pass
EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0  # 4‑second cue window
TARGET_FS = 128                  # unify sampling rates (128 Hz)
N_CSP = 8                        # spatial filters
RANDOM_STATE = 42

# Mapping EXP‑id → PhysioNet run numbers (PDF page 8)
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
# MEMORY MANAGEMENT UTILITIES
###############################################################################

def _check_memory_usage():
    """Check available memory and warn if low."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 2.0:
            print(f"WARNING: Low memory available ({available_gb:.1f} GB). Consider using --max-subjects to reduce dataset size.")
            return False
        elif available_gb < 4.0:
            print(f"CAUTION: Limited memory available ({available_gb:.1f} GB). Processing may be slow.")
        return True
    except ImportError:
        # psutil not available, skip check
        return True

def _cleanup_memory():
    """Force garbage collection and provide memory cleanup."""
    import gc
    gc.collect()
    # Try to hint to OS to free memory
    try:
        import os
        if hasattr(os, 'sync'):
            os.sync()
    except:
        pass

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
    present = [ch for ch in MOTOR_CHANS if ch in raw.ch_names]
    raw.pick(present)
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)

    if int(raw.info["sfreq"]) != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto", verbose=False)
    return raw


def _epochs(raw: mne.io.BaseRaw, exp: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) ready for scikit‑learn."""
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
    """Load subject data with memory cleanup."""
    import gc
    try:
        result = _epochs(_load_raw(subj, RUNS[exp]), exp)
        # Force garbage collection after loading heavy data
        gc.collect()
        return result
    except Exception as e:
        gc.collect()  # Clean up even on error
        raise e

###############################################################################
# PIPELINE (CSP ➔ LDA) – compatible with MNE ≥ 1.3
###############################################################################

class FixedCSP(CSP):
    """CSP wrapper that ensures data is float64 to avoid MNE copy issues."""
    
    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        return super().fit(X, y, **kwargs)
    
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return super().transform(X)
    
    def fit_transform(self, X, y=None, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        return super().fit_transform(X, y, **kwargs)


def _make_csp():
    params = dict(n_components=8, log=True)
    # MNE ≥ 1.4 accepts random_state & cov_est. Add them if present.
    import inspect
    sig = inspect.signature(CSP)
    if "random_state" in sig.parameters:
        params["random_state"] = RANDOM_STATE
    if "cov_est" in sig.parameters:
        params["cov_est"] = "concat"
    return FixedCSP(**params)


def build_pipeline() -> Pipeline:
    return Pipeline([("csp", _make_csp()), ("lda", LDA())])

###############################################################################
# TRAIN / EVALUATION UTILITIES
###############################################################################

def _stack(exp: int, subjects: List[int], batch_size: int = 10):
    """Load and stack data with memory-efficient batching for large datasets."""
    Xs, ys, groups = [], [], []
    loaded_count = 0
    
    for i, s in enumerate(subjects):
        try:
            X, y = load_subject(exp, s)
            Xs.append(X); ys.append(y); groups.extend([s]*len(y))
            loaded_count += 1
            
            # Progress indicator for large datasets
            if len(subjects) > 20 and (i + 1) % 10 == 0:
                print(f"Loaded {i + 1}/{len(subjects)} subjects...")
                
        except Exception as err:
            print(f"[SKIP] S{s:03d}: {err}")
            continue
            
        # Memory management: if we have too much data, process in batches
        if len(subjects) > 50 and len(Xs) >= batch_size:
            # Concatenate current batch
            X_batch = np.concatenate(Xs).astype(np.float64)
            y_batch = np.concatenate(ys)
            g_batch = np.array(groups)
            
            # Clear lists to free memory
            Xs.clear(); ys.clear(); groups.clear()
            
            # Yield batch for processing
            yield X_batch, y_batch, g_batch
            
    if not Xs and loaded_count == 0:
        raise RuntimeError("No usable data loaded.")
    
    # Return final batch if any data remains
    if Xs:
        X_final = np.concatenate(Xs).astype(np.float64)
        y_final = np.concatenate(ys)
        g_final = np.array(groups)
        
        if len(subjects) > 50:
            yield X_final, y_final, g_final
        else:
            # For small datasets, return normally (backward compatibility)
            return X_final, y_final, g_final


def train_exp(exp: int, subjects: List[int]):
    print(f"\n=== EXP {exp} – {len(subjects)} subjects ===")
    
    # Handle large datasets with batch processing
    if len(subjects) > 50:
        print("Using memory-efficient batch processing for large dataset...")
        return _train_exp_batched(exp, subjects)
    
    # Original method for smaller datasets
    X, y, g = _stack(exp, subjects)
    cv = GroupKFold(n_splits=5)
    # Reduce parallel jobs for memory efficiency
    n_jobs = 2 if len(subjects) > 30 else 4
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, groups=g, n_jobs=n_jobs)
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    model = build_pipeline().fit(X, y)
    joblib.dump(model, MODEL_DIR / f"exp{exp}.pkl")
    return scores.mean()


def _train_exp_batched(exp: int, subjects: List[int]):
    """Memory-efficient training for large datasets using incremental learning."""
    import gc
    
    print("Phase 1: Collecting all data in batches for CV evaluation...")
    
    # First pass: collect data for cross-validation in manageable chunks
    all_X, all_y, all_g = [], [], []
    total_trials = 0
    
    for batch_data in _stack(exp, subjects):
        X_batch, y_batch, g_batch = batch_data
        all_X.append(X_batch)
        all_y.append(y_batch) 
        all_g.append(g_batch)
        total_trials += len(X_batch)
        
        # Force garbage collection to free memory
        gc.collect()
    
    print(f"Collected {len(all_X)} batches with total {total_trials} trials")
    
    # Evaluate using a subset for CV (to avoid memory issues)
    print("Phase 2: Cross-validation on subset...")
    
    # Use only first few batches for CV to estimate performance
    n_cv_batches = min(3, len(all_X))
    X_cv = np.concatenate(all_X[:n_cv_batches])
    y_cv = np.concatenate(all_y[:n_cv_batches])
    g_cv = np.concatenate(all_g[:n_cv_batches])
    
    cv = GroupKFold(n_splits=min(5, len(np.unique(g_cv))))
    pipeline_cv = build_pipeline()
    scores = cross_val_score(pipeline_cv, X_cv, y_cv, cv=cv, groups=g_cv, n_jobs=2)
    print(f"CV accuracy (subset): {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Clean up CV data
    del X_cv, y_cv, g_cv, pipeline_cv
    gc.collect()
    
    print("Phase 3: Training final model on all data...")
    
    # Train final model incrementally if possible, otherwise use all data
    if hasattr(build_pipeline().named_steps['lda'], 'partial_fit'):
        # Incremental training (not available for standard LDA, but keeping for extensibility)
        model = build_pipeline()
        for i, (X_batch, y_batch, g_batch) in enumerate(zip(all_X, all_y, all_g)):
            print(f"Training on batch {i+1}/{len(all_X)}...")
            if i == 0:
                model.fit(X_batch, y_batch)
            # Note: LDA doesn't support partial_fit, so we'll use full training
            gc.collect()
    else:
        # Full training on concatenated data (risky for memory but necessary)
        print("Concatenating all data for final training...")
        X_all = np.concatenate(all_X)
        y_all = np.concatenate(all_y)
        
        print(f"Final training data shape: {X_all.shape}")
        model = build_pipeline().fit(X_all, y_all)
        
        # Clean up
        del X_all, y_all
        gc.collect()
    
    # Save model
    joblib.dump(model, MODEL_DIR / f"exp{exp}.pkl")
    print(f"Model saved to {MODEL_DIR / f'exp{exp}.pkl'}")
    
    # Clean up remaining data
    del all_X, all_y, all_g
    gc.collect()
    
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

    p = argparse.ArgumentParser(
        description="TPV baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("experiment", type=int, nargs="?", help="Experiment id 0‑5")
    p.add_argument("subject", type=int, nargs="?", help="Subject id 1‑109 for predict/stream")
    p.add_argument("mode", choices=["train", "predict", "stream"], nargs="?", help="Mode")

    p.add_argument("--full", action="store_true", help="Use all 109 subjects instead of 10")
    p.add_argument("--data-dir", help="PhysioNet root to avoid downloads")
    p.add_argument("--delay", type=float, default=2.0, help="Inter‑epoch delay for stream")
    p.add_argument("--batch-size", type=int, default=10, help="Batch size for memory-efficient processing")
    p.add_argument("--max-subjects", type=int, help="Limit number of subjects (useful for testing)")
    p.add_argument("--gc-freq", type=int, default=5, help="Garbage collection frequency (every N subjects)")

    args = p.parse_args()
    DATA_DIR = args.data_dir

    # Check available memory for large datasets
    if args.full:
        print("Full dataset mode enabled (109 subjects)")
        _check_memory_usage()
        print("Using memory-efficient batch processing...")

    subjects = _default_subjects(args.full)
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
        print(f"Limited to {len(subjects)} subjects for testing")

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
        scores = cross_val_score(build_pipeline(), X, y, cv=cv, n_jobs=-2)
        print(f"Single‑subject CV: {scores.mean():.3f} ± {scores.std():.3f}")
        return

    if args.mode == "predict":
        predict(args.experiment, args.subject)
        return

    if args.mode == "stream":
        stream(args.experiment, args.subject, args.delay)
        return


if __name__ == "__main__":
    main()
