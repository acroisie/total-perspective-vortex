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
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

MOTOR_CHANS = [
    "Cz",
    "C3",
    "C4",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "FC3",
    "FC4",
    "CP3",
    "CP4",
    "C1",
    "C2",
    "CPz",
    "FCz",
    "P3",
    "P4",
    "Pz",
    "F3",
    "F4",
    "Fz",
    "Oz",
]

###############################################################################
# CONSTANTS – optimized for better performance
###############################################################################

# Optimized frequency bands for motor imagery
FMIN, FMAX = 7.0, 35.0  # Wider μ/β band for better coverage
EPOCH_TMIN, EPOCH_TMAX = (
    0.5,
    3.5,
)  # 3‑second window, skip first 0.5s for motor preparation
TARGET_FS = 128  # unify sampling rates (128 Hz)
N_CSP = 8  # optimal number of spatial filters
RANDOM_STATE = 42

# Optimized frequency bands for multi-band analysis
FREQ_BANDS = {
    "mu": (7, 13),  # μ rhythm (wider)
    "beta": (13, 35),  # β rhythm (extended)
    "all": (7, 35),  # combined (full range)
}

# Mapping EXP‑id → PhysioNet run numbers (PDF page 8)
RUNS = {
    0: [3, 7, 11],  # L/R execution
    1: [4, 8, 12],  # L/R imagery
    2: [5, 9, 13],  # H/F execution
    3: [6, 10, 14],  # H/F imagery
    4: [3, 4, 7, 8, 11, 12],  # L/R mixed
    5: [5, 6, 9, 10, 13, 14],  # H/F mixed
}

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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
            print(
                f"WARNING: Low memory available ({available_gb:.1f} GB). Consider using --max-subjects to reduce dataset size."
            )
            return False
        elif available_gb < 4.0:
            print(
                f"CAUTION: Limited memory available ({available_gb:.1f} GB). Processing may be slow."
            )
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

        if hasattr(os, "sync"):
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
    """Concatenate runs, band‑pass & resample with enhanced filtering."""
    fnames = _local_files(subj, runs) or eegbci.load_data(
        subj, runs, path=DATA_DIR, verbose=False
    )
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw = mne.concatenate_raws(raws)

    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    present = [ch for ch in MOTOR_CHANS if ch in raw.ch_names]
    raw.pick(present)

    # Enhanced filtering: apply notch filter for power line noise
    raw.notch_filter([50, 60], picks="eeg", fir_design="firwin", verbose=False)
    raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)

    if int(raw.info["sfreq"]) != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto", verbose=False)
    return raw


def _epochs(raw: mne.io.BaseRaw, exp: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) ready for scikit‑learn with advanced preprocessing."""
    events, _ = mne.events_from_annotations(raw, verbose=False)
    event_id = (
        {"left": 1, "right": 2} if exp in (0, 1, 4) else {"hands": 2, "feet": 3}
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        EPOCH_TMIN,
        EPOCH_TMAX,
        baseline=None,
        detrend=1,
        preload=True,
        picks="eeg",
        verbose=False,
    )

    # Enhanced preprocessing: apply baseline correction
    epochs.apply_baseline(baseline=(EPOCH_TMIN, EPOCH_TMIN + 0.2))

    # Apply automatic artifact rejection
    reject_criteria = dict(eeg=75e-6)  # 75 μV threshold
    epochs.drop_bad(reject=reject_criteria)

    X = epochs.get_data().astype(np.float64)  # (n_epochs, n_ch, n_times)
    y = epochs.events[:, 2] % 2  # 0 / 1

    if len(np.unique(y)) != 2:
        raise RuntimeError("Only one class present – skip subject.")

    # Advanced temporal smoothing with Gaussian filter
    from scipy.ndimage import gaussian_filter1d

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Apply Gaussian smoothing (sigma=1.0)
            X[i, j, :] = gaussian_filter1d(X[i, j, :], sigma=1.0)

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
# ADVANCED FEATURE EXTRACTION
###############################################################################


class AdvancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Advanced temporal-spectral feature extraction for EEG motor imagery."""

    def __init__(self, fs=128, freq_bands=None):
        self.fs = fs
        self.freq_bands = freq_bands or FREQ_BANDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extract comprehensive features from EEG data."""
        n_trials, n_channels, n_times = X.shape
        features = []

        for trial in range(n_trials):
            trial_features = []
            trial_data = X[trial, :, :]

            # 1. Multi-band power features
            for band_name, (low, high) in self.freq_bands.items():
                for ch in range(n_channels):
                    signal = trial_data[ch, :]
                    freqs, psd = welch(
                        signal, fs=self.fs, nperseg=min(64, n_times // 4)
                    )

                    # Extract power in frequency band
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.mean(psd[band_mask])
                        trial_features.append(np.log(band_power + 1e-12))

            # 2. Channel variance features (log-transformed)
            var_features = np.var(trial_data, axis=1)
            trial_features.extend(np.log(var_features + 1e-12))

            # 3. Hemispheric asymmetry (C3/C4-like features)
            if n_channels >= 4:
                left_power = np.mean(
                    np.var(trial_data[: n_channels // 2, :], axis=1)
                )
                right_power = np.mean(
                    np.var(trial_data[n_channels // 2 :, :], axis=1)
                )
                asymmetry = (left_power - right_power) / (
                    left_power + right_power + 1e-12
                )
                trial_features.append(asymmetry)

            # 4. Temporal dynamics: early vs late activity
            mid_point = n_times // 2
            early_power = np.mean(np.var(trial_data[:, :mid_point], axis=1))
            late_power = np.mean(np.var(trial_data[:, mid_point:], axis=1))
            temporal_ratio = early_power / (late_power + 1e-12)
            trial_features.append(np.log(temporal_ratio + 1e-12))

            # 5. Cross-correlation features between key channels
            if n_channels >= 3:
                # Approximate motor cortex channels (C3, Cz, C4)
                c1, c2, c3 = 0, n_channels // 2, n_channels - 1
                corr_c1_c2 = np.corrcoef(trial_data[c1, :], trial_data[c2, :])[
                    0, 1
                ]
                corr_c2_c3 = np.corrcoef(trial_data[c2, :], trial_data[c3, :])[
                    0, 1
                ]
                trial_features.extend([corr_c1_c2, corr_c2_c3])

            features.append(trial_features)

        return np.array(features)


class OptimizedCSP(CSP):
    """Optimized CSP with better regularization and feature extraction."""

    def __init__(self, n_components=8, reg=0.01, log=True, **kwargs):
        # Lower regularization for better spatial filtering
        super().__init__(n_components=n_components, reg=reg, log=log, **kwargs)

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        return super().fit(X, y, **kwargs)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Get CSP features
        csp_features = super().transform(X)

        # Add differential power features for better discrimination
        n_trials, n_channels, n_times = X.shape
        power_features = np.zeros((n_trials, n_channels))

        # Calculate band power for each channel and trial
        for trial in range(n_trials):
            for ch in range(n_channels):
                signal = X[trial, ch, :]
                # Simple power calculation (variance-based)
                power_features[trial, ch] = np.var(signal)

        # Log transform and normalize
        power_features = np.log(power_features + 1e-12)

        # Combine CSP features with power features
        enhanced_features = np.hstack([csp_features, power_features])
        return enhanced_features

    def fit_transform(self, X, y=None, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        return self.fit(X, y, **kwargs).transform(X)


###############################################################################
# PIPELINE (CSP ➔ LDA) – compatible with MNE ≥ 1.3
###############################################################################


class EnhancedCSP(CSP):
    """Enhanced CSP with regularization and better feature extraction."""

    def __init__(self, n_components=8, reg=0.05, log=True, **kwargs):
        # Add regularization by default
        super().__init__(n_components=n_components, reg=reg, log=log, **kwargs)

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        return super().fit(X, y, **kwargs)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Get basic CSP features
        csp_features = super().transform(X)

        # Add variance features for raw channels (complementary to CSP)
        n_trials, n_channels, n_times = X.shape
        var_features = np.zeros((n_trials, n_channels))

        for trial in range(n_trials):
            var_features[trial] = np.var(X[trial], axis=1)

        # Log transform variance features
        var_features = np.log(var_features + 1e-10)

        # Combine CSP and variance features
        enhanced_features = np.hstack([csp_features, var_features])
        return enhanced_features

    def fit_transform(self, X, y=None, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        return self.fit(X, y, **kwargs).transform(X)


def _make_csp():
    params = dict(n_components=N_CSP, log=True, reg=0.01)
    # MNE ≥ 1.4 accepts random_state & cov_est. Add them if present.
    import inspect

    sig = inspect.signature(CSP)
    if "random_state" in sig.parameters:
        params["random_state"] = RANDOM_STATE
    if "cov_est" in sig.parameters:
        params["cov_est"] = "concat"
    return OptimizedCSP(**params)


def build_pipeline() -> Pipeline:
    """Build optimized pipeline with advanced feature extraction."""
    return Pipeline(
        [
            ("csp", _make_csp()),
            ("scaler", StandardScaler()),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ]
    )


def build_advanced_pipeline() -> Pipeline:
    """Build advanced pipeline with multiple feature extractors."""
    from sklearn.pipeline import FeatureUnion

    feature_union = FeatureUnion(
        [
            ("csp", _make_csp()),
            (
                "advanced",
                AdvancedFeatureExtractor(fs=TARGET_FS, freq_bands=FREQ_BANDS),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", feature_union),
            ("scaler", StandardScaler()),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ]
    )


def build_ensemble_pipeline() -> VotingClassifier:
    """Build optimized ensemble pipeline for maximum performance."""

    # Pipeline 1: Optimized CSP + LDA
    pipe1 = Pipeline(
        [
            ("csp", _make_csp()),
            ("scaler", StandardScaler()),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ]
    )

    # Pipeline 2: CSP + SVM with RBF kernel
    pipe2 = Pipeline(
        [
            ("csp", _make_csp()),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True)),
        ]
    )

    # Pipeline 3: CSP + SVM with polynomial kernel
    pipe3 = Pipeline(
        [
            ("csp", _make_csp()),
            ("scaler", StandardScaler()),
            (
                "svm_poly",
                SVC(
                    kernel="poly",
                    degree=2,
                    C=1.0,
                    gamma="scale",
                    probability=True,
                ),
            ),
        ]
    )

    # Create voting classifier with optimized weights
    ensemble = VotingClassifier(
        estimators=[
            ("csp_lda", pipe1),
            ("csp_svm_rbf", pipe2),
            ("csp_svm_poly", pipe3),
        ],
        voting="soft",
        weights=[3, 2, 1],  # Prioritize LDA, then RBF SVM
    )

    return ensemble


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
            Xs.append(X)
            ys.append(y)
            groups.extend([s] * len(y))
            loaded_count += 1

        except Exception:
            continue

        # Memory management: if we have too much data, process in batches
        if len(subjects) > 50 and len(Xs) >= batch_size:
            # Concatenate current batch
            X_batch = np.concatenate(Xs).astype(np.float64)
            y_batch = np.concatenate(ys)
            g_batch = np.array(groups)

            # Clear lists to free memory
            Xs.clear()
            ys.clear()
            groups.clear()

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
    # Handle large datasets with batch processing
    if len(subjects) > 50:
        return _train_exp_batched(exp, subjects)

    # Load data quietly
    X, y, g = _stack(exp, subjects)

    # Test multiple configurations and pick the best
    best_score = 0
    best_model = None
    best_name = ""

    configs = [
        ("CSP+LDA", build_pipeline()),
        ("Advanced", build_advanced_pipeline()),
        (
            "Ensemble",
            (
                build_ensemble_pipeline()
                if len(subjects) <= 30
                else build_advanced_pipeline()
            ),
        ),
    ]

    cv = GroupKFold(n_splits=5)
    n_jobs = 2 if len(subjects) > 30 else 4

    for name, model in configs:
        try:
            scores = cross_val_score(
                model, X, y, cv=cv, groups=g, n_jobs=n_jobs, verbose=0
            )
            score = scores.mean()

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception:
            continue

    if best_model is None:
        best_model = build_pipeline()
        scores = cross_val_score(
            best_model, X, y, cv=cv, groups=g, n_jobs=n_jobs, verbose=0
        )
        best_score = scores.mean()
        best_name = "CSP+LDA"

    # Train final model on all data
    final_model = best_model.fit(X, y)
    joblib.dump(final_model, MODEL_DIR / f"exp{exp}.pkl")
    return best_score, best_name


def _train_exp_batched(exp: int, subjects: List[int]):
    """Memory-efficient training for large datasets."""
    import gc

    # Collect data for cross-validation in manageable chunks
    all_X, all_y, all_g = [], [], []

    for batch_data in _stack(exp, subjects):
        X_batch, y_batch, g_batch = batch_data
        all_X.append(X_batch)
        all_y.append(y_batch)
        all_g.append(g_batch)
        gc.collect()

    # Cross-validation on subset to estimate performance
    n_cv_batches = min(3, len(all_X))
    X_cv = np.concatenate(all_X[:n_cv_batches])
    y_cv = np.concatenate(all_y[:n_cv_batches])
    g_cv = np.concatenate(all_g[:n_cv_batches])

    cv = GroupKFold(n_splits=min(5, len(np.unique(g_cv))))
    pipeline_cv = build_advanced_pipeline()
    scores = cross_val_score(
        pipeline_cv, X_cv, y_cv, cv=cv, groups=g_cv, n_jobs=2
    )

    # Clean up CV data
    del X_cv, y_cv, g_cv, pipeline_cv
    gc.collect()

    # Train final model on all data
    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)

    model = build_advanced_pipeline().fit(X_all, y_all)

    # Clean up and save
    del X_all, y_all, all_X, all_y, all_g
    gc.collect()

    joblib.dump(model, MODEL_DIR / f"exp{exp}.pkl")
    return scores.mean(), "Advanced"


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
# ENHANCED PIPELINE BUILDING
###############################################################################
# CLI ENTRY POINT
###############################################################################


def _default_subjects(full: bool):
    return list(range(1, 110)) if full else list(range(1, 11))


def main():  # noqa: C901 – linear, readable
    global DATA_DIR

    p = argparse.ArgumentParser(
        description="TPV baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("experiment", type=int, nargs="?", help="Experiment id 0‑5")
    p.add_argument(
        "subject",
        type=int,
        nargs="?",
        help="Subject id 1‑109 for predict/stream",
    )
    p.add_argument(
        "mode", choices=["train", "predict", "stream"], nargs="?", help="Mode"
    )

    p.add_argument(
        "--full", action="store_true", help="Use all 109 subjects instead of 10"
    )
    p.add_argument("--data-dir", help="PhysioNet root to avoid downloads")
    p.add_argument(
        "--delay", type=float, default=2.0, help="Inter‑epoch delay for stream"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for memory-efficient processing",
    )
    p.add_argument(
        "--max-subjects",
        type=int,
        help="Limit number of subjects (useful for testing)",
    )
    p.add_argument(
        "--gc-freq",
        type=int,
        default=5,
        help="Garbage collection frequency (every N subjects)",
    )

    args = p.parse_args()
    DATA_DIR = args.data_dir

    # Check available memory for large datasets
    if args.full:
        print("Full dataset mode enabled (109 subjects)")
        _check_memory_usage()
        print("Using memory-efficient batch processing...")

    subjects = _default_subjects(args.full)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
        print(f"Limited to {len(subjects)} subjects for testing")

    # No positional → train all experiments
    if args.experiment is None:
        print("Training all 6 experiments...")
        results = []
        for e in range(6):
            score, model_name = train_exp(e, subjects)
            results.append((e, score, model_name))
            print(f"experiment {e}: accuracy = {score:.4f} ({model_name})")

        mean_acc = np.mean([score for _, score, _ in results])
        print(f"\nMean CV accuracy over 6 experiments: {mean_acc:.4f}")

        if mean_acc >= 0.65:
            print("✓ Target accuracy achieved!")
        else:
            print(f"✗ Need {0.65 - mean_acc:.4f} more accuracy to reach target")
        return

    # Exp given, no mode → train that experiment
    if args.mode is None:
        score, model_name = train_exp(args.experiment, subjects)
        print(
            f"experiment {args.experiment}: accuracy = {score:.4f} ({model_name})"
        )
        return

    # Modes below need a subject id
    if args.subject is None:
        p.error("<subject> required for this mode")

    if args.mode == "train":
        X, y = load_subject(args.experiment, args.subject)
        cv = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=RANDOM_STATE
        )
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
