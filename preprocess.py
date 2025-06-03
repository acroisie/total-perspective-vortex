#!/usr/bin/env python3
"""
Total Perspective Vortex - Data Preprocessing and Visualization
This script implements the preprocessing, parsing and formatting phase as required by the subject.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from pathlib import Path
import argparse

# Same constants as main script
FMIN, FMAX = 7.0, 30.0
TMIN, TMAX = 0.0, 4.0
EXPERIMENT_RUNS = {
    0: [3, 7, 11],  # L/R execution
    1: [4, 8, 12],  # L/R imagery
    2: [5, 9, 13],  # Hand/Feet execution
    3: [6, 10, 14],  # Hand/Feet imagery
    4: [3, 4, 7, 8, 11, 12],  # L/R mixed (exec + imag)
    5: [5, 6, 9, 10, 13, 14],  # H/F mixed (exec + imag)
}

def load_raw_data(exp: int, subj: int, data_path=None):
    """Load raw data for visualization"""
    runs = EXPERIMENT_RUNS[exp]
    
    if data_path is not None:
        # Load from local files
        subj_dir = data_path / f"S{subj:03d}"
        files = []
        for run in runs:
            edf_file = subj_dir / f"S{subj:03d}R{run:02d}.edf"
            if edf_file.exists():
                files.append(str(edf_file))
        raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    else:
        # Load from PhysioNet
        files = eegbci.load_data(subj, runs, verbose=False)
        raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    return raw

def visualize_raw_data(raw, title="Raw EEG Data"):
    """Visualize raw EEG data"""
    print(f"\\n=== {title} ===")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print(f"Channels: {raw.ch_names[:10]}..." if len(raw.ch_names) > 10 else f"Channels: {raw.ch_names}")
    
    # Plot PSD for all channels
    plot_psd_all_channels(raw, title=f"{title} - Power Spectral Density")
    
    # Plot mean PSD
    plot_mean_psd(raw, title=f"{title} - Mean PSD")

def plot_psd_all_channels(raw, fmin=0, fmax=80, n_fft=2048, title="PSD"):
    """Plot PSD for all channels"""
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, method="welch", n_fft=n_fft, verbose=False)
    psds = psd.get_data()
    freqs = psd.freqs
    psd_db = 10 * np.log10(psds)

    plt.figure(figsize=(12, 6))
    for ch in range(psd_db.shape[0]):
        plt.plot(freqs, psd_db[ch], color="black", alpha=0.3, linewidth=0.5)
    
    # Mark frequency bands of interest
    plt.axvline(FMIN, color='red', linestyle='--', label=f'Filter min: {FMIN} Hz')
    plt.axvline(FMAX, color='red', linestyle='--', label=f'Filter max: {FMAX} Hz')
    plt.axvspan(8, 13, alpha=0.2, color='blue', label='Alpha band (8-13 Hz)')
    plt.axvspan(13, 30, alpha=0.2, color='green', label='Beta band (13-30 Hz)')
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_mean_psd(raw, fmin=0, fmax=80, n_fft=2048, title="Mean PSD"):
    """Plot mean PSD across channels"""
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, method="welch", n_fft=n_fft, verbose=False)
    psds = psd.get_data()
    freqs = psd.freqs
    psd_db = 10 * np.log10(psds)

    mean = psd_db.mean(axis=0)
    std = psd_db.std(axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, mean, color="black", linewidth=2, label="Mean PSD")
    plt.fill_between(freqs, mean - std, mean + std, color="gray", alpha=0.4, label="±1 std")
    
    # Mark frequency bands
    plt.axvline(FMIN, color='red', linestyle='--', label=f'Filter min: {FMIN} Hz')
    plt.axvline(FMAX, color='red', linestyle='--', label=f'Filter max: {FMAX} Hz')
    plt.axvspan(8, 13, alpha=0.2, color='blue', label='Alpha band (8-13 Hz)')
    plt.axvspan(13, 30, alpha=0.2, color='green', label='Beta band (13-30 Hz)')
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_filtered_data(raw, title="Filtered EEG Data"):
    """Apply filtering and visualize"""
    raw_filtered = raw.copy()
    raw_filtered.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
    
    print(f"\\n=== {title} ===")
    print(f"Applied bandpass filter: {FMIN}-{FMAX} Hz")
    
    # Plot filtered PSD
    plot_psd_all_channels(raw_filtered, title=f"{title} - Power Spectral Density")
    plot_mean_psd(raw_filtered, title=f"{title} - Mean PSD")
    
    return raw_filtered

def analyze_events(raw):
    """Analyze events in the data"""
    events, event_dict = mne.events_from_annotations(raw)
    
    print(f"\\n=== Event Analysis ===")
    print(f"Total events: {len(events)}")
    print(f"Event types: {event_dict}")
    print(f"Event counts:")
    for event_name, event_id in event_dict.items():
        count = np.sum(events[:, 2] == event_id)
        print(f"  {event_name} (ID {event_id}): {count} events")
    
    return events, event_dict

def extract_epochs_features(raw, events, event_dict):
    """Extract epoch features for analysis"""
    # Determine event_id based on available events
    if 'T1' in event_dict and 'T2' in event_dict:
        if 'T0' in event_dict:
            # Standard motor imagery setup
            event_id = {"T1": event_dict['T1'], "T2": event_dict['T2']}
        else:
            event_id = {"T1": event_dict['T1'], "T2": event_dict['T2']}
    else:
        print("Warning: Expected T1/T2 events not found")
        return None, None
    
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
    
    print(f"\\n=== Epoch Extraction ===")
    print(f"Epochs shape: {epochs.get_data().shape}")
    print(f"Time window: {TMIN} to {TMAX} seconds")
    print(f"Epochs per class:")
    for event_name, event_id_val in event_id.items():
        count = np.sum(epochs.events[:, 2] == event_id_val)
        print(f"  {event_name}: {count} epochs")
    
    # Extract features (mean power by frequency band and channel)
    X = epochs.get_data().astype(np.float64)
    y = epochs.events[:, 2]
    
    # Compute power spectral density for each epoch
    print(f"\\nExtracting features...")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features per epoch: {X.shape[1] * X.shape[2]} (channels × time points)")
    
    return X, y

def main():
    parser = argparse.ArgumentParser(
        description="EEG Data Preprocessing and Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess.py 4 14                    # Visualize experiment 4, subject 14
  python preprocess.py 4 14 --data files      # Use local files
  python preprocess.py 0 1 --save-plots       # Save plots to files
        """
    )
    
    parser.add_argument("experiment", type=int, help="Experiment number (0-5)")
    parser.add_argument("subject", type=int, help="Subject number (1-109)")
    parser.add_argument("--data", type=str, help="Path to local data directory")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    
    args = parser.parse_args()
    
    # Set up data path
    data_path = Path(args.data) if args.data else None
    if data_path and not data_path.exists():
        print(f"Error: Data path {data_path} does not exist")
        return
    
    print(f"=== EEG Data Analysis ===")
    print(f"Experiment: {args.experiment}")
    print(f"Subject: {args.subject}")
    print(f"Data source: {'Local files' if data_path else 'PhysioNet'}")
    
    try:
        # Load raw data
        raw = load_raw_data(args.experiment, args.subject, data_path)
        
        # Visualize raw data
        visualize_raw_data(raw, f"Raw Data - Exp {args.experiment}, Subject {args.subject}")
        
        # Apply filtering and visualize
        raw_filtered = visualize_filtered_data(raw, f"Filtered Data - Exp {args.experiment}, Subject {args.subject}")
        
        # Analyze events
        events, event_dict = analyze_events(raw_filtered)
        
        # Extract epochs and features
        X, y = extract_epochs_features(raw_filtered, events, event_dict)
        
        if X is not None:
            print(f"\\n=== Feature Summary ===")
            print(f"Total samples: {len(X)}")
            print(f"Classes: {np.unique(y)}")
            print(f"Class distribution: {np.bincount(y)}")
            print(f"Feature dimensionality: {X.shape[1]} channels × {X.shape[2]} time points = {X.shape[1] * X.shape[2]} features")
            
            # This demonstrates the high dimensionality that will be reduced by CSP
            print(f"\\nThis high-dimensional data ({X.shape[1] * X.shape[2]} features) will be reduced")
            print(f"by the CSP algorithm to extract the most discriminative spatial patterns.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
