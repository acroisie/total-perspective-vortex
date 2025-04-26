import numpy as np
import matplotlib.pyplot as plt

from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


def plot_psd_all_channels(raw, fmin=0, fmax=80, n_fft=2048):
    psd = raw.compute_psd(
        fmin=fmin, fmax=fmax, method="welch", n_fft=n_fft, verbose=False
    )
    psds = psd.get_data()
    freqs = psd.freqs
    psd_db = 10 * np.log10(psds)

    plt.figure(figsize=(8, 5))
    for ch in range(psd_db.shape[0]):
        plt.plot(freqs, psd_db[ch], color="black", alpha=0.3, linewidth=0.5)
    plt.axvline(8, linestyle="--", label="8 Hz")
    plt.axvline(13, linestyle="--", label="13 Hz")
    plt.axvline(30, linestyle="--", label="30 Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.title("EEG: PSD per channel (0–80 Hz)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_mean_psd(raw, fmin=0, fmax=80, n_fft=2048):
    psd = raw.compute_psd(
        fmin=fmin, fmax=fmax, method="welch", n_fft=n_fft, verbose=False
    )
    psds = psd.get_data()
    freqs = psd.freqs
    psd_db = 10 * np.log10(psds)

    mean = psd_db.mean(axis=0)
    std = psd_db.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, mean, color="black", label="Mean")
    plt.fill_between(
        freqs, mean - std, mean + std, color="gray", alpha=0.4, label="±1 std"
    )
    plt.axvline(8, linestyle="--")
    plt.axvline(13, linestyle="--")
    plt.axvline(30, linestyle="--")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.title("EEG: Mean PSD ± Std (0–80 Hz)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main():
    subject = 1
    runs = [6, 10, 14]  # left hand, right hand, feet
    fmin, fmax = 7.0, 30.0

    print(f"Loading subject {subject}, runs {runs}...")
    files = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))

    raw.plot(
        duration=5.0, n_channels=30, title="Raw EEG", show=True, block=True
    )

    raw.filter(fmin, fmax, fir_design="firwin")
    raw.plot(
        duration=5.0,
        n_channels=30,
        title=f"Filtered EEG ({fmin}-{fmax} Hz)",
        show=True,
        block=True,
    )

    plot_psd_all_channels(raw, fmin=0, fmax=80)
    plot_mean_psd(raw, fmin=0, fmax=80)

if __name__ == "__main__":
    main()
