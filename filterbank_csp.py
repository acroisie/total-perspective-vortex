import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from custom_csp import CustomCSP
import mne

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank CSP: Applique le CSP sur plusieurs bandes de fréquences et concatène les features.
    """
    def __init__(self, freq_bands=None, n_csp=3, sfreq=160):
        # freq_bands: liste de tuples (fmin, fmax)
        if freq_bands is None:
            # Par défaut, bandes de 7 à 30 Hz par tranches de 4 Hz
            self.freq_bands = [(f, f+4) for f in range(7, 30, 4)]
        else:
            self.freq_bands = freq_bands
        self.n_csp = n_csp
        self.sfreq = sfreq
        self.csp_list = []

    def fit(self, X, y):
        # X: (n_trials, n_channels, n_times)
        self.csp_list = []
        for fmin, fmax in self.freq_bands:
            X_f = self._bandpass_filter(X, fmin, fmax)
            csp = CustomCSP(n_components=self.n_csp)
            csp.fit(X_f, y)
            self.csp_list.append(csp)
        return self

    def transform(self, X):
        features = []
        for idx, (fmin, fmax) in enumerate(self.freq_bands):
            X_f = self._bandpass_filter(X, fmin, fmax)
            csp = self.csp_list[idx]
            features.append(csp.transform(X_f))
        return np.concatenate(features, axis=1)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _bandpass_filter(self, X, fmin, fmax):
        # X: (n_trials, n_channels, n_times)
        X_filt = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_filt[i] = mne.filter.filter_data(X[i], self.sfreq, fmin, fmax, verbose=False)
        return X_filt
