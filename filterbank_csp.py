import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from custom_csp import CustomCSP
import mne
from mne.decoding import CSP
import warnings
import logging
import joblib

# Désactivation des warnings Python
warnings.filterwarnings('ignore')
# Désactivation du logger MNE
mne.set_log_level('ERROR')
logging.getLogger('mne').setLevel(logging.ERROR)
logging.getLogger('mne').propagate = False
# Désactivation du root logger (pour joblib/sklearn)
logging.getLogger().setLevel(logging.ERROR)
# Désactivation des logs joblib
try:
    joblib_logger = logging.getLogger('joblib')
    joblib_logger.setLevel(logging.ERROR)
    joblib_logger.propagate = False
except Exception:
    pass

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
            csp = CSP(n_components=self.n_csp)
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
        n_trials, n_channels, n_times = X.shape
        # On vectorise le filtrage sur tous les essais et canaux d'un coup
        X_reshaped = X.reshape(-1, n_times)  # (n_trials * n_channels, n_times)
        X_filt = mne.filter.filter_data(X_reshaped, self.sfreq, fmin, fmax, verbose='ERROR')
        X_filt = X_filt.reshape(n_trials, n_channels, n_times)
        return X_filt
