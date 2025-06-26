import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CustomCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def _compute_covariance(self, X):
        n_trials, n_channels, n_times = X.shape
        cov = np.zeros((n_channels, n_channels))
        for trial in range(n_trials):
            trial_data = X[trial]
            if self.norm_trace:
                trial_cov = np.cov(trial_data)
                trial_cov = trial_cov / np.trace(trial_cov)
            else:
                trial_cov = np.cov(trial_data)
            cov += trial_cov
        cov /= n_trials
        if self.reg is not None:
            cov += self.reg * np.eye(n_channels)
        return cov

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(f"X should be 3D array, got {X.ndim}D")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {len(classes)}"
            )
        self.classes_ = classes
        X_class_0 = X[y == classes[0]]
        X_class_1 = X[y == classes[1]]
        if len(X_class_0) == 0 or len(X_class_1) == 0:
            raise ValueError("Both classes must have at least one sample")
        C1 = self._compute_covariance(X_class_0)
        C2 = self._compute_covariance(X_class_1)
        self.cov_1_ = C1
        self.cov_2_ = C2
        C_composite = C1 + C2
        eigenvals_comp, eigenvecs_comp = eigh(C_composite)
        if np.any(eigenvals_comp <= 0):
            eigenvals_comp = np.maximum(eigenvals_comp, 1e-10)
        W = eigenvecs_comp @ np.diag(eigenvals_comp**-0.5) @ eigenvecs_comp.T
        C1_white = W @ C1 @ W.T
        # C2_white = W @ C2 @ W.T
        eigenvals, eigenvecs = eigh(C1_white)
        sorted_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sorted_indices]
        eigenvecs = eigenvecs[:, sorted_indices]
        self.filters_ = W.T @ eigenvecs
        n_comp_half = self.n_components // 2
        if self.n_components % 2 == 1:
            indices = np.concatenate(
                [
                    np.arange(n_comp_half),
                    np.arange(-n_comp_half - 1, 0),
                ]
            )
        else:
            indices = np.concatenate(
                [
                    np.arange(n_comp_half),
                    np.arange(-n_comp_half, 0),
                ]
            )
        self.filters_ = self.filters_[:, indices]
        self.eigenvalues_ = eigenvals[indices]
        self.patterns_ = np.linalg.pinv(self.filters_.T)
        return self

    def transform(self, X):
        X = np.asarray(X)
        if not hasattr(self, "filters_"):
            raise ValueError("CSP must be fitted before transform")
        if X.ndim != 3:
            raise ValueError(f"X should be 3D array, got {X.ndim}D")
        n_trials, n_channels, n_times = X.shape
        if n_channels != self.filters_.shape[0]:
            raise ValueError(
                f"Number of channels mismatch: expected {self.filters_.shape[0]}, got {n_channels}"
            )
        X_filtered = np.zeros((n_trials, self.n_components, n_times))
        for trial in range(n_trials):
            X_filtered[trial] = self.filters_.T @ X[trial]
        features = np.zeros((n_trials, self.n_components))
        for trial in range(n_trials):
            for comp in range(self.n_components):
                variance = np.var(X_filtered[trial, comp, :])
                if self.log:
                    features[trial, comp] = np.log(variance + 1e-10)
                else:
                    features[trial, comp] = variance
        return features

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_spatial_patterns(self):
        if not hasattr(self, "patterns_"):
            raise ValueError("CSP must be fitted before getting patterns")
        return self.patterns_

    def get_spatial_filters(self):
        if not hasattr(self, "filters_"):
            raise ValueError("CSP must be fitted before getting filters")
        return self.filters_


def test_custom_csp():
    print("Testing Custom CSP Implementation...")
    np.random.seed(42)
    n_channels = 10
    n_times = 100
    n_trials_per_class = 20
    X_class_0 = np.random.randn(n_trials_per_class, n_channels, n_times)
    X_class_0[:, :3, :] *= 2
    X_class_1 = np.random.randn(n_trials_per_class, n_channels, n_times)
    X_class_1[:, -3:, :] *= 2
    X = np.concatenate([X_class_0, X_class_1], axis=0)
    y = np.concatenate(
        [np.zeros(n_trials_per_class), np.ones(n_trials_per_class)]
    )
    csp = CustomCSP(n_components=4, log=True)
    X_csp = csp.fit_transform(X, y)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_csp.shape}")
    print(f"Spatial filters shape: {csp.filters_.shape}")
    print(f"Spatial patterns shape: {csp.patterns_.shape}")
    print(f"Eigenvalues: {csp.eigenvalues_}")
    class_0_features = X_csp[:n_trials_per_class]
    class_1_features = X_csp[n_trials_per_class:]
    mean_0 = np.mean(class_0_features, axis=0)
    mean_1 = np.mean(class_1_features, axis=0)
    print(f"Class 0 mean features: {mean_0}")
    print(f"Class 1 mean features: {mean_1}")
    print(f"Feature difference: {np.abs(mean_1 - mean_0)}")
    print("Custom CSP test completed successfully!")


if __name__ == "__main__":
    test_custom_csp()
