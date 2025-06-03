"""
Custom CSP (Common Spatial Patterns) Implementation
This module implements the CSP algorithm as required by the subject.

The CSP algorithm finds spatial filters that maximize the variance of one class
while minimizing the variance of another class, making it ideal for motor imagery
classification in BCI applications.
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CustomCSP(BaseEstimator, TransformerMixin):
    """
    Custom implementation of Common Spatial Patterns (CSP) algorithm.

    CSP finds spatial filters that maximize the ratio of variances between two classes.
    For EEG motor imagery data, this helps extract discriminative spatial patterns.

    Mathematical formulation:
    - Given covariance matrices C1 and C2 for two classes
    - Find transformation matrix W such that W^T * C1 * W is maximized
    - While W^T * C2 * W is minimized
    - This is solved as a generalized eigenvalue problem

    Parameters
    ----------
    n_components : int, default=4
        Number of CSP components to extract (spatial filters)
    reg : float or None, default=None
        Regularization parameter for covariance matrix estimation
    log : bool, default=True
        Whether to apply log transformation to the features
    norm_trace : bool, default=False
        Whether to normalize by trace of covariance matrices

    Attributes
    ----------
    filters_ : ndarray of shape (n_channels, n_components)
        The spatial filters (columns of W matrix)
    patterns_ : ndarray of shape (n_channels, n_components)
        The spatial patterns (inverse of filters)
    eigenvalues_ : ndarray of shape (n_components,)
        The eigenvalues corresponding to the selected filters
    """

    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def _compute_covariance(self, X):
        """
        Compute covariance matrix with optional regularization.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_times)
            EEG data for one class

        Returns
        -------
        cov : ndarray of shape (n_channels, n_channels)
            Covariance matrix
        """
        n_trials, n_channels, n_times = X.shape

        # Compute covariance for each trial and average
        cov = np.zeros((n_channels, n_channels))

        for trial in range(n_trials):
            trial_data = X[trial]  # shape: (n_channels, n_times)

            if self.norm_trace:
                # Normalize by trace (total power)
                trial_cov = np.cov(trial_data)
                trial_cov = trial_cov / np.trace(trial_cov)
            else:
                trial_cov = np.cov(trial_data)

            cov += trial_cov

        cov /= n_trials

        # Add regularization if specified
        if self.reg is not None:
            cov += self.reg * np.eye(n_channels)

        return cov

    def fit(self, X, y):
        """
        Fit CSP spatial filters.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_times)
            EEG data
        y : ndarray of shape (n_trials,)
            Class labels (should be binary: 0 and 1)

        Returns
        -------
        self : object
            Returns self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(f"X should be 3D array, got {X.ndim}D")

        # Get unique classes
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {len(classes)}"
            )

        self.classes_ = classes

        # Split data by class
        X_class_0 = X[y == classes[0]]
        X_class_1 = X[y == classes[1]]

        if len(X_class_0) == 0 or len(X_class_1) == 0:
            raise ValueError("Both classes must have at least one sample")

        # Compute covariance matrices for each class
        C1 = self._compute_covariance(X_class_0)
        C2 = self._compute_covariance(X_class_1)

        # Store covariance matrices for debugging
        self.cov_1_ = C1
        self.cov_2_ = C2

        # Solve generalized eigenvalue problem: C1 * w = λ * C2 * w
        # This is equivalent to: (C1 + C2)^(-1) * C1 * w = λ * w
        # We compute (C1 + C2)^(-1) * (C1 - C2) for better numerical stability

        # Composite covariance matrix
        C_composite = C1 + C2

        # Compute eigendecomposition of composite matrix for whitening
        eigenvals_comp, eigenvecs_comp = eigh(C_composite)

        # Check for numerical issues
        if np.any(eigenvals_comp <= 0):
            print(
                f"Warning: Found {np.sum(eigenvals_comp <= 0)} non-positive eigenvalues"
            )
            eigenvals_comp = np.maximum(eigenvals_comp, 1e-10)

        # Whitening matrix
        W = eigenvecs_comp @ np.diag(eigenvals_comp**-0.5) @ eigenvecs_comp.T

        # Transform covariance matrices to whitened space
        C1_white = W @ C1 @ W.T
        C2_white = W @ C2 @ W.T

        # Solve eigenvalue problem in whitened space
        # Since C1_white + C2_white = I, we can solve: C1_white * v = λ * v
        eigenvals, eigenvecs = eigh(C1_white)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sorted_indices]
        eigenvecs = eigenvecs[:, sorted_indices]

        # Transform back to original space
        self.filters_ = W.T @ eigenvecs

        # Select n_components filters:
        # Take first and last n_components//2 filters (most discriminative)
        n_comp_half = self.n_components // 2
        if self.n_components % 2 == 1:
            # If odd number, take one more from the end
            indices = np.concatenate(
                [
                    np.arange(n_comp_half),  # First n_comp_half
                    np.arange(-n_comp_half - 1, 0),  # Last n_comp_half+1
                ]
            )
        else:
            indices = np.concatenate(
                [
                    np.arange(n_comp_half),  # First n_comp_half
                    np.arange(-n_comp_half, 0),  # Last n_comp_half
                ]
            )

        self.filters_ = self.filters_[:, indices]
        self.eigenvalues_ = eigenvals[indices]

        # Compute spatial patterns (inverse of filters)
        self.patterns_ = np.linalg.pinv(self.filters_.T)

        return self

    def transform(self, X):
        """
        Transform data using fitted CSP filters.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_times)
            EEG data to transform

        Returns
        -------
        X_csp : ndarray of shape (n_trials, n_components)
            CSP features (log variance of filtered signals)
        """
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

        # Apply spatial filters
        X_filtered = np.zeros((n_trials, self.n_components, n_times))

        for trial in range(n_trials):
            # Apply filters: filtered_signal = W^T * signal
            X_filtered[trial] = self.filters_.T @ X[trial]

        # Compute features: variance of each filtered signal
        features = np.zeros((n_trials, self.n_components))

        for trial in range(n_trials):
            for comp in range(self.n_components):
                # Compute variance of filtered signal
                variance = np.var(X_filtered[trial, comp, :])

                if self.log:
                    # Apply log transformation (common in CSP)
                    features[trial, comp] = np.log(
                        variance + 1e-10
                    )  # Add small constant to avoid log(0)
                else:
                    features[trial, comp] = variance

        return features

    def fit_transform(self, X, y):
        """
        Fit CSP and transform data.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_times)
            EEG data
        y : ndarray of shape (n_trials,)
            Class labels

        Returns
        -------
        X_csp : ndarray of shape (n_trials, n_components)
            CSP features
        """
        return self.fit(X, y).transform(X)

    def get_spatial_patterns(self):
        """
        Get spatial patterns for visualization.

        Returns
        -------
        patterns : ndarray of shape (n_channels, n_components)
            Spatial patterns (topographical maps)
        """
        if not hasattr(self, "patterns_"):
            raise ValueError("CSP must be fitted before getting patterns")

        return self.patterns_

    def get_spatial_filters(self):
        """
        Get spatial filters.

        Returns
        -------
        filters : ndarray of shape (n_channels, n_components)
            Spatial filters
        """
        if not hasattr(self, "filters_"):
            raise ValueError("CSP must be fitted before getting filters")

        return self.filters_


# Test function to verify the implementation
def test_custom_csp():
    """Test the custom CSP implementation with synthetic data."""
    print("Testing Custom CSP Implementation...")

    # Generate synthetic EEG data
    np.random.seed(42)
    n_channels = 10
    n_times = 100
    n_trials_per_class = 20

    # Create synthetic data with different spatial patterns for each class
    # Class 0: stronger activity in first channels
    X_class_0 = np.random.randn(n_trials_per_class, n_channels, n_times)
    X_class_0[:, :3, :] *= 2  # Amplify first 3 channels

    # Class 1: stronger activity in last channels
    X_class_1 = np.random.randn(n_trials_per_class, n_channels, n_times)
    X_class_1[:, -3:, :] *= 2  # Amplify last 3 channels

    # Combine data
    X = np.concatenate([X_class_0, X_class_1], axis=0)
    y = np.concatenate(
        [np.zeros(n_trials_per_class), np.ones(n_trials_per_class)]
    )

    # Test CSP
    csp = CustomCSP(n_components=4, log=True)
    X_csp = csp.fit_transform(X, y)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_csp.shape}")
    print(f"Spatial filters shape: {csp.filters_.shape}")
    print(f"Spatial patterns shape: {csp.patterns_.shape}")
    print(f"Eigenvalues: {csp.eigenvalues_}")

    # Verify separability
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
