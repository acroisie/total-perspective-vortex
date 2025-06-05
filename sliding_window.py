import numpy as np

def segment_sliding_window(X, window_fraction=0.5, step_fraction=0.25):
    """
    Découpe les epochs en segments avec fenêtre glissante.
    
    Parameters
    ----------
    X : ndarray of shape (n_epochs, n_channels, n_times)
        Données EEG
    window_fraction : float
        Fraction de la durée totale pour la fenêtre (ex: 0.5 = 50%)
    step_fraction : float
        Fraction du pas (ex: 0.25 = 25% de pas)
        
    Returns
    -------
    segments : ndarray of shape (n_segments, n_channels, window_length)
        Segments issus de la fenêtre glissante
    segment_labels : ndarray of shape (n_segments,)
        Liste indiquant l'epoch source de chaque segment (pour traçabilité)
    """
    n_epochs, n_channels, n_times = X.shape
    window_length = int(n_times * window_fraction)
    step_size = int(n_times * step_fraction)
    
    segments = []
    segment_labels = []
    
    for epoch_idx in range(n_epochs):
        for start in range(0, n_times - window_length + 1, step_size):
            segment = X[epoch_idx, :, start:start + window_length]
            segments.append(segment)
            segment_labels.append(epoch_idx)
    
    return np.array(segments), np.array(segment_labels)

def apply_sliding_to_epochs(X, y, window_fraction=0.5, step_fraction=0.25):
    """
    Applique une fenêtre glissante aux epochs et propage les labels.
    
    Parameters
    ----------
    X : ndarray of shape (n_epochs, n_channels, n_times)
        Données EEG
    y : ndarray of shape (n_epochs,)
        Labels des epochs
    window_fraction, step_fraction : float
        Paramètres de la fenêtre glissante
        
    Returns
    -------
    X_segments : ndarray
        Segments issus de la fenêtre glissante
    y_segments : ndarray
        Labels associés à chaque segment
    """
    segments, segment_sources = segment_sliding_window(
        X, window_fraction, step_fraction
    )
    
    # Associer le label de l'epoch source à chaque segment
    y_segments = np.array([y[source_idx] for source_idx in segment_sources])
    
    return segments, y_segments
