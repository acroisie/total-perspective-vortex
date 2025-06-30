from pathlib import Path
import numpy as np
from typing import List, Tuple, Callable


def list_subject_dirs(data_path: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in data_path.iterdir()
            if p.is_dir() and p.name.startswith("S") and len(p.name) == 4
        ]
    )


def aggregate_multi_subject_data(
    subject_dirs: List[Path], exp: int, load_data_func: Callable
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    all_X, all_y, valid_subjects, epoch_lengths = [], [], [], []
    for subj_dir in subject_dirs:
        try:
            subj_num = int(subj_dir.name[1:])
            X, y = load_data_func(exp, subj_num)
            if X.shape[0] > 0:
                all_X.append(X)
                all_y.append(y)
                valid_subjects.append(subj_num)
                epoch_lengths.append(X.shape[2])
        except Exception as e:
            print(f"Skip subject {subj_dir.name}: {e}")
    if not all_X:
        raise ValueError(f"No valid data found for experiment {exp}")
    min_len = min(epoch_lengths)
    all_X = [X[:, :, :min_len] for X in all_X]
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    return X_combined, y_combined, valid_subjects
