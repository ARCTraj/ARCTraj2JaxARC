"""Utility functions for ARCTraj → JaxARC conversion."""

import numpy as np


MAX_GRID_SIZE = 30
BACKGROUND_COLOR = -1


def pad_grid(grid: np.ndarray, max_size: int = MAX_GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Pad a variable-size grid to (max_size, max_size) with a validity mask.

    Args:
        grid: (H, W) int array with color values 0-9.
        max_size: Target size (default 30 for JaxARC).

    Returns:
        padded_grid: (max_size, max_size) int32 array, padded with BACKGROUND_COLOR.
        mask: (max_size, max_size) bool array, True for valid cells.
    """
    h, w = grid.shape
    padded = np.full((max_size, max_size), BACKGROUND_COLOR, dtype=np.int32)
    mask = np.zeros((max_size, max_size), dtype=bool)
    padded[:h, :w] = grid
    mask[:h, :w] = True
    return padded, mask


def pad_selection(selection: np.ndarray, max_size: int = MAX_GRID_SIZE) -> np.ndarray:
    """Pad a boolean selection mask to (max_size, max_size).

    Args:
        selection: (H, W) bool array.
        max_size: Target size.

    Returns:
        padded: (max_size, max_size) bool array.
    """
    h, w = selection.shape
    padded = np.zeros((max_size, max_size), dtype=bool)
    padded[:h, :w] = selection
    return padded


def compute_similarity(grid: np.ndarray, target: np.ndarray,
                       grid_mask: np.ndarray = None) -> float:
    """Compute pixel-wise similarity between grid and target.

    Args:
        grid: (H, W) int array.
        target: (H, W) int array.
        grid_mask: Optional (H, W) bool mask for valid cells.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    if grid.shape != target.shape:
        return 0.0

    if grid_mask is not None:
        valid = grid_mask
        total = valid.sum()
        if total == 0:
            return 1.0
        matching = ((grid == target) & valid).sum()
        return float(matching / total)
    else:
        total = grid.size
        matching = (grid == target).sum()
        return float(matching / total)
