"""Input check for the per-tree ``crown_base_height`` column (m).

Every stage that reads crown base height runs this check on the frame it
is handed, so a stage called on its own rejects the same rows the full
chain would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_crown_base_height(trees: pd.DataFrame) -> None:
    """Raise unless every crown base height is finite and in [0, height].

    Raises
    ------
    ValueError
        If any ``crown_base_height`` is missing, non-finite, negative, or
        above the tree's ``height``.
    """
    height = trees["height"].to_numpy(dtype=np.float64, na_value=np.nan)
    crown_base = trees["crown_base_height"].to_numpy(dtype=np.float64, na_value=np.nan)
    invalid = ~np.isfinite(crown_base) | (crown_base < 0.0) | (crown_base > height)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} of {len(trees)} trees have a "
            f"crown_base_height that is missing, non-finite, negative, or "
            f"above height; expected 0 <= crown_base_height <= height (m)."
        )
