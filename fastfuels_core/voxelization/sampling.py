# Core imports
from __future__ import annotations

# External imports
import numpy as np
from numpy import ndarray
from scipy.ndimage import distance_transform_edt


def compute_crown_probability_field(
    volume_fraction_array: ndarray,
    alpha: float,
    beta: float,
    rho: float = None,
) -> tuple[ndarray, int]:
    """Precompute the deterministic inputs to occupancy sampling.

    Returns the joint crown-occupancy probability grid and the number of voxels
    ``n`` to draw from it. Both depend only on
    ``(volume_fraction_array, alpha, beta, rho)`` -- not on any random seed -- so
    a caller drawing multiple occupancy realizations of the same crown can
    compute this once and pass it to :func:`sample_occupancy` repeatedly,
    avoiding a redundant (and EDT-heavy) field recompute per realization.
    """
    # Create a probability mask for the crown grid
    mask_bool = np.where(volume_fraction_array > 0.0, 1.0, 0.0)
    field = _compute_joint_probability(mask_bool, alpha, beta)

    # Number of voxels to occupy from the crown density
    if rho is None:
        rho = _estimate_crown_density(np.sum(mask_bool))
    n = int(np.count_nonzero(mask_bool) * rho)

    return field, n


def sample_occupancy(
    volume_fraction_array: ndarray,
    field: ndarray,
    n: int,
    seed: int = None,
) -> ndarray:
    """Draw one stochastic occupancy realization from a precomputed field.

    ``field`` and ``n`` come from :func:`compute_crown_probability_field`. Only
    this step consumes ``seed``: different seeds yield different occupancy
    realizations drawn from the same shared field.
    """
    sampled = _sample_voxels_from_probability_grid(n, field, seed)

    # Make non-zero selected voxels 1
    selected = np.where(sampled > 0, 1.0, 0.0)

    return selected * volume_fraction_array


def sample_occupied_cells(
    volume_fraction_array: ndarray,
    alpha: float,
    beta: float,
    rho: float = None,
    seed: int = None,
) -> ndarray:
    """One-shot occupancy sampling: build the crown probability field and draw
    a single realization from it.

    Equivalent to ``sample_occupancy(vfa, *compute_crown_probability_field(vfa,
    alpha, beta, rho), seed)``; kept as the convenience path for callers that
    only need one realization.
    """
    field, n = compute_crown_probability_field(volume_fraction_array, alpha, beta, rho)
    return sample_occupancy(volume_fraction_array, field, n, seed)


def _compute_joint_probability(mask: ndarray, alpha: float, beta: float) -> ndarray:
    """
    Combines the horizontal and vertical probability spatial to create a joint
    probability grid for the crown mask.
    """
    # Build the horizontal and vertical probability spatial
    horizontal_probability = _compute_horizontal_probability(mask, alpha)
    vertical_probability = _compute_vertical_probability(mask, beta)

    # Combine the probability spatial for joint probability
    joint_probability = horizontal_probability * vertical_probability
    joint_probability /= np.max(joint_probability)

    return joint_probability


def _compute_horizontal_probability(mask: ndarray, alpha: float) -> ndarray:
    """
    Builds a horizontal probability grid from a binary mask using a Euclidean
    Distance Transform (EDT). The function computes the EDT of the input
    mask, inverts the EDT values by subtracting them from the maximum value,
    applies a power function using the alpha parameter, and finally
    normalizes the result. Any NaN values in the final probability grid are
    replaced with 0.
    """
    # Compute the Euclidean Distance Transform (EDT) of the mask
    edt = distance_transform_edt(mask)

    # Inverse the distance transform
    # Add a small value to avoid zero probabilities
    max_dist = np.max(edt) + 1e-6
    edt = max_dist - edt
    edt[edt == max_dist] = 0

    # Apply the alpha parameter
    edt = np.nan_to_num(mask * edt**alpha)

    # Normalize and replace nans with 0
    horizontal_probability = edt / np.max(edt)
    horizontal_probability = np.nan_to_num(horizontal_probability)

    return horizontal_probability


def _compute_vertical_probability(mask: ndarray, beta: float) -> ndarray:
    """
    Builds a vertical probability grid from a binary mask. The function
    computes a 1D grid along the vertical axis (axis=2) of the input mask,
    raises this grid to the power of beta, and then broadcasts this
    transformed grid across the 2D plane of each layer of the mask (along
    axis=0 and axis=1), and multiplies it with the mask. Finally, the result
    is normalized and any NaN values are replaced with 0.
    """
    # Create a grid for the vertical axis (axis=2)
    z_grid = np.arange(mask.shape[0]).astype(float)

    # Compute the power of the vertical axis grid with beta
    z_power_beta = z_grid**beta

    # Add a small value to avoid zero probabilities
    z_power_beta += 0.01

    # Broadcast the vertical probability grid along axis=0 and axis=1
    vertical_probability = mask * z_power_beta[:, np.newaxis, np.newaxis]

    # Normalize and replace nans with 0
    vertical_probability /= np.max(vertical_probability)
    vertical_probability = np.nan_to_num(vertical_probability)

    return vertical_probability


def _estimate_crown_density(volume: float, threshold: float = 16) -> float:
    if volume < threshold:
        return 1
    return 0.5


def _sample_voxels_from_probability_grid(
    n: int, joint_probability: ndarray, seed: int = None
) -> ndarray:
    """
    Samples n voxels from the joint probability grid. Sampling is weighted by
    the joint probability of each voxel, such that voxels with higher joint
    probability are more likely to be sampled. Voxels are sampled without
    replacement.

    Uses the Efraimidis-Spirakis / Gumbel-top-k method: each candidate voxel i
    is given a key ``log(u_i) / w_i`` (u_i ~ U(0, 1), w_i the voxel's
    probability), and the n voxels with the largest keys are kept. This draws
    from the same distribution as sequential weighted sampling without
    replacement -- i.e. ``np.random.choice(..., replace=False, p=...)`` -- but
    in a single O(N) pass rather than numpy's rejection loop, whose cost grows
    with n. See Efraimidis & Spirakis (2006), Inf. Process. Lett. 97(5).
    """
    rng = np.random.default_rng(seed)

    # If joint probability is all zeros, return an empty array
    if np.all(joint_probability == 0):
        return joint_probability

    # Flatten and normalize the joint probability to sum to one
    jp_flat = joint_probability.flatten()
    jp_flat = jp_flat / np.sum(jp_flat)

    # If jp_flat contains nan values, return an empty array
    if np.any(np.isnan(jp_flat)):
        return joint_probability

    # Only voxels with positive probability are candidates for selection.
    # (numpy's weighted choice likewise cannot draw more than this many.)
    candidates = np.flatnonzero(jp_flat)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > candidates.size:
        raise ValueError(
            f"Cannot sample more voxels than have positive probability "
            f"({n} requested, {candidates.size} available)"
        )

    selected_flat = np.zeros(joint_probability.size)
    if n == 0:
        return selected_flat.reshape(joint_probability.shape)

    # Efraimidis-Spirakis keys; the n largest keys are the sampled voxels.
    # argpartition finds the top n in O(N) without a full sort.
    weights = jp_flat[candidates]
    keys = np.log(rng.random(candidates.size)) / weights
    kth = candidates.size - n
    chosen_indices = candidates[np.argpartition(keys, kth)[kth:]]

    # Build flat selection array and reshape to original shape
    selected_flat[chosen_indices] = jp_flat[chosen_indices]
    return selected_flat.reshape(joint_probability.shape)
