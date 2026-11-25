"""Physical speaker center coordinates and disk validation in meters."""

from itertools import combinations
from math import dist, isclose
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _length(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if value < 0 or (value == 0 and not allow_zero):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {bound}")
    return float(value)


def speaker_radii(diameter: ArrayLike, count: int) -> NDArray[np.float64]:
    """Convert a shared diameter or one diameter per speaker to radii."""
    values = np.asarray(diameter)
    if values.ndim == 0:
        return np.full(count, _length(values.item(), "diameter") / 2)
    if (
        values.shape != (count,) or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all() or (values <= 0).any()
    ):
        raise ValueError("diameter must contain one finite positive value per speaker")
    return values.astype(float) / 2


def validate_speaker_centers(centers: ArrayLike, diameter: ArrayLike) -> NDArray[np.float64]:
    """Validate centers and reject disks closer than the sum of their radii.

    Coordinates retain their original order and frame.
    """
    points = np.asarray(centers)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("centers must be a nonempty sequence of (x, y) pairs")
    if points.dtype.kind not in "iuf" or not np.isfinite(points).all():
        raise ValueError("centers must contain finite real coordinates")
    points = points.astype(float)

    radii = speaker_radii(diameter, len(points))
    for (i, first), (j, second) in combinations(enumerate(points), 2):
        separation = dist(first, second)
        clearance = radii[i] + radii[j]
        if separation < clearance and not isclose(separation, clearance, rel_tol=1e-12):
            raise ValueError(f"speaker disks {i} and {j} overlap")
    return points


def generate_speaker_centers(
    n_rows: int,
    n_cols: int,
    diameter: float,
    horizontal_gap: float = 0.0,
    vertical_gap: float = 0.0,
    row_offsets: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Generate centers in row-major order, from lowest y to highest y.

    A scalar offset shifts every second row along x. A sequence supplies one
    x offset per row. The disk bounding box is centered at the origin.
    """
    for name, count in (("n_rows", n_rows), ("n_cols", n_cols)):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    diameter = _length(diameter, "diameter")
    horizontal_gap = _length(horizontal_gap, "horizontal_gap", allow_zero=True)
    vertical_gap = _length(vertical_gap, "vertical_gap", allow_zero=True)
    offsets = np.asarray(row_offsets)
    if offsets.shape not in ((), (n_rows,)):
        raise ValueError("row_offsets must be scalar or contain one offset per row")
    if offsets.dtype.kind not in "iuf" or not np.isfinite(offsets).all():
        raise ValueError("row_offsets must contain finite real distances")
    if offsets.ndim == 0:
        offsets = (np.arange(n_rows) % 2) * float(offsets)

    x = np.arange(n_cols) * (diameter + horizontal_gap)
    y = np.arange(n_rows) * (diameter + vertical_gap)
    centers = np.column_stack(((x + offsets[:, None]).ravel(), np.repeat(y, n_cols)))
    centers -= centers.min(axis=0) / 2 + centers.max(axis=0) / 2
    return validate_speaker_centers(centers, diameter)
