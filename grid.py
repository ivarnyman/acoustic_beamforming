"""Cell-centered Cartesian sampling and FFT coordinates for the speaker array."""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import next_fast_len

from geometry import speaker_radii, validate_speaker_centers

Extent = tuple[float, float, float, float]


@dataclass(frozen=True)
class XYGrid:
    """Uniform samples stored as (y, x), with positions in meters."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    dx: float
    dy: float

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y), len(self.x)

    @property
    def extent(self) -> Extent:
        """Outer cell edges as (xmin, xmax, ymin, ymax), in meters."""
        return (
            float(self.x[0] - self.dx / 2),
            float(self.x[-1] + self.dx / 2),
            float(self.y[0] - self.dy / 2),
            float(self.y[-1] + self.dy / 2),
        )

    def frequency_axes(
        self, *, shifted: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return read-only (fx, fy) in cycles/m, cached by sampling parameters."""
        return _frequency_axes(self.shape, self.dx, self.dy, shifted)


@lru_cache(maxsize=32)
def _frequency_axes(
    shape: tuple[int, int], dx: float, dy: float, shifted: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ny, nx = shape
    fx, fy = np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dy)
    if shifted:
        fx, fy = np.fft.fftshift(fx), np.fft.fftshift(fy)
    fx.setflags(write=False)
    fy.setflags(write=False)
    return fx, fy


@dataclass(frozen=True)
class SimulationGrid:
    """Aperture bounds and three aligned sampling domains."""

    aperture_extent: Extent
    source: XYGrid
    computational: XYGrid
    observation: XYGrid
    source_slice: tuple[slice, slice]
    observation_slice: tuple[slice, slice]


def build_grid(
    centers: ArrayLike,
    diameter: ArrayLike,
    *,
    dx: float,
    dy: float,
    margin: float | tuple[float, float] = 0.0,
    padding_factor: float = 1.0,
    fast_fft: bool = False,
    observation_shape: tuple[int, int] | None = None,
    observation_width_x_m: float | None = None,
    observation_width_y_m: float | None = None,
) -> SimulationGrid:
    """Build the source grid and aligned FFT and observation grids.

    Margin is a per-side distance, either scalar or (x, y). Observation shape
    uses (ny, nx) and defaults to the source shape. Slices index the padded
    computational array and can be used to embed the source or crop results.
    """
    points = validate_speaker_centers(centers, diameter)
    aperture = speaker_bounds(points, diameter)
    xmin, xmax, ymin, ymax = aperture
    center_x, center_y = (xmax + xmin) / 2, (ymax + ymin) / 2
    (ny, nx), (fft_ny, fft_nx) = grid_dimensions(
        aperture, dx=dx, dy=dy, margin=margin, padding_factor=padding_factor, fast_fft=fast_fft,
    )

    observation_shape = resolve_observation_shape(
        (ny, nx), (fft_ny, fft_nx), dx, dy, observation_shape,
        observation_width_x_m, observation_width_y_m,
    )

    # Preserve the source lattice through parity changes by keeping the
    # source and computational index n//2 at the same physical coordinate.
    reference_x = center_x + (0.5 if nx % 2 == 0 else 0.0) * dx
    reference_y = center_y + (0.5 if ny % 2 == 0 else 0.0) * dy
    x = reference_x + (np.arange(fft_nx) - fft_nx // 2) * dx
    y = reference_y + (np.arange(fft_ny) - fft_ny // 2) * dy
    computational = XYGrid(x, y, dx, dy)
    source_slice = _centered_slice((ny, nx), computational.shape)
    observation_slice = _centered_slice(observation_shape, computational.shape)
    sy, sx = source_slice
    oy, ox = observation_slice
    return SimulationGrid(
        aperture_extent=aperture,
        source=XYGrid(x[sx], y[sy], dx, dy),
        computational=computational,
        observation=XYGrid(x[ox], y[oy], dx, dy),
        source_slice=source_slice,
        observation_slice=observation_slice,
    )


def speaker_bounds(points: NDArray[np.float64], diameter: ArrayLike) -> Extent:
    """Return the envelope of the individual disk edges for validated centers."""
    radii = speaker_radii(diameter, len(points))[:, None]
    lower = (points - radii).min(axis=0)
    upper = (points + radii).max(axis=0)
    return float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])


def grid_dimensions(
    aperture: Extent,
    *,
    dx: float,
    dy: float,
    margin: float | tuple[float, float],
    padding_factor: float,
    fast_fft: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the source and padded (ny, nx) shapes without allocating grids."""
    for name, value in (("dx", dx), ("dy", dy), ("padding_factor", padding_factor)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value <= 0
            or (name == "padding_factor" and value < 1)
        ):
            bound = "at least 1" if name == "padding_factor" else "positive"
            raise ValueError(f"{name} must be finite and {bound}")
    margins = np.asarray(margin)
    if (
        margins.shape not in ((), (2,))
        or margins.dtype.kind not in "iuf"
        or not np.isfinite(margins).all()
        or (margins < 0).any()
    ):
        raise ValueError("margin must be a finite nonnegative scalar or (x, y) pair")
    margin_x, margin_y = np.broadcast_to(margins, (2,))

    xmin, xmax, ymin, ymax = aperture
    nx = ceil((xmax - xmin + 2 * margin_x) / dx)
    ny = ceil((ymax - ymin + 2 * margin_y) / dy)
    fft_nx = ceil(nx * padding_factor)
    fft_ny = ceil(ny * padding_factor)
    if fast_fft:
        fft_nx, fft_ny = next_fast_len(fft_nx), next_fast_len(fft_ny)
    return (ny, nx), (fft_ny, fft_nx)


def resolve_observation_shape(
    source_shape: tuple[int, int],
    computational_shape: tuple[int, int],
    dx: float,
    dy: float,
    observation_shape: tuple[int, int] | None,
    width_x_m: float | None,
    width_y_m: float | None,
) -> tuple[int, int]:
    """Resolve optional physical widths to whole cells on the source lattice."""
    shape = source_shape if observation_shape is None else observation_shape
    if (
        not isinstance(shape, tuple) or len(shape) != 2
        or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in shape)
    ):
        raise ValueError("observation_shape must contain two positive integers")
    counts = list(shape)
    for index, width, spacing, axis in ((0, width_y_m, dy, "y"), (1, width_x_m, dx, "x")):
        if width is not None:
            if isinstance(width, bool) or not isinstance(width, Real) or not isfinite(width) or width <= 0:
                raise ValueError(f"observation_width_{axis}_m must be finite and positive")
            # Round up to whole cells while allowing a one-ULP roundoff at exact multiples.
            if width > np.nextafter(computational_shape[index] * spacing, np.inf):
                raise ValueError(f"observation width along {axis} exceeds the computational grid. Increase padding.")
            counts[index] = max(1, ceil(np.nextafter(width / spacing, -np.inf)))
    if any(count > limit for count, limit in zip(counts, computational_shape)):
        raise ValueError("observation_shape must fit within the computational shape")
    return counts[0], counts[1]


def _centered_slice(
    shape: tuple[int, int], outer_shape: tuple[int, int]
) -> tuple[slice, slice]:
    ny, nx = shape
    outer_ny, outer_nx = outer_shape
    y_start = outer_ny // 2 - ny // 2
    x_start = outer_nx // 2 - nx // 2
    return slice(y_start, y_start + ny), slice(x_start, x_start + nx)
