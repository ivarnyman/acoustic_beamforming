# Acoustic speaker-array simulator

Simulate baffled speaker arrays with the angular spectrum method (ASM) or synthesize far-field directivity from measured element patterns. The Matplotlib interface supports single tones and frequency sweeps.

## Installation and usage

From the project directory, with Python 3.10 or newer on Linux or macOS, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
python -m pytest
```

`requirements.txt` declares NumPy and SciPy for numerical work, Matplotlib for plotting, and pytest for testing.

## Configuration

`config.py` defines the immutable `SimulationConfig` dataclass and the acoustic conventions. The default example uses a 7 × 7 array at 850 Hz with uniform, in-phase drive to produce a broadside beam. Dimensions, gap widths, grid spacing, and observation distance are in meters. Gaps measure the clearance between neighboring diaphragm edges along each axis before shifting the rows. `row_stagger` shifts rows 1, 3, 5, and so on toward increasing x. It ranges from 0 (aligned) to 1 (one speaker diameter plus one horizontal gap). The padding factor specifies the intended ratio of padded to unpadded grid size per axis.

Complex amplitudes use the time factor `exp(-i omega t)`, where `omega = 2 pi f`. Diaphragm velocities are dimensionless relative values, with unity as the reference amplitude. The initial amplitude settings specify a uniform profile with relative amplitude 1. The configuration also has sound speed in m/s and air density in kg/m³.

`main.py` prints the default configuration and the modeled peak pressure at the configured observation distance, then displays the grid, aperture, and pressure diagnostics.

### Optional speaker types

Omit `speaker_array` to use the existing generated array of identical diaphragms. An explicit array stores physical diaphragm properties once per `SpeakerType` and references them by ID from each `SpeakerElement`.

```python
from aperture import RadialProfile
from config import SimulationConfig, save_configuration
from speakers import SpeakerArray, SpeakerElement, SpeakerType

speakers = SpeakerArray(
    speaker_types=(
        SpeakerType.generic(0.10),
        SpeakerType("mid", "Example midrange", 0.16, RadialProfile("cosine")),
    ),
    elements=(
        SpeakerElement("generic", (-0.18, 0.0)),
        SpeakerElement("mid", (0.0, 0.0), weight=0.8 + 0.2j),
        SpeakerElement("generic", (0.18, 0.0)),
    ),
)
config = SimulationConfig(speaker_array=speakers)
save_configuration("mixed_array.json", config)
```

Explicit positions are `(x, y)` in meters on the existing `z = 0` plane. Their positions and type diameters determine the layout and source bounds. A type with no aperture profile inherits the configuration's radial profile. `SpeakerType.generic()` explicitly labels the idealized flat, unity-response model. Element weights multiply the existing beamforming and optional configuration weights before normalization. For a nonuniform rectangular window, set `rows` and `columns` to describe the explicit elements in row-major order.

Load typed configurations through **Load Configuration** in the interface or with `load_configuration` in Python. The JSON stores each type once and represents complex element weights as `[real, imaginary]`. Response and directivity metadata are optional JSON objects. Response metadata is descriptive. Element `delay_s` is applied as part of the signal chain. The modeled field comes from the prescribed diaphragm velocities, and the ASM aperture already provides diaphragm directivity.

### Generated layout groups

Use `SpeakerGroup` to add rectangular sets of one speaker type to an explicit array. Each group has its own row and column counts, edge-to-edge gaps, row offsets, and position. For example, add two groups to the `speakers` catalog above:

```python
from dataclasses import replace
from speakers import SpeakerGroup

speakers = replace(speakers, groups=(
    SpeakerGroup("generic", rows=2, columns=3, center_m=(0.0, 0.4),
                 horizontal_gap_m=0.02, vertical_gap_m=0.03, row_offsets_m=0.02),
    SpeakerGroup("mid", rows=1, columns=2, center_m=(0.55, 0.45),
                 horizontal_gap_m=0.03),
))
config = replace(config, speaker_array=speakers)
```

All dimensions and positions are in meters. `center_m` locates the center of the group's diaphragm-edge bounding box. A scalar `row_offsets_m` shifts every second row along x. A sequence supplies one x offset per row. The array can contain any combination of groups and individually placed elements. Each group's elements inherit its type's diameter and default to unity weight.

JSON preserves the group definitions. For simulation, `SpeakerArray.flattened_elements` lists explicit elements first, followed by groups in declaration order and each group's elements in row-major order. This is also the order for global speaker weights, tapers, and layout indices. Collision checks use the sum of the two diaphragm radii across the full array and permit tangency. Grid bounds follow the outer diaphragm edges.

### Complex diaphragm frequency response

Attach a `FrequencyResponse` from `frequency_response.py` to a speaker type. Its complex `H(f)` maps normalized input drive to a relative diaphragm velocity scale factor. Supply linear magnitude as a dimensionless ratio and phase in radians under `exp(-i omega t)`. An acoustic SPL curve includes radiation and measurement effects, so it needs a separate conversion before it can serve as this transfer function.

```python
from frequency_response import FrequencyResponse
from speakers import SpeakerType

response = FrequencyResponse.synthetic_lowpass(
    [100, 200, 400, 800, 1600, 3200], corner_hz=800,
)
demo = SpeakerType("demo", "Synthetic low-pass demo", 0.16, frequency_response=response)

# For tabulated drive-to-velocity data:
# response = FrequencyResponse(frequencies_hz, complex_gains, out_of_range="error")
# response = FrequencyResponse.from_magnitude_phase(frequencies_hz, magnitudes, phases_rad)
```

Sample frequencies must be strictly increasing and positive. Magnitudes must be finite and nonzero so their logarithms are defined. Interpolation is linear in log magnitude and unwrapped phase versus log frequency. Stored complex samples are returned exactly at their sample frequencies. When starting from complex samples, automatic phase unwrapping assumes the sampling resolves the phase turns. Use `from_magnitude_phase` with explicitly unwrapped radians to preserve additional turns between samples.

- `error` is the default and rejects frequencies outside the sampled band.
- `zero` gives zero velocity contribution outside the band.
- `hold` uses the nearest endpoint's complex value outside the band.

Saved JSON places `frequency_response` on the speaker type with `frequencies_hz`, `values` as `[real, imaginary]` pairs, `out_of_range`, and `unwrapped_phase_rad`. A type with no response retains the earlier prescribed-velocity model with unity scale. `SpeakerType.generic()` labels this as idealized, and `FrequencyResponse.flat()` provides an explicit unity curve.

At a given frequency, the simulator multiplies each speaker's normalized drive weight by its type's complex gain before sampling the aperture, giving `w_m H_m(f) A_m(x,y,f)`. Drive normalization happens before applying the gain, so a response dip or an out-of-band zero retains its effect. The same velocity field is used for pressure and directivity. This stage uses the existing aperture to model directivity, and response files referenced only by metadata remain descriptors.

## Geometry

`generate_speaker_centers` in `geometry.py` returns an array of `(x, y)` pairs in meters, ordered left to right within each row and from lowest to highest y. The disk bounding box is centered at the origin. For manually placed speakers, pass coordinates to `validate_speaker_centers` to preserve their locations and order. Both functions reject overlapping disks and allow tangency.

`plot_speaker_geometry` in `visualization.py` draws the disks to scale with equal axes in meters. Typed arrays use a distinct color and legend entry for each type. Use `show_indices=True` for zero-based element IDs or `show_type_ids=True` to place type IDs on the disks. Geometry uses physical coordinates independently of the sampling grid.

## Sampling and FFT coordinates

`build_grid` in `grid.py` takes independent `dx` and `dy` in meters and a scalar or per-axis margin on each side. In the default configuration, `grid_spacing_m` supplies x spacing and serves as the fallback for `grid_spacing_y_m`. The diagnostic overlays the source grid lines, speaker disks, and domain bounds.

- The physical aperture bounds enclose the speaker disks. The unpadded source grid covers these bounds plus margins.
- The computational domain extends the source lattice for zero-padding. Use `source_slice` to insert source values into a zero-filled array. Set `fast_fft=True` to round padded lengths up to SciPy's FFT-friendly sizes.
- The low-level observation grid defaults to the source grid and can be resized using `observation_shape=(ny, nx)`. Use `observation_slice` to crop results from the computational grid.

Arrays index samples as `[iy, ix]`, with x along the last axis. Grid extents describe outer cell edges. The source cell edges are symmetric about the aperture midpoint. When grid sizes differ in parity, padding and cropping align the physical sample at index `n // 2` on each axis, so extent midpoints can differ by half a cell.

`frequency_axes()` returns `(fx, fy)` in cycles per meter, ordered to match `fft2` output. The even-length Nyquist bin is negative. Use `shifted=True` to match an `fftshift`-ordered spectrum. For a center-referenced transform, use `fft2(ifftshift(field))` and apply `fftshift` separately if centered frequency order is desired. The center reference is the sample at `n // 2` on each axis.

## Prescribed diaphragm velocity

The radial profile specifies prescribed velocity perpendicular to the diaphragm surface. Acoustic pressure is a separate field computed by propagation. `rasterize_aperture` in `aperture.py` returns the dimensionless complex source field `v[iy, ix]`. Each speaker has a complex weight that defaults to `1+0j`. Weights set each speaker's relative amplitude and phase, and contributions add in cells that sample more than one speaker.

`RadialProfile` supports a uniform piston, a cosine taper over the outer `taper_width_fraction` of the radius, and a Gaussian with standard deviation `gaussian_sigma_fraction` times the radius. Each profile is zero outside the disk. In `peak` mode, the continuous profile has a peak of one before averaging, so a sampled peak can be below one. In `equal_energy` mode, each speaker's sampled profile is scaled so that `sum(abs(v)**2) * dx * dy` equals the physical disk area before applying its weight. The default radial plot compares the continuous peak-normalized shapes. The amplitude and phase maps show the actual rasterized field, with phase hidden wherever amplitude is zero.

Each cell averages a configurable square set of subpixel midpoint samples. Increase `aperture_subsamples` for better edge coverage at a fixed grid spacing, or refine the grid to resolve a narrow profile. Use the source grid to rasterize, then insert the result into a zero-filled computational array with `source_slice` when needed.

### Complex diaphragm profiles

A speaker type's `aperture_profile` can describe spatial amplitude and phase on its circular diaphragm. The default uniform, cosine, and Gaussian profiles use real amplitudes. To add radial phase, set `rim_phase_rad` on `RadialProfile`. It produces phase `rim_phase_rad * (r/R)**2`. Each width parameter and the rim phase can be a scalar or a `FrequencyParameter` table. Idealized parameter tables interpolate the parameter values against log frequency and hold their endpoints outside the table. These tables round-trip through typed configuration JSON.

```python
from aperture import CallableProfile, FrequencyParameter, RadialProfile

gaussian = RadialProfile(
    kind="gaussian",
    gaussian_sigma_fraction=FrequencyParameter((500, 2000), (0.5, 0.25)),
    rim_phase_rad=FrequencyParameter((500, 2000), (0.0, 1.2)),
)

# The callable receives local coordinates divided by the diaphragm radius.
import numpy as np
asymmetric = CallableProfile(
    lambda x, y, f: (1 + 0.3 * x) * np.exp(1j * y * f / 1000),
    coordinates="xy",
    label="Asymmetric phase",
)
```

Use `CallableProfile(function, coordinates="radial")` for a function taking `(r_over_R, f_hz)`, or `coordinates="xy"` for `(x_over_R, y_over_R, f_hz)`. The function receives vectors of in-disk samples and returns a finite scalar or one complex value per sample. Use a deterministic callable so that cached apertures are reproducible. Callables are Python-only. JSON saving reports an error if the configuration contains one.

Subpixel sampling averages the **complex velocity**, which allows neighboring phases to cancel within a cell. Profile normalization sets the scale of the spatial shape before applying the source drive. In peak mode, built-in profiles retain their continuous unit peak and callable profiles use the largest magnitude among the in-disk subpixel samples. In equal-energy mode, the cell averages are scaled so `sum(abs(A)**2) * dx * dy = pi * R**2`. A shape with zero sampled energy stays silent in peak mode and cannot be equal-energy normalized. Beamforming normalization sets the drive weights, while `H(f)` sets the speaker type's drive-to-velocity gain. Both remain outside profile normalization.

To view isolated amplitude and phase at selected frequencies, pass the selected profiles and physical diameters to `plot_diaphragm_profiles`:

```python
from matplotlib import pyplot as plt
from visualization import plot_diaphragm_profiles

plot_diaphragm_profiles(
    (gaussian, asymmetric), (0.16, 0.10), (500, 1000),
    profile_labels=("Speaker 0", "Speaker 1"),
)
plt.show()
```

For an existing typed array, select instances from `config.speaker_array.element_types()` in flattened element order and use their aperture profiles and diameters. A type with no profile inherits `simulation.create_profile(config)`. The isolated plots show sampled profile shape before drive weights or frequency-response gain. The existing array aperture plot shows the combined velocity, including both gains.

## Angular spectrum propagation

`propagate_pressure` accepts the sampled normal velocity, a grid, and one or more nonnegative distances in meters. With a `SimulationGrid`, it embeds the source in the zero-padded computational plane and crops each result to the observation grid. An `XYGrid` instead uses the entire supplied plane directly. All result arrays have shape `(nz, ny, nx)`, including a single distance.

For the `exp(-i omega t)` convention, `k = omega / c` and `kz = sqrt(k² - kx² - ky²)`, with positive real `kz` for propagating waves and positive imaginary `kz` for evanescent waves. The solver maps the velocity spectrum to pressure with `rho * omega * exp(i * kz * z) / kz`. Both FFTs use native bin order. Evanescent components decay with distance and can be omitted with `include_evanescent=False`. The grazing singularity is handled by setting bins with `abs(hypot(kx, ky) / k - 1) <= 32 * eps(float64)` to zero. This omits exact grazing and its roundoff neighborhood. Nearby retained bins can still have large gains.

The result includes complex pressure, magnitude, phase in radians, and relative pressure levels in decibels. The level is referenced to the largest magnitude across all returned planes, so the same reference applies at every requested distance. Zero magnitude has undefined phase (`NaN`) and level `-inf`. Because the input velocity is dimensionless and uncalibrated, pressure is in model units and the reported decibels are relative to the simulated peak.

The finite FFT grid represents a periodically repeated source. Padding separates the repeated copies and makes frequency bins denser, while finer spatial sampling improves aperture and near-field resolution. Periodic copies can still affect the observation region, especially as distance increases. Near-grazing bins can make convergence uneven across domain sizes. Compare multiple padding and sampling settings for the region of interest.

The model assumes a planar source in an infinite rigid baffle, a homogeneous lossless medium, and radiation into the forward half-space. It prescribes diaphragm velocity and models no driver response, acoustic loading feedback on diaphragm motion, finite-baffle diffraction, or room reflections.

### Multi-frequency pressure sweeps

`SimulationSession.recalculate_frequencies` takes a scalar or a vector of positive frequencies in Hz and optional nonnegative z distances in meters. It returns a complex pressure array indexed as `[frequency, distance, y, x]`, with `frequencies_hz`, `distances_m`, `grid.y`, and `grid.x` as its axes. Input order is preserved, including repeated values. Omit distances to evaluate the configured observation plane.

```python
from config import SimulationConfig
from simulation import SimulationSession

config = SimulationConfig()
session = SimulationSession()
result = session.recalculate_frequencies(config, [650, 850, 1050], distances_m=[1, 5])
pressure_at_850_hz_and_5_m = result.pressure[1, 1]
```

At each frequency, steering or focus, signal chains, speaker responses, and diaphragm profiles are evaluated together. The element apertures add to one complex source grid before the FFT. The session reuses geometry, FFT frequency coordinates, and the source FFT when the aperture is unchanged. Frequencies remain separate phasor fields under the single-tone convention. The existing `recalculate` method still returns the single-frequency pressure result and directivity.

For large sweeps, use `iter_frequency_planes` to process one xy plane at a time. Each plane carries its own frequency, distance, indices, and observation grid. Consume or write each plane before moving on to keep memory bounded. The batch method holds the entire requested output array in memory.

```python
from threading import Event

stop = Event()  # A UI cancel action can call stop.set().
for plane in session.iter_frequency_planes(
    config, [650, 850, 1050], distances_m=[1, 5],
    progress=lambda completed, total: print(f"{completed}/{total} planes"),
    should_cancel=stop.is_set,
):
    # Process plane.pressure here.
    pass
```

Both methods report progress after each completed xy plane and raise `concurrent.futures.CancelledError` if cancellation is requested before the next frequency or propagation plane. A running FFT finishes before cancellation takes effect. Callbacks run in the calling thread and can be connected to a worker's progress queue or cancellation event. The desktop controls continue to use the single-frequency workflow.

## Pressure plots and far-field directivity

Set `observation_width_x_factor` and `observation_width_y_factor` to size the pressure plots as multiples of the physical array bounds. For example, `1.0` matches the array on that axis and `2.0` doubles it. The defaults are `1.5` on both axes. The x bounds include the row stagger. If supplied, fixed widths in meters override the corresponding factors, and an explicit `observation_shape` sets the remaining axes in cells. Widths round up to whole cells. Increase `padding_factor` if a width exceeds the computational grid.

To size the observation plots relative to the padded computational domain, use:

```python
from dataclasses import replace

from config import SimulationConfig
from simulation import computational_dimensions_m

config = SimulationConfig()
width_x, width_y = computational_dimensions_m(config)
config = replace(config, observation_width_x_m=0.75 * width_x)
```

The helper returns full widths in meters, including margins, padding, and optional FFT-friendly rounding.

Set `z_evaluation_points` in `SimulationConfig` to control the number of evenly spaced planes in the batch pressure slices. The default is 61, including both endpoints. A value of 1 evaluates only the observation plane in ordinary mode. Focus mode also evaluates the exact observation and focus depths if the sweep misses either. The batch run streams planes and retains the requested xy maps and the samples needed for both cross-sections.

`plot_pressure_planes` selects xy planes by index from a propagation result. `plot_pressure_slice` extracts an xz or yz cross-section at the nearest transverse sample, using the actual z distances for the vertical axis. Pass two or more distinct distances to `propagate_pressure` to create a slice. Both plots use one reference and the same magnitude, relative level, and phase scales across all selected views. The default reference is the peak over the entire propagation result. For comparisons across separate runs, pass the same positive `reference_pressure`, `magnitude_max`, and `level_limits_db` to each plot. The default display ranges are 0–1 for normalized magnitude, −40–0 dB for level, and −π–π for phase.

`far_field_directivity` interpolates the complex velocity FFT at `kx = k sin(theta) cos(phi)` and `ky = k sin(theta) sin(phi)`. Theta measures the angle from +z, and phi measures azimuth from +x toward +y. A negative theta selects the opposite side of the same cut. The returned coefficient `D` gives asymptotic pressure `D exp(i k R) / R` at range `R`. The far-field projection factor cancels the planar pressure spectrum's `1/kz`. This leaves `D = -i rho omega V(kx, ky) / (2 pi)`, with `V` approximated by the cell-area-scaled FFT. The formula gives the forward limit at grazing. Padding makes angular interpolation finer, and any requested directions beyond the spatial Nyquist limits require a finer grid. [Baffled-piston Rayleigh integral](https://homepages.uc.edu/~masttd/papers/2005_jasa_piston.pdf)

`plot_angular_directivity` displays normalized pressure magnitude and normalized far-field intensity in a signed angular cut. With a shared pressure reference `D_ref`, the intensity ratio is `(|D| / D_ref)²`. Relative pressure level uses `20 log10(|D| / D_ref)`, while relative intensity level uses `10 log10(I / I_ref)`. The curves coincide when the intensity reference corresponds to the pressure reference. All levels are relative.

### Broadband pressure and directivity views

Use the Python sweep and plotting APIs to display a selected frequency, observation-point responses, and frequency-dependent directivity. Pressure map selection matches a stored frequency bin. Point responses use the nearest sampled x, y, and z coordinates and show the actual sampled locations in the legend. Frequency-response curves and heatmaps order their frequency axes from low to high. These analysis helpers accept unique frequency bins.

```python
from matplotlib import pyplot as plt
from config import SimulationConfig
from simulation import SimulationSession
from visualization import (
    plot_broadband_pressure, plot_directivity_heatmap, plot_frequency_directivity,
    plot_frequency_pressure, plot_point_frequency_responses,
)

config = SimulationConfig()
session = SimulationSession()
frequencies = [650, 850, 1050]
sweep = session.recalculate_frequencies(config, frequencies, distances_m=[1, 5])
cuts = session.recalculate_frequency_directivity(config, frequencies, directivity_azimuth_deg=0)

# Reuse these example references and display limits when comparing configurations.
reference_pressure = 500.0
reference_directivity = 4000.0
plot_frequency_pressure(sweep, 850, distance_index=1, reference_pressure=reference_pressure)
plot_point_frequency_responses(sweep, [(0, 0, 5), (0.5, 0, 5)], reference_pressure=reference_pressure)
plot_frequency_directivity(cuts, [650, 1050], reference_directivity=reference_directivity)
plot_directivity_heatmap(cuts, reference_directivity=reference_directivity)
plot_broadband_pressure(
    sweep, spectral_weights=[0.25, 0.5, 0.25], weighting="discrete",
    distance_index=1, reference_pressure=reference_pressure,
)
plt.show()
```

Pressure and directivity levels use `20 log10` of a magnitude ratio. A directivity reference is a far-field pressure coefficient in model pressure units times meters. An explicit reference keeps level differences visible between designs, and common display limits keep their color scales comparable. Automatic references use the peak across the pressure sweep or angular cuts. The broadband map uses the peak RMS pressure on its selected plane if no reference is supplied. The angular heatmap displays irregular samples on a logarithmic frequency axis using geometric midpoints. The single-frequency interface and plots remain available.

#### Spectral weights and mean-square pressure

Treat each `P_k` as the pressure phasor produced by unit drive at frequency `f_k`. The broadband map sums `|P_k|²` times the corresponding drive contribution. Supply one nonnegative weight per frequency in the sweep's input order. The function uses those weights as supplied.

| Weighting | Meaning of each weight | Contribution to mean-square pressure |
| --- | --- | --- |
| `discrete` | Mean-square drive for a discrete tone or integrated bin | `|P_k|² * weight_k` |
| `tone_peak_squared` | Squared peak drive amplitude of a tone | `|P_k|² * weight_k / 2` |
| `one_sided_psd` | One-sided mean-square drive density per Hz | `|P_k|² * weight_k * bandwidth_k` |

For PSD input, supply `bin_widths_hz` for known bin bandwidths. With two or more samples and omitted widths, the function uses trapezoidal quadrature over the sampled band. The endpoint weights are half their adjacent intervals. A single PSD sample requires an explicit bandwidth. One-sided PSD values already account for the negative-frequency power folded into the positive side. [SciPy spectral analysis](https://docs.scipy.org/doc/scipy/tutorial/signal.html#spectral-analysis)

The broadband result is mean-square pressure in squared model units. Its relative level is `10 log10(mean_square_pressure / reference_pressure²)` with an RMS reference. Calibrated SPL requires RMS pressure in pascals and the air reference of 20 µPa. Acoustic intensity is a directional energy flux given by the time average of pressure times particle velocity. The pressure-squared plots show only the pressure proxy. [Sound pressure level reference](https://www.osha.gov/otm/section-3-health-hazards/chapter-5) · [Acoustic intensity and RMS pressure](https://ntrs.nasa.gov/citations/19820003985)

For numeric analysis without figures, use `frequency_analysis.point_pressure_responses(sweep, points_m)` or `frequency_analysis.broadband_mean_square_pressure(sweep.pressure, sweep.frequencies_hz, weights, ...)`.

### Imported speaker measurements

The simulator has two propagation paths. In **aperture ASM**, the optional imported velocity response `H(f)` scales the prescribed diaphragm velocity. The aperture and propagation determine the pressure and directivity. In **measured far-field synthesis**, each acoustic pattern already includes the element response and radiation. The sum applies the existing beamforming weights, amplitude, and optional signal chains to the patterns. Attaching a pattern leaves ASM results unchanged.

#### CSV schemas

The loaders in `measured_data.py` accept UTF-8 CSV with `# key=value` metadata preceding the column header. Samples must be finite. The two schemas share the following required metadata.

| Key | Accepted values or meaning |
| --- | --- |
| `reference_input_value` | Positive drive level used for the measurement |
| `reference_input_unit` | `normalized`, `V_peak`, or `V_rms` |
| `calibration` | `calibrated`, `relative`, or `synthetic` |
| `calibration_id` | Identifier for a common amplitude and phase reference |
| `calibration_note` | A description of how the signal was calibrated or generated, including its phase datum |
| `magnitude_unit` | `linear` or amplitude `dB` with `20 log10` relative to one unit of the file's linear quantity |
| `phase_unit` | `deg` or `rad` |
| `time_convention` | `exp(-i omega t)` or `exp(+i omega t)`. The loader conjugates the latter. |

**Velocity response.** Set `schema=velocity_response_v1` and use columns `frequency_hz,magnitude,phase`. Set `quantity=relative_diaphragm_velocity` and `phase_wrapping=wrapped` or `unwrapped`. Prepare the data as a relative complex diaphragm velocity scale at the documented reference drive. The frequency samples must be positive and unique, and the magnitudes must be strictly positive for log interpolation. Wrapped phases are unwrapped in frequency order. Supply unwrapped phase when adjacent samples differ by more than 180°. The loader returns a `FrequencyResponse` and its metadata. The default out-of-band policy is `error`, with `hold` and `zero` also available. Use `load_frequency_response_csv(path)`, then set `SpeakerType.frequency_response` to the returned `.response` and `SpeakerType.frequency_response_metadata` to `.metadata`. Acoustic SPL measurements belong outside this velocity schema.

**Element pattern.** Set `schema=element_pattern_v1` and use columns `frequency_hz,theta_deg,phi_deg,magnitude,phase`. Use pressure magnitude at the stated reference distance. The optional orientation angles default to zero.

| Key | Accepted values or meaning |
| --- | --- |
| `angle_convention` | `polar_from_z_azimuth_from_x_deg`. Polar angle starts at local +z and azimuth increases from local +x toward +y. |
| `reference_distance_m` | Positive far-field distance from the element position to the microphone |
| `sound_speed_m_s` | Positive sound speed used for the measurement |
| `pressure_unit` | `relative` or `Pa`. The Pa option describes peak pressure phasors and requires calibrated or explicitly synthetic data. |
| `phase_reference` | `includes_propagation` if the measurement includes `+kR` in the simulator convention, or `range_removed` if that phase was already removed |
| `yaw_deg`, `pitch_deg`, `roll_deg` | Active rotation from measurement axes to nominal speaker axes, in degrees, as `Rz(yaw) Ry(pitch) Rx(roll)` |

Provide a complete frequency × polar angle × azimuth grid with unique rows, at least one positive frequency, and at least two samples on each angular axis. Supply `theta_deg` in [0, 180] and `phi_deg` in [0, 360), with azimuth coverage around the full circle. Azimuth wraps through the seam. At either pole, supply a common complex value for every azimuth. Interpolation is linear in the real and imaginary parts along polar angle, periodic azimuth, and log frequency. Use angular and frequency resolution fine enough to resolve phase changes and nulls. Queries outside the measured polar range or frequency band raise an error.

With the simulator's phasor convention, the loader stores the coefficient `D = R p exp(-i k R)` for measurements that include propagation phase, and `D = R p` for range-removed phase. The measured array coefficient for unit observation direction `s` is `sum_m [w_m C_m D_m exp(-i k s·r_m)]`. Each pattern is queried in its local frame. An optional `PatternOrientation` on a `SpeakerElement` or `SpeakerGroup` rotates the nominal speaker axes into the array frame. The element remains at its planar position `r_m`. This mounting orientation is used in measured synthesis, while baffled ASM diaphragms lie on the original plane. The far-field position phase follows from the outgoing factor `exp(+i k R)` and the path approximation `R_m ≈ R - s·r_m`. [Mast and Yu, baffled circular piston solutions](https://homepages.uc.edu/~masttd/papers/2005_jasa_piston.pdf)

All used types need patterns on a common pressure, input-drive, and calibration basis. A unit drive weight and `relative_amplitude=1` correspond to the stated reference input. The calibration ID should represent data with a shared amplitude scale and time/phase origin, including any microphone or instrumentation delay corrections. Each pattern's sound speed must match the simulation. Different reference distances are supported through the range conversion above. The measured API returns far-field coefficients only. Use far-field element measurements and interpret the summed patterns in the far field of the entire array. The sum models independent elements and omits mutual coupling, changes in the mounting environment, and near-field reconstruction. The ASM pressure paths remain available for near-field evaluation of the prescribed baffled source. Configuration JSON embeds the processed measurements and orientation so it can be reloaded independently of the CSV.

#### Synthetic piston comparison

[Synthetic piston pattern](examples/stage8/synthetic_piston_pattern.csv) contains analytical data for a 0.16 m diameter uniform baffled piston at a 20 m reference range, computed for `c = 343 m/s`, `rho = 1.21 kg/m³`, and unit relative velocity. [Synthetic velocity response](examples/stage8/synthetic_velocity_response.csv) contains a separate analytical low-pass response with an 800 Hz corner.

```python
from matplotlib import pyplot as plt

from config import SimulationConfig
from measured_data import load_element_pattern_csv
from simulation import SimulationSession
from speakers import SpeakerArray, SpeakerElement, SpeakerType
from visualization import plot_aperture_measured_comparison

pattern = load_element_pattern_csv("examples/stage8/synthetic_piston_pattern.csv")
piston = SpeakerType("piston", "Synthetic baffled piston", 0.16, measured_pattern=pattern)
config = SimulationConfig(
    speaker_array=SpeakerArray((piston,), (SpeakerElement("piston", (0.0, 0.0)),)),
    grid_spacing_m=0.0025, observation_shape=(1, 1), padding_factor=16,
)
session = SimulationSession()
frequencies = [500, 1000, 1500]
aperture = session.recalculate_frequency_directivity(config, frequencies)
measured = session.recalculate_measured_directivity(config, frequencies)
plot_aperture_measured_comparison(aperture, measured, 1000)
plt.show()
```

The comparison defaults to each curve's peak as its 0 dB reference. Pass `reference_directivity` for a common coefficient reference when both curves have compatible amplitude units and reference drive. Set `show_phase=True` when the phase and position origins are aligned. Both sweeps use the existing signed −90° to +90° forward cut and can also be plotted with the broadband directivity tools. Use `session.measured_far_field(config, theta_deg, phi_deg)` for arbitrary angles within the available coverage at `config.frequency_hz`.

## Beamforming

Choose `beamforming_mode="steer"` with `steering_azimuth_deg` and `steering_elevation_deg`, or `beamforming_mode="focus"` with those same angles plus `focus_distance_m`. The distance is measured along the beam from the array center. Older Cartesian configurations can still supply `focus_point_m=(x, y, z)` with `focus_distance_m=None`. For example, run `main(SimulationConfig(beamforming_mode="steer", steering_azimuth_deg=30, steering_elevation_deg=55))` after importing `main` from `main.py` and `SimulationConfig` from `config.py`. Elevation is measured above the xy plane, so broadside is 90° elevation and corresponds to zero polar angle in the directivity plot. Azimuth increases from +x toward +y.

For an outgoing factor `exp(+i k R)`, the far-field path from a speaker at position `r` is approximately `R - s · r` in direction `s`. Steering uses `exp(+i k s · r)` to cancel that direction-dependent phase. Focusing uses `exp(-i k (R_element - R_ref))` with the exact center-to-target distance for each speaker and the origin-to-target distance as a common phase reference. This is phase-only focusing based on speaker centers. Diffraction, finite diaphragms, and unequal path amplitudes affect the resulting focal spot.

Set `speaker_weights` to supply one complex amplitude and phase per speaker and `speaker_taper` for optional nonnegative amplitude multipliers. Both follow the geometry's row-major order and multiply the steering or focusing weights. `weight_normalization="peak"` sets the largest weight magnitude to one. `weight_normalization="power"` fixes the sum of squared magnitudes to `total_squared_weight`, or to the number of speakers by default for comparison with untapered unity weights. This controls the total squared relative drive weight. Radiated acoustic power also depends on the array and its phase relationships. The common `relative_amplitude` is applied afterward.

The weights are applied during aperture rasterization, so the pressure views and far-field directivity use the same weighted velocity field in their FFTs. Focus mode includes the focus depth in the plotted xy planes. Steer mode plots the angular cut at the steering azimuth.

### Speaker-array windows

Set `array_window` to `uniform`, `hann`, `hamming`, `blackman`, `bartlett`, `kaiser`, or `tukey`. Kaiser uses β = 6 and Tukey uses α = 0.5. The simulator multiplies symmetric row and column windows to get one amplitude per speaker. Axes with fewer than three elements use unity. The window multiplies the complex speaker weights and any additional `speaker_taper` before weight normalization. The window plot shows the selected taper on a 0–1 scale.

Tapered windows generally lower sidelobes and widen the main beam. Use `weight_normalization="power"` when comparing windows at the same sum of squared speaker weights.

### Per-element and per-group signal chains

Attach a `SignalChain` to a `SpeakerElement` or `SpeakerGroup`. A group shares its chain across its generated elements. Use `element_signal_overrides` to replace the chain at selected zero-based, row-major indices within that group. An override of `SignalChain()` bypasses the group processing for that element.

```python
from signal_chain import Crossover, SignalChain
from speakers import SpeakerElement, SpeakerGroup

low = SignalChain(crossover=Crossover("lowpass", 1200, order=4, family="linkwitz_riley"))
high = SignalChain(crossover=Crossover("highpass", 1200, order=4, family="linkwitz_riley"),
                   gain=0.8, delay_s=0.00015)
woofer_group = SpeakerGroup("woofer", rows=1, columns=3, signal_chain=low,
                           element_signal_overrides=((2, SignalChain()),))
tweeter = SpeakerElement("tweeter", (0.0, 0.30), signal_chain=high)
```

Gain is a nonnegative linear amplitude multiplier, polarity is `+1` or `-1`, phase offset is in radians, and delay is in seconds. Under `exp(-i omega t)`, a delay contributes `exp(+i omega delay_s)`. The effective drive is `w_m(f) C_m(f) H_m(f)`, where `w_m` is the existing normalized drive, `C_m` is the optional gain, polarity, phase, delay, and crossover, and `H_m` is the speaker type's drive-to-velocity response. The chain acts after drive-weight normalization. Identity processing gives `C_m = 1`. An existing element `delay_s` fills in a zero or absent chain delay. Matching declarations apply once, and conflicting nonzero values raise an error.

The filters are analog prototypes evaluated at `s = -i omega`. Butterworth low-pass, high-pass, and band-pass edges are at −3 dB. Use two increasing edge frequencies for a band-pass. Butterworth band-pass `order` is the prototype order, so the realized filter has twice as many poles. [SciPy Butterworth design](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) · [SciPy analog pole-zero response](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs_zpk.html)

Linkwitz–Riley low-pass and high-pass sections use two cascaded Butterworth sections. Their edge level is −6 dB and `order` is the total even order. LR2 branches have opposite polarity. Set an explicit `polarity=-1` on one branch if you want their electrical outputs to add in phase. The acoustic sum still depends on diaphragm responses and path lengths. [Linkwitz crossover notes](https://www.linkwitzlab.com/filters.htm)

Python and JSON preserve individual chains, group defaults, and overrides. JSON stores each override as `[index, chain]`. For a transfer-function plot, pass the chain and a positive frequency sweep to the plotting helper. Optionally pass a type's `frequency_response` to show the separate diaphragm response and the combined drive-to-velocity transfer. The plot uses wrapped phase in radians and a −120 dB display floor.

```python
import numpy as np
from matplotlib import pyplot as plt
from visualization import plot_transfer_functions

plot_transfer_functions(high, np.geomspace(100, 10000, 400))
plt.show()
```

## Interactive interface

After installing the dependencies, launch the desktop interface with `python interface.py` using a graphical Matplotlib backend.

- Commit edits with Enter or by clicking outside the textbox to update the layout. The layout labels the speaker order. Enter complex weights in that order, separated by commas, for example `1, 0.5-0.5j` for a two-speaker array. Leave the field blank for unity weights.
- Set the frequency, observation distance, and aperture profile. Enter taper width as a fraction of the diaphragm radius. Select **steer** and enter azimuth and elevation to steer the beam. Select **focus** and enter azimuth, elevation, and **Focus distance (m)** together to set the 3D target. The distance is measured along the steered direction. **Observation z (m)** sets the plane displayed in the pressure map.
- Set **Pressure x width (m)** and **Pressure y width (m)** to control the full area sampled in the pressure map. Leave either field blank to use the configured size for that axis. Increase **FFT padding** if a requested width exceeds the computational grid.
- Choose a **Speaker window** to preview its amplitude across the array. Select **Recalculate** to apply it to the pressure and directivity plots.
- Select **Recalculate** to update the aperture velocity amplitude, relative pressure map, and angular directivity. The calculation runs in the background. The plots show the last successful calculation while edits are pending. Changing only the observation distance reuses the source FFT and far-field pattern.
- Enter a file path in **JSON**, then select **Save Configuration** or **Load Configuration**. The file stores all simulation settings, with complex weights encoded as `[real, imaginary]` pairs. **Reset** restores startup settings. Loading or resetting updates the layout immediately, and the acoustic plots update on the next recalculation.

### Speaker library and sweeps

Open **Array & sweep settings** for the optional controls. The main window starts in the generic, single-frequency ASM mode.

1. **Choose an array.** Select a preset and **Load preset**, or build your own. The woofer/tweeter preset uses an idealized 1.8 kHz Linkwitz–Riley crossover. The three-way preset uses idealized Butterworth low-pass, band-pass, and high-pass filters with edges at 500 Hz and 2.5 kHz. Both include demonstration sweeps and idealized flat driver responses.
2. **Manage speaker types.** Use **Previous type** and **Next type** to browse types. Enter a new ID, name, and diameter and select **Save type** to add one. Saving an existing ID updates that type. For **Velocity H**, choose `keep`, `flat`, `synthetic_lowpass` with a corner in Hz, or `csv` with a Stage 8 velocity-response file path. The optional **Element pattern CSV** accepts a Stage 8 angular pattern file. **Export library** writes a `{"speaker_types": [...]}` JSON document. **Import / merge** replaces matching IDs and adds new ones. Save the full configuration from the main window to include placements, crossovers, sweep frequencies, and solver mode.
3. **Place groups.** Enter a type ID, row and column counts, and the center, edge gaps, and row offsets in meters. Use a scalar offset for alternating rows or a comma-separated offset per row. Leave **Group number** blank and select **Save group** to add a group. Enter a number and select **Read group** to edit a saved group. Each group can have its own crossover. Enter the filter kind, `butterworth` or `linkwitz_riley`, cutoff frequency or pair, and order. Speaker-specific mode uses these placements for the geometry. Individual placements and signal-chain overrides from loaded array JSON are preserved.
4. **Run a sweep.** Choose **Speaker-specific** or **Generic** and select **ASM** or **Measured far field**. Enter comma-separated frequencies or a linear range such as `500:1500:3`. A blank sweep field uses the main **Frequency (Hz)** field. Select **Apply run settings**, then **Recalculate**. Measured far-field mode requires compatible element patterns for all used types and displays angular directivity only.
5. **View completed results.** Enter a calculated frequency and select **Apply display**. The pressure view supports `level`, `magnitude`, and `phase`. Reference fields accept common positive values for comparisons and use peak normalization when blank. **Sweep plots** opens angular directivity versus frequency and, for ASM, the pressure response at the center of the observation grid. To open a mean-square-pressure map, supply one spectral weight per frequency in evaluation order and choose `discrete`, `tone_peak_squared`, or `one_sided_psd`. PSD bin widths are optional for a multi-bin sweep.

The sweep worker calculates one frequency at a time and saves a bounded, possibly subsampled copy of each field for display. Changing the displayed frequency, pressure view, level floor, or reference uses those completed results. Edits invalidate a running calculation. A new explicit request replaces any queued request and starts after the current calculation yields. ASM outputs are uncalibrated, and the interface displays relative levels in both propagation modes.

The interface reports sampling, padding, and FFT-size warnings before allocating the computational field. These are heuristics that flag fewer than six cells per wavelength or diameter, fewer than three across a tapered profile width, less than 2× padding or half a wavelength of added margin on an axis, and FFTs with at least 1,048,576 cells. Confirm convergence by comparing the region of interest at finer spacing and larger padding. The synthetic piston interface test covers 1 and 1.5 kHz. At the pattern's sampled angles, refining from 40 mm spacing with 3× padding to 2.5 mm with 16× padding brought the maximum complex directivity error from about 3.45% to 0.08% of the reference peak.

## Analytical validation and profiling

Run the piston benchmarks and the performance profile with:

```bash
python validation.py --output stage9_results/validation.json
```

The reference is a uniform circular piston in an infinite rigid baffle. The on-axis complex pressure is `rho c v0 [exp(i k z) - exp(i k sqrt(z² + a²))]`. The far-field coefficient relative to broadside is `2 J1(k a sin(theta)) / (k a sin(theta))`, with limit one at broadside. The benchmark compares complex fields, so it checks phase and scaling as well as magnitude. [Mast and Yu, *Simplified expansions for radiation from a baffled circular piston*](https://homepages.uc.edu/~masttd/papers/2005_jasa_piston.pdf)

The report separates spatial resolution, padding, boundary averaging, and observation distance. For the 0.2 m piston at 720 Hz, the on-axis error at 0.12 m was 116% with 2× padding, 5.7% with 36×, 24% with 40×, and 4.4% with 56×. At 0.8 m, it was still 18% with 56× padding. A denser FFT frequency grid changes the distance from its bins to the grazing circle, so convergence can be nonmonotonic. At fixed computational width, finer sampling improved the far-field pattern and disk area while on-axis error remained dominated by FFT-domain effects. Averaging more boundary samples reduced disk-area error overall, with small nonmonotonic changes as the midpoint samples moved.

The isolated grazing bin is removed by the documented cutoff. A retained bin just above the corresponding grazing frequency in the mode test still had a gain of about 1,414 times `rho c`. The isolated evanescent mode followed its analytical exponential decay, and excluding evanescent modes substantially changed the piston result at the source plane. Use padding and resolution sweeps at the observation distances of interest. The numerical solver models periodic copies, a finite sampled bandwidth, a prescribed velocity in a lossless homogeneous medium, and an infinite rigid baffle. An angular cut can miss lobes at other azimuths. The far-field coefficient is asymptotic and the reported levels remain uncalibrated and relative.

The batch and interactive calculations reuse the aperture FFT for pressure and directivity. Frequency axes are cached by grid shape and spacing. Pressure propagation reuses scratch buffers, and far-field interpolation reads four FFT bins per angle without allocating shifted copies of the full spectrum.

On one local 800 × 800 workload, the median over seven warmed repetitions changed from about 91 to 80 ms for four pressure planes and from 6.63 to 0.18 ms for 721 directivity samples. Incremental traced peaks went from 60.4 to 52.2 MiB and from 19.7 to 0.13 MiB respectively. Timings vary by machine. The measured peaks are allocations traced during each operation after setup. Process RSS and memory already held by the prepared spectrum are outside those measurements.

## Compare saved experiments

Use files saved by the interface or the included example configurations:

```bash
MPLBACKEND=Agg python compare.py examples/stage9/broadside.json \
  examples/stage9/steered.json examples/stage9/steered_tapered.json \
  --azimuth-deg 0 --distance-m 0.4 --output stage9_results
```

The command writes `directivity.png`, `beam_metrics.png`, `pressure_fields.png`, and `metrics.json`. All directivity curves use the same azimuth and coefficient reference. The pressure maps use the same physical axis limits and pressure reference. By default each reference is the largest value over all cases for that quantity. To reproduce scales across runs, supply `--pressure-reference` and `--directivity-reference`, along with common `--floor-db` and `--ceiling-db` values. If `--distance-m` is omitted, each configuration supplies its own pressure-map distance.

Beamwidth is the separation between interpolated half-power crossings around the dominant peak of the cut. The sidelobe metric is the strongest lobe beyond the nearest resolved main-lobe minima, relative to that case's peak. Crossings or sidelobes that the sampled cut cannot resolve are reported as unavailable. In the example at a 25° steered angle, tapering changed the sidelobe level from −12.6 to −25.1 dB and widened the beam from 15.7° to 20.5° at the same sum-of-squared-weights budget.
