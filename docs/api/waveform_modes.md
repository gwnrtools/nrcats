---
title: nrcats.waveform.modes
parent: API Reference
nav_order: 6
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.waveform.modes`

WaveformModes class and related helpers.


## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Constants

| Name | Value |
|---|---|
| `logger` | `logging.getLogger(__name__)` |

---

## *class* `WaveformModes`

Bases: `sxs_WaveformModes`

Catalog-agnostic container for spin-weighted spherical-harmonic waveform modes.

Inherits from ``sxs.WaveformModes`` (itself an ``numpy.ndarray`` subclass)
so that instances *are* NumPy arrays.  This is an **intentional design
choice**, not technical debt, motivated by three requirements:

1. **Zero-copy performance.**  Mismatch calculations (``match_single_mode``,
   ``match_sphere_averaged``, BMS supertranslation optimization) pass mode
   data directly to PyCBC and SciPy routines that expect array-protocol
   objects.  Inheritance lets NumPy hand them the underlying buffer without
   an intermediate copy.

2. **Wigner-rotation reuse.**  The parent class exposes ``evaluate()``,
   ``index()``, ``LM``, and Wigner-D rotation infrastructure from the
   ``sxs`` / ``spherical`` stack.  Inheriting avoids re-implementing or
   wrapping that non-trivial mathematics.

3. **Downstream compatibility.**  Research workflows in PyCBC, ``scri``,
   and user scripts rely on ``isinstance(wfm, sxs.WaveformModes)``
   checks and on standard NumPy slicing semantics.  Breaking that
   contract would impose migration costs across the gravitational-wave
   community.

**Attribute propagation.**  Because ``numpy.ndarray`` subclasses lose
plain instance attributes during slicing and view-casting, all custom
state (``_filepath``, ``_present_modes``, ``_peak_time_22``,
``_t_ref_nr``, ``verbosity``) is stored inside the ``_metadata`` dict
that ``sxs.TimeSeries`` already propagates.  Property descriptors
provide transparent read/write access.  See ``_custom_meta_keys``,
``__array_finalize__``, ``__copy__``, and ``__deepcopy__`` for details.


### *classmethod* `load_from_h5`

```python
load_from_h5(file_path_or_open_file, metadata={}, verbosity=0)
```

Load SWSH waveform modes from an HDF5 file (RIT/MAYA catalog format).

See ``nrcats.waveform.loaders.load_from_h5`` for full docs.

---

### *classmethod* `load_from_targz`

```python
load_from_targz(file_path, metadata={}, verbosity=0)
```

Load SWSH waveform modes from a ``.tar.gz`` archive (RIT psi4 format).

See ``nrcats.waveform.loaders.load_from_targz`` for full docs.

---

### *property* `filepath`

Return the data file path

---

### *property* `sim_metadata`

Return the simulation metadata dictionary

---

### *property* `metadata`

Return the simulation metadata dictionary

---

### *property* `label`

Return a LaTeX label summarizing key simulation parameters.

---

### *property* `label_nolatex`

Return a plain-text label summarizing key simulation parameters.

---

### `get_parameters`

```python
get_parameters(total_mass: float = 1.0) -> dict
```

Return the initial physical parameters for the simulation.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `total_mass` | `float` | Total Mass of Binary (solar masses). |

#### Returns

| Name | Type | Description |
|---|---|---|
| `dict` | `dict` | Initial binary parameters compatible with PyCBC. |

---

### `get_mode_data`

```python
get_mode_data(ell, em)
```

*No docstring.*

---

### `get_mode`

```python
get_mode(ell, em, total_mass=1.0, distance=1.0, delta_t=None, to_pycbc=True, delta_t_seconds=None, delta_t_Msun=None, t_relax=None)
```

Return a single (ℓ, m) waveform mode, rescaled to physical units.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `ell` | `int` | Spherical-harmonic indices. |
| `em` | `int` | Spherical-harmonic indices. |
| `total_mass` | `float` | Total mass in solar masses (default 1). |
| `distance` | `float` | Luminosity distance in Mpc (default 1). |
| `delta_t_seconds` | `float` | Sample spacing in physical seconds. Mutually exclusive with ``delta_t_Msun``. |
| `delta_t_Msun` | `float` | Sample spacing in dimensionless M units. Mutually exclusive with ``delta_t_seconds``. |
| `delta_t` | `float` | *Deprecated.* Use ``delta_t_seconds`` or ``delta_t_Msun`` instead. |
| `to_pycbc` | `bool` | Return a ``pycbc.types.TimeSeries`` (default True). |
| `t_relax` | `float` | Time (in dimensionless M units) before which the waveform is sliced off to remove junk radiation. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `pycbc.types.TimeSeries or sxs.TimeSeries` |  |

---

### `f_lower_at_1Msun`

```python
f_lower_at_1Msun(t=None)
```

Return the instantaneous GW frequency of the (2,2) mode at 1 M☉.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `t` | `float or None` | Evaluation time in dimensionless M units. If None, returns the frequency at the first sample. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float` | GW frequency in Hz at 1 M☉. Divide by ``total_mass`` [M☉] to get physical Hz. |

---

### `trim_to_relaxation_time`

```python
trim_to_relaxation_time(total_mass, delta_t=1.0 / 4096)
```

Return the (2,2) mode trimmed to start at the relaxation epoch.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `total_mass` | `float` | Total mass of the binary (solar masses). |
| `delta_t` | `float` | Sample spacing in seconds (default 1/4096). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `pycbc.types.TimeSeries` |  |

---

### `f_lower_at_relaxation`

```python
f_lower_at_relaxation(total_mass)
```

Return the GW frequency at the relaxation epoch, in Hz.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `total_mass` | `float` | Total mass of the binary (solar masses). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float` |  |

---

### `get_polarizations`

```python
get_polarizations(inclination, coa_phase, f_ref=None, t_ref=None, tol=1e-06)
```

Sum over modes and return plus/cross GW polarizations.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `inclination` | `float` | Inclination angle (radians). |
| `coa_phase` | `float` | Coalescence orbital phase (radians). |
| `tol` | `float` | Floating-point tolerance for rotation angle computation (1e-6). |

---

### `get_td_waveform`

```python
get_td_waveform(total_mass, distance, inclination, coa_phase, delta_t=None, f_ref=None, t_ref=None, k=3, kind=None, tol=1e-06, lal_convention=False, delta_t_seconds=None, delta_t_Msun=None, t_relax=None)
```

Sum over modes and return GW polarizations rescaled to physical units.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `total_mass` | `float` | Total mass (solar masses). |
| `distance` | `float` | Luminosity distance (megaparsecs). |
| `inclination` | `float` | Inclination angle (radians). |
| `coa_phase` | `float` | Coalescence orbital phase (radians). |
| `delta_t_seconds` | `float` | Sample spacing in physical seconds. |
| `delta_t_Msun` | `float` | Sample spacing in dimensionless M units. |
| `delta_t` | `float` | *Deprecated.* Use ``delta_t_seconds`` or ``delta_t_Msun`` instead. |
| `lal_convention` | `bool` | If True, return h₊ − i h× (LAL convention). Default returns h₊ + i h× (imaginary part = +h×). |
| `t_relax` | `float` | Time (in dimensionless M units) before which the waveform is sliced off to remove junk radiation. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `pycbc.types.TimeSeries(complex128)` |  |

---

### `get_angles`

```python
get_angles(inclination, coa_phase, f_ref=None, t_ref=None, tol=1e-06)
```

Get the inclination, azimuthal and polarization angles
of the observer in the NR source frame.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `inclination` | `float` | Inclination angle in the LAL source frame. |
| `coa_phase` | `float` | Coalescence phase. |
| `f_ref` | `float` | Reference frequency and time. |
| `t_ref` | `float` | Reference frequency and time. |
| `tol` | `float` | Tolerance for rotation angle computation (1e-6). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `dict` | Angles dict with keys ``theta``, ``psi``, ``alpha``, and optionally ``t_ref``, ``f_ref``. |

---

### `to_pycbc`

```python
to_pycbc(input_array=None, delta_t=None, epoch=None)
```

*No docstring.*

---

### `get_nr_coa_phase`

```python
get_nr_coa_phase()
```

Get the NR coalescence orbital phase from the (2,2) mode.

---

### `get_obs_phi_ref_from_obs_coa_phase`

```python
get_obs_phi_ref_from_obs_coa_phase(coa_phase, t_ref=None, f_ref=None)
```

Get the observer reference phase given the observer coalescence phase.

---

### `to_lal`

```python
to_lal()
```

*No docstring.*

---

### `to_astropy`

```python
to_astropy()
```

*No docstring.*

---

### *property* `t_ref_nr`

Fetch the reference time of the simulation.

---

### *property* `peak_time_22`

Dimensionless time of the peak amplitude of the (2,2) mode.

---

### `rotated`

```python
rotated(R)
```

Rotate the waveform modes.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `R` | `quaternionic.array` | Unit quaternion representing the rotation. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `WaveformModes` |  |

---

### `match_single_mode`

```python
match_single_mode(other, ell, em, psd, f_lower, delta_t=1.0 / 4096, f_upper=None)
```

Compute the noise-weighted match for a single spherical harmonic mode.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `other` | `WaveformModes or dict` | The second waveform. |
| `ell` | `int` | Spherical harmonic indices. |
| `em` | `int` | Spherical harmonic indices. |
| `psd` | `pycbc.types.FrequencySeries` | One-sided noise PSD. |
| `f_lower` | `float` | Orbital reference frequency in Hz. |
| `delta_t` | `float` | Sample spacing in physical seconds (default 1/4096). |
| `f_upper` | `float` | Upper frequency cutoff in Hz. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float` | Match value in [0, 1]. |

---

### `match_sphere_averaged`

```python
match_sphere_averaged(other, psd, f_lower, f_upper=None, delta_t=1.0 / 4096, return_rotation=False, total_mass=1.0, distance=1.0)
```

Calculate the match (noise-weighted overlap) between this waveform
and another, integrated over all observer directions on the sphere
(sky-averaged) and maximized over time shift, phase shift, and
active/passive SO(3) coordinate rotation of the source frame.

> **Mathematical Formulation**
> The full multi-mode gravitational-wave strain field $H(t, \theta, \phi) = h_+ - i h_\times$
> as observed at polar angles $(\theta, \phi)$ in the source frame is:
> $$
> H(t, \theta, \phi) = \sum_{\ell=2}^{\infty} \sum_{m=-\ell}^{\ell}
> h_{\ell m}(t) \, {}^{-2}Y_{\ell m}(\theta, \phi)
> $$
> where ${}^{-2}Y_{\ell m}$ are the spin-weight $-2$ spherical harmonics.
> 
> The global overlap between two waveforms $h_1$ and $h_2$, integrated over the entire
> sphere of possible observer directions (sky locations), is defined as:
> $$
> \mathcal{O}_{\text{sphere}}(h_1, h_2) =
> \frac{\int_{S^2} \langle h_1(t, \Omega) \mid h_2(t, \Omega) \rangle_t \, d\Omega}{
> \sqrt{\left[ \int_{S^2} \langle h_1(t, \Omega) \mid h_1(t, \Omega) \rangle_t \,
> d\Omega \right] \left[ \int_{S^2} \langle h_2(t, \Omega) \mid h_2(t, \Omega) \rangle_t \,
> d\Omega \right]}}
> $$
> where $\langle \cdot \mid \cdot \rangle_t$ is the standard frequency-domain noise-weighted
> inner product:
> $$
> \langle u \mid v \rangle_t = 4 \, \mathrm{Re}
> \int_{f_{\mathrm{min}}}^{f_{\mathrm{max}}}
> \frac{\tilde{u}(f) \, \tilde{v}^*(f)}{S_n(f)} \, df
> $$
> 
> By utilizing the orthonormality of the spin-weighted spherical harmonics:
> $$
> \int_{S^2} {}^{-2}Y_{\ell m}^*(\Omega) \, {}^{-2}Y_{\ell' m'}(\Omega) \, d\Omega
> = \delta_{\ell \ell'} \, \delta_{m m'}
> $$
> the angular integral decouples, simplifying the sphere-integrated inner product
> into a simple sum over all common modes $(\ell, m)$:
> $$
> \int_{S^2} \langle h_1(t, \Omega) \mid h_2(t, \Omega) \rangle_t \, d\Omega
> = \sum_{\ell, m} \langle h_{1, \ell m} \mid h_{2, \ell m} \rangle_t
> $$

> **Coordinate Frame Optimization**
> Because the two waveforms may be defined in different coordinate systems (source frames)
> and have arbitrary reference times/phases, we align the target waveform $h_2$ to $h_1$
> by active/passive rigid rotation $R \in SO(3)$, time translation $t_c$, and coalescence phase shift $\phi_c$:
> 1. **Rotation ($R$)**: Rotates the modes using Wigner D-matrices:
>    $$
>    h_{2, \ell m}^{\mathrm{rot}}(t) = \sum_{m'=-\ell}^{\ell} h_{2, \ell m'}(t) \, D^{\ell}_{m' m}(R)
>    $$
> 
> 2. **Time Shift ($t_c$)**: Shifts time via $t \to t - t_c$,
>    implemented efficiently as a linear phase in the frequency domain.
> 3. **Phase Shift ($\phi_c$)**: Twist around the rotated $z$-axis via:
>    $$
>    h_{2, \ell m}^{\mathrm{rot, shifted}}(t) \to e^{-i m \phi_c} \, h_{2, \ell m}^{\mathrm{rot}}(t - t_c)
>    $$
> 
> The method then returns the maximized match (overlap):
> 
> $$
> \mathcal{O}_{\mathrm{max}} = \max_{t_c, \phi_c, R \in SO(3)} \left[
> \frac{
>     \sum_{\ell, m} \langle h_{1, \ell m} \mid
>     h_{2, \ell m}^{\mathrm{rot, shifted}}(t_c, \phi_c, R) \rangle_t
> }{
>     \sqrt{
>         \left( \sum_{\ell, m} \langle h_{1, \ell m} \mid
>         h_{1, \ell m} \rangle_t \right)
>         \left( \sum_{\ell, m} \langle h_{2, \ell m} \mid
>         h_{2, \ell m} \rangle_t \right)
>     }
> } \right]
> $$
> 
> The maximization over $t_c$ is performed efficiently using Fast Fourier Transforms (FFTs),
> $\phi_c$ is maximized analytically, and the SO(3) rotation $R$ (parameterized by
> Euler angles $\alpha, \beta, \gamma$) is optimized using the differential evolution algorithm.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `other` | `WaveformModes or dict` | The second waveform to compare against. Can be a `WaveformModes` object or a dict of PyCBC TimeSeries modes. |
| `psd` | `pycbc.types.FrequencySeries` | One-sided noise power spectral density (PSD). |
| `f_lower` | `float` | Lower frequency cutoff in Hz. |
| `f_upper` | `float` | Upper frequency cutoff in Hz. If None, the Nyquist frequency of the PSD is used. |
| `delta_t` | `float` | Sample spacing in physical seconds (default 1/4096). |
| `return_rotation` | `bool` | If True, returns a tuple `(match, R_opt)` containing the maximum match and the optimal quaternionic rotation. |
| `total_mass` | `float` | Total mass of the binary system in solar masses (default 1.0). |
| `distance` | `float` | Luminosity distance to the source in Mpc (default 1.0). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float or tuple` | If `return_rotation` is False, returns the maximum match value in $[0, 1]$. If `return_rotation` is True, returns `(match, R_opt)` where `R_opt` is the optimal `quaternionic.array` unit quaternion representing the rotation. |

---

### `match_sphere_averaged_bms_maximized`

```python
match_sphere_averaged_bms_maximized(other, psd, f_lower, f_upper=None, j_max=1)
```

Calculate the match maximized over BMS supertranslations in addition to
standard time shift, phase shift, and SO(3) rotation.

> **BMS Supertranslation Mathematical Formulation**
> At null infinity $\mathcal{I}^+$, the asymptotic symmetry group of General Relativity
> is the infinite-dimensional Bondi-Metzner-Sachs (BMS) group. This group is the
> semi-direct product of the Lorentz group and the abelian group of **supertranslations**,
> which correspond to direction-dependent shifts in the retarded time coordinate $u$:
> $$
> u' = u - \alpha(\theta, \phi)
> $$
> where the supertranslation field $\alpha(\theta, \phi)$ is an arbitrary smooth real
> function on the sphere, decomposed into scalar spherical harmonics $Y_{j k}$:
> $$
> \alpha(\theta, \phi) = \sum_{j=0}^{j_{\mathrm{max}}} \sum_{k=-j}^{j} \alpha_{j k} \, Y_{j k}(\theta, \phi)
> $$
> Here, $j=0$ corresponds to a global time translation ($t_c$), $j=1$ corresponds to
> spatial translations (origin shifts), and $j \ge 2$ modes correspond to proper
> supertranslations.
> 
> Under a small supertranslation, the strain waveform modes $h_{\ell m}(u)$ undergo
> first-order mode mixing:
> $$
> h'_{\ell m}(u) \approx h_{\ell m}(u) -
> \sum_{j=0}^{j_{\mathrm{max}}} \sum_{k=-j}^{j} \sum_{p, q}
> \alpha_{j k} \, \mathcal{G}^{\ell m}_{j k, p q} \, \dot{h}_{p q}(u)
> $$
> where $\dot{h}_{p q}(u) = \partial h_{p q} / \partial u$, and $\mathcal{G}^{\ell m}_{j k, p q}$
> are the spin-weighted Gaunt coefficients (integrals of products of three spherical harmonics):
> $$
> \mathcal{G}^{\ell m}_{j k, p q} = \int_{S^2}
> {}^{-2}Y_{\ell m}^*(\Omega) \, Y_{j k}(\Omega) \,
> {}^{-2}Y_{p q}(\Omega) \, d\Omega
> $$
> 
> This method optimizes both the rigid rotation $R \in SO(3)$, time translation $t_c$,
> phase shift $\phi_c$, and the supertranslation coefficients $\alpha_{j k}$ for $j \ge 1$
> up to `j_max` using the Nelder-Mead downhill simplex algorithm to minimize the mismatch
> (maximize the overlap).

#### Parameters

| Name | Type | Description |
|---|---|---|
| `other` | `WaveformModes` | The second waveform to compare against. |
| `psd` | `pycbc.types.FrequencySeries` | One-sided noise power spectral density (PSD). |
| `f_lower` | `float` | Lower frequency cutoff in Hz. |
| `f_upper` | `float` | Upper frequency cutoff in Hz. If None, the Nyquist frequency of the PSD is used. |
| `j_max` | `int` | Maximum spherical-harmonic order of the supertranslation field to optimize (default 1, which corresponds to time translation + spatial translation). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float` | Maximum match value in $[0, 1]$. |

---

### `diff_l2_norm`

```python
diff_l2_norm(other, time_window=None, phase_align=True)
```

Calculate the relative L2 error norm between self and another waveform object.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `other` | `WaveformModes` | The other waveform object. |
| `time_window` | `tuple` | The time window (t_min, t_max) to restrict the calculation. |
| `phase_align` | `bool` | Whether to phase align the waveforms by finding a constant phase shift that minimizes the error. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float` | The relative L2 error norm (i.e. \|\|self - other\|\| / \|\|self\|\|). |

---

{% endraw %}
