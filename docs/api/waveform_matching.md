---
title: nrcats.waveform.matching
parent: API Reference
nav_order: 7
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.waveform.matching`

Standalone waveform matching and rotation helpers.

These are module-level functions (not bound to WaveformModes) so they can
be unit-tested and used independently of the class.


## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Constants

| Name | Value |
|---|---|
| `MIN_CYCLES_AT_BAND_EDGE` | `2.0` |
| `MIN_BINS_IN_BAND` | `8` |
| `TAPER_FRACTION` | `0.05` |
| `STRAIN_MATCH_N_BETA` | `720` |

---

## *class* `ModeMatchResult`

Outcome of a single-mode match, with the reason it succeeded or failed.

``compute_mode_match`` returns only ``match`` and therefore cannot say why a
NaN appeared.  A bare NaN is indistinguishable between "the mode carries no
signal", "the two waveforms do not overlap in time" and "the requested band
lies below where this waveform has support" -- three different problems with
three different remedies.  This type keeps them separable.

#### Attributes

| Name | Type | Description |
|---|---|---|
| `match` | `float` | Match in [0, 1], or NaN when ``reason`` describes a failure. |
| `reason` | `str` | One of ``'ok'``, ``'band_raised'``, ``'no_overlap'``, ``'zero_norm'``, ``'insufficient_bins'``, ``'no_psd_support'``. Both ``'ok'`` and ``'band_raised'`` carry a usable ``match``; the rest carry NaN. |
| `f_lower_requested` | `float` | The cutoff the caller asked for, in Hz. |
| `f_lower_used` | `float` | The cutoff actually integrated from, in Hz. Differs from ``f_lower_requested`` only when ``reason == 'band_raised'``. |
| `overlap_seconds` | `float` | Duration of the common time window of the two waveforms. |
| `cycles_at_band_edge` | `float` | ``overlap_seconds * f_lower_requested`` -- how many cycles of the *requested* band edge fit in the window. Values below ``MIN_CYCLES_AT_BAND_EDGE`` are what trigger a raise. |
| `n_bins_in_band` | `int` | Resolved frequency bins between ``f_lower_used`` and the upper cutoff. |


### *property* `is_usable`

True when the match is a number rather than a failure code.

---

## *class* `StrainMatchResult`

Outcome of a frame-maximised match between two mode sets.

#### Attributes

| Name | Type | Description |
|---|---|---|
| `match` | `float` | Match in [0, 1] maximised over ``(alpha, beta, t_c)``, or NaN on failure. |
| `mismatch` | `float` | ``1 - match``. |
| `alpha` | `float` | Overall phase of the complex strain, in radians on (-pi, pi]. Applied *after* the mode sum, so it is a rotation of the polarisation basis by ``alpha / 2`` -- not an orbital phase. |
| `beta` | `float` | Rotation of the source frame about the orbital angular momentum axis, in radians on [0, 2pi). Degenerate with the azimuthal viewing angle. |
| `phase_offsets` | `dict` | ``{m: alpha + m * beta}`` wrapped to (-pi, pi], the composite offset actually applied to each ``m``. This is the quantity a single-mode match absorbs and therefore cannot report. |
| `time_shift` | `float` | Time offset applied to ``modes_a``, in seconds, relative to the (2,2)-peak alignment. |
| `match_at_zero_beta` | `float` | The same match with ``beta`` held at 0 (``alpha`` and ``t_c`` still maximised). ``match - match_at_zero_beta`` is what the extra degree of freedom bought, and a large gap means the two mode sets are reported in different frames rather than disagreeing physically. |
| `inclination, azimuth` | `float` | Viewing angles the strain was evaluated at, in radians. |
| `f_lower_used, f_upper_used` | `float` | Band actually integrated over, in Hz. |
| `n_bins_in_band` | `int` | Positive-frequency bins between the two cutoffs. |
| `ms_used` | `tuple` | The ``m`` values that contributed, sorted. |
| `reason` | `str` | ``'ok'``, or a failure code: ``'no_common_modes'``, ``'zero_norm'``, ``'insufficient_bins'``. |


## Functions

### `apply_wigner_rotation_to_mode_dict`

```python
apply_wigner_rotation_to_mode_dict(mode_dict, R, ell_max=4)
```

Apply a Wigner rotation to a dictionary of spherical harmonic modes.

This is useful for rotating the output of ``gwsurrogate`` or
``pycbc.waveform.get_td_waveform_modes`` (which return dicts) into the
NR source frame before computing mode-by-mode matches.

The rotation is applied mode-by-mode via Wigner D-matrices:

    h'_{ℓm}(t) = Σ_{m'} D^{(ℓ)}_{m'm}(R) h_{ℓm'}(t)

where R ∈ SO(3) is a unit quaternion and D^{(ℓ)} is the (2ℓ+1)×(2ℓ+1)
Wigner D-matrix for angular momentum ℓ.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `mode_dict` | `dict` | Keys are ``(l, m)`` integer tuples; values are complex ``pycbc.types.TimeSeries`` objects (or 1-D numpy arrays of matching length). |
| `R` | `quaternionic.array` | Unit quaternion representing the rotation. |
| `ell_max` | `int` | Maximum ℓ to include (default 4). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `dict` | Rotated mode dictionary with the same ``(l, m)`` keys. |

---

### `load_psd`

```python
load_psd(f_lower: float, delta_t: float, waveform_length_seconds: float, psd_name: str = 'aLIGOZeroDetHighPower')
```

Load a named analytic PSD sampled to match a waveform's frequency grid.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `f_lower` | `float` | Low-frequency cutoff in Hz. |
| `delta_t` | `float` | Time step of the waveforms in seconds (sets the Nyquist limit). |
| `waveform_length_seconds` | `float` | Duration of the longest waveform in seconds (sets frequency resolution). |
| `psd_name` | `str` | PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `pycbc.types.FrequencySeries` |  |

---

### `compute_mode_match_detailed`

```python
compute_mode_match_detailed(h_nr, h_sur, f_lower_mode: float, psd=None, psd_name: str = 'aLIGOZeroDetHighPower', f_upper=None, min_cycles: float = MIN_CYCLES_AT_BAND_EDGE, alignment: str = 'peak') -> ModeMatchResult
```

Match one NR mode against one model mode, reporting *why* on failure.

Same computation as :func:`compute_mode_match`, but it returns a
:class:`ModeMatchResult` instead of a bare float and it will **raise the
lower cutoff** rather than return NaN when the requested band lies below
what the common time window can resolve.

> **Why the band is raised rather than the simulation discarded**
> The per-mode cutoff scales as ``|m|/2`` (see :func:`mode_f_lower`), which
> *lowers* the cutoff for ``m=1`` by a factor of two.  That mapping is
> physically correct but presumes the waveform extends to the lower
> frequency.  For a short waveform it does not: the integral then runs over a
> band where one input has no support, the normalisation degenerates, and
> ``pycbc.filter.match`` returns a bare NaN.  Measured, the transition sits at
> ``overlap * f_lower ≈ 1`` cycle.
> 
> Discarding the simulation is the wrong remedy, because the failure is
> per-mode: at the duration where ``(2,1)`` fails, ``(2,2)``, ``(4,4)`` and
> ``(3,2)`` are still fine, and dropping the simulation throws those away
> too.  Raising the cutoff to the resolvable value instead reports the band
> that was *actually* integrated, and is close to value-neutral in practice
> because the detector PSD already suppresses the trimmed region.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h_nr` | `pycbc.types.TimeSeries` | Complex NR and model mode time series, same ``delta_t``. |
| `h_sur` | `pycbc.types.TimeSeries` | Complex NR and model mode time series, same ``delta_t``. |
| `f_lower_mode` | `float` | Requested low-frequency cutoff in Hz, normally ``mode_f_lower(f, m)``. |
| `psd` | `pycbc.types.FrequencySeries or None` | A pre-built one-sided PSD. It is **resampled** onto the frequency grid this function actually integrates on, so its ``delta_f`` need not match the padded segment -- the caller cannot generally know that resolution in advance, since it depends on the common window found here. Outside the supplied range the PSD is taken as infinite, excluding those bins. When ``None`` (default) the PSD is built from ``psd_name``. |
| `psd_name` | `str` | PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``). Ignored when ``psd`` is given. |
| `f_upper` | `float or None` | Upper frequency cutoff in Hz (default: Nyquist). |
| `min_cycles` | `float` | Cycles at the band edge the window must contain (default :data:`MIN_CYCLES_AT_BAND_EDGE`). Pass ``0`` to disable the raise and reproduce the historical behaviour exactly. |
| `alignment` | `str` | Method to align waveforms before matching: 'peak' (default) finds the merger robustly from the end; 'crosscorr' uses a fast-FFT matched filter over the full envelope to find maximum phase coherence. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `ModeMatchResult` |  |

---

### `compute_mode_match`

```python
compute_mode_match(h_nr, h_sur, f_lower_mode: float, psd=None, psd_name: str = 'aLIGOZeroDetHighPower', f_upper=None, min_cycles: float = MIN_CYCLES_AT_BAND_EDGE, alignment: str = 'peak') -> float
```

Compute the noise-weighted match between one NR and one model mode.

Thin wrapper over :func:`compute_mode_match_detailed` that returns only the
numeric match, for callers that do not need the failure reason.  The
signature and return type are unchanged from earlier versions.

Both inputs should be the *complex* strain mode, sampled at the same
``delta_t``.  The function pads to the next power-of-two, builds a PSD at
the matching frequency resolution, and calls ``pycbc.filter.match()``.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h_nr` | `pycbc.types.TimeSeries` | Complex NR mode time series. |
| `h_sur` | `pycbc.types.TimeSeries` | Complex surrogate mode time series. |
| `f_lower_mode` | `float` | Low-frequency cutoff for this mode in Hz. Use ``f_lower * \|m\| / 2`` (GW frequency scales as \|m\| × f_orbital). |
| `psd` | `pycbc.types.FrequencySeries or None` | Pre-built one-sided PSD, resampled onto the grid actually integrated. When ``None`` (default) it is built from ``psd_name``. |
| `psd_name` | `str` | PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``). |
| `f_upper` | `float or None` | Upper frequency cutoff in Hz (default: Nyquist). |
| `min_cycles` | `float` | Cycles at the band edge the common window must contain before the cutoff is raised (default :data:`MIN_CYCLES_AT_BAND_EDGE`). Pass ``0`` to reproduce the historical behaviour exactly. |
| `alignment` | `str` | Method used to align waveforms before matching. 'peak' (default) finds the merger peak robustly from the end; 'crosscorr' uses a fast-FFT matched filter over the full envelope to find the maximum phase coherence. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float` | Match in [0, 1], or ``float('nan')`` if the mode carries no signal, the two waveforms do not overlap in time, or the band cannot be resolved. Use :func:`compute_mode_match_detailed` to tell those cases apart. |

> **See Also**
> compute_mode_match_detailed : same computation, with the reason attached.

---

### `compute_phase_diff_per_cycle`

```python
compute_phase_diff_per_cycle(h_nr, h_sur, alignment: str = 'peak') -> tuple
```

Compute accumulated phase difference per GW cycle over the common window.

Both inputs are the *complex* mode time series (h_lm = h+ - i h×).
The two waveforms are trimmed to their shared time window (both should have
epoch set so t=0 is at peak amplitude), then the total accumulated phase of
each is computed from the unwrapped angle.

The metric returned is::

    phase_diff_per_cycle = |ΔΦ_NR - ΔΦ_sur| / N_cycles_NR   [rad / cycle]

where ``ΔΦ = |φ(t_end) - φ(t_start)|`` is the total phase evolved and
``N_cycles_NR = ΔΦ_NR / (2π)``.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h_nr` | `pycbc.types.TimeSeries` | Complex NR mode time series. |
| `h_sur` | `pycbc.types.TimeSeries` | Complex surrogate mode time series. |
| `alignment` | `str` | Method to align waveforms before computing phase diff. 'peak' (default) or 'crosscorr'. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `tuple[float, float]` | ``(phase_diff_per_cycle, n_cycles_nr)``. Returns ``(nan, nan)`` if either waveform has zero norm or the common window contains fewer than 2 samples. |

---

### `mode_f_lower`

```python
mode_f_lower(f_lower: float, em: int) -> float
```

Return the GW frequency cutoff for mode (ell, m).

GW frequency for the (ell, |m|) mode is approximately |m| times the
orbital frequency: ``f_gw ≈ |m| * f_orbital = |m| * f_lower / 2``
(since the (2,2) mode has ``f_gw = 2 * f_orbital``).

#### Parameters

| Name | Type | Description |
|---|---|---|
| `f_lower` | `float` | GW frequency of the (2,2) mode in Hz (= 2 × orbital frequency). This is what ``CatalogBase.get_parameters()`` returns as ``f_lower``. |
| `em` | `int` | Azimuthal mode number m. For m=0 the mode carries no oscillatory GW power at a well-defined frequency; ``f_lower`` is returned as a conservative lower bound but the result should not be interpreted as a physically meaningful frequency cutoff for that mode. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `float` | Mode-specific GW frequency cutoff in Hz. |

---

### `interpolate_in_amp_phase`

```python
interpolate_in_amp_phase(obj, new_time, k=3, kind=None)
```

Interpolate in amplitude and phase using a variety of methods.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `obj` | `sxs.TimeSeries` | Complex waveform time series. |
| `new_time` | `array_like` | New time axis to interpolate onto. |
| `k` | `int` | Spline order for ``InterpolatedUnivariateSpline`` (default 3). |
| `kind` | `str` | Alternative interpolation: ``'linear'``, ``'quadratic'``, ``'cubic'``, or ``'CubicSpline'``. When specified, ``k`` is ignored. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `sxs.TimeSeries` | Interpolated complex waveform on ``new_time``. |

---

### `sylm`

```python
sylm(ell: int, em: int, inclination: float, azimuth: float = 0.0) -> complex
```

Spin-weight -2 spherical harmonic ``{}_{-2}Y_{lm}(iota, phi)``.

Thin wrapper over LAL so that every caller in this package uses one
convention.  ``spherical`` is also a dependency here but indexes its
harmonics differently, and mixing the two is exactly the kind of silent
convention error this function exists to prevent.

---

### `complete_negative_m`

```python
complete_negative_m(modes: dict) -> dict
```

Add the ``m < 0`` modes implied by equatorial symmetry.

For a non-precessing binary ``h_{l,-m} = (-1)^l conj(h_lm)``.  Verified
against SXS data carrying both signs: the relation holds to 3e-6 (2,2) and
4e-4 (3,3), i.e. to the numerical error of the simulation.

**This is wrong for precessing systems**, which have no such symmetry, and
is why it is a separate function rather than something applied silently
inside the match.  Modes already present are never overwritten.

---

### `compute_strain_match`

```python
compute_strain_match(modes_a, modes_b, delta_t: float, inclination: float, azimuth: float = 0.0, psd=None, psd_name: str = 'aLIGOZeroDetHighPower', f_lower: float = 20.0, f_upper: float = None, n_beta: int = STRAIN_MATCH_N_BETA, taper_fraction: float = TAPER_FRACTION, symmetrize: bool = True) -> StrainMatchResult
```

Match two mode sets as coherent strain, maximised over the frame offset.

Builds the complex strain :math:`h = h_+ - i h_\times` from each mode set,

.. math::

    h^A(t; \alpha, \beta) = e^{i\alpha}
        \sum_{\ell m} e^{i m \beta}\, h^A_{\ell m}(t)\,
        {}_{-2}Y_{\ell m}(\iota, \varphi)

and returns the noise-weighted match maximised over :math:`\alpha`,
:math:`\beta` and the time shift :math:`t_c`.

> **Why these three and no others**
> :math:`t_c` and :math:`\alpha` are the parameters a per-mode match already
> maximises over -- :math:`\alpha` multiplies :math:`h_+ - i h_\times` by a
> constant phase, which is a rotation of the polarisation basis by
> :math:`\alpha/2`.  :math:`\beta` is the one that per-mode matching cannot
> see: it rotates the source about the orbital angular momentum axis and is
> exactly degenerate with the azimuthal viewing angle, so it is unobservable
> and must be maximised over, but because it acts as :math:`e^{im\beta}` it is
> *not* absorbed by any single-mode phase maximisation.  Leaving it in place
> reports a mismatch dominated by a convention difference: measured at
> :math:`\iota = 60^\circ`, 5.7e-2 against 2.1e-4 for SXS:BBH:0201.
> 
> Nothing else is maximised over.  Inclination is a physical parameter of the
> comparison, not a nuisance, so it is an argument rather than a search
> dimension; a mode set that only agrees at some fitted inclination is not the
> same waveform.

> **Argument order and azimuth**
> ``beta`` is applied to ``modes_a``, which makes the result mildly asymmetric
> under swapping the arguments: measured on SXS:BBH:0201 at
> :math:`\iota = 60^\circ`, 1.87e-3 one way against 1.45e-3 the other.  That
> asymmetry is not an artefact -- an independent reference implementation
> reproduces both numbers -- and it is not really about argument order either.
> Rotating the source by ``beta`` is the same as viewing it from azimuth
> ``beta``, so putting the rotation on ``a`` rather than ``b`` amounts to
> evaluating the pair at a different absolute azimuth, and the mismatch
> genuinely depends on azimuth: the same simulation spans 1.35e-3 to 2.09e-3
> over a uniform azimuth grid, a range that contains both of the numbers
> above.  ``beta`` itself is unaffected, agreeing to 2e-4 rad under a swap.
> 
> Use :func:`compute_strain_mismatch_averaged` when a single
> convention-independent number is wanted; use this function when the
> dependence on viewing geometry is the thing being studied.

> **The inner product**
> Two-sided, so that it is correct for the *complex* strain and agrees with
> the usual real-signal convention without a case split:
> 
> .. math::
> 
>     \langle a | b \rangle = 2 \int_{-\infty}^{\infty}
>         \frac{\tilde a(f)\, \tilde b^*(f)}{S_n(|f|)}\, df
> 
> For real :math:`a, b` the negative-frequency half is the conjugate of the
> positive half and this collapses to :math:`4\,\mathrm{Re}\int_0^\infty`.
> Restricting to positive frequencies instead would silently discard the
> counter-rotating content, which is exactly the ``m < 0`` half of the mode
> sum.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `modes_a` | `dict` | ``{(ell, em): complex array}``, both sampled at ``delta_t``. Need not be the same length or contain the same modes; only the common ``(l, m)`` are used. |
| `modes_b` | `dict` | ``{(ell, em): complex array}``, both sampled at ``delta_t``. Need not be the same length or contain the same modes; only the common ``(l, m)`` are used. |
| `delta_t` | `float` | Sample spacing in seconds. Must be the same for both. |
| `inclination` | `float` | Viewing angles in radians. |
| `azimuth` | `float` | Viewing angles in radians. |
| `psd` | `pycbc.types.FrequencySeries` | Supply to override ``psd_name``. Resampled onto the internal grid. |
| `f_lower` | `float` | Band in Hz. ``f_upper`` defaults to Nyquist. Note ``f_lower`` here is the *strain* band edge and should be the (2,2) cutoff, not a value scaled by ``\|m\|/2``: the mode-wise scaling in :func:`mode_f_lower` exists because each mode is filtered separately, whereas the coherent strain carries every mode at once. |
| `f_upper` | `float` | Band in Hz. ``f_upper`` defaults to Nyquist. Note ``f_lower`` here is the *strain* band edge and should be the (2,2) cutoff, not a value scaled by ``\|m\|/2``: the mode-wise scaling in :func:`mode_f_lower` exists because each mode is filtered separately, whereas the coherent strain carries every mode at once. |
| `n_beta` | `int` | Coarse ``beta`` grid size; refined locally afterwards. |
| `taper_fraction` | `float` | Start-only taper, as elsewhere in this module. 0 disables. |
| `symmetrize` | `bool` | Fill in absent ``m < 0`` modes via :func:`complete_negative_m`. Leave on for non-precessing systems and **off** for precessing ones. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `StrainMatchResult` | Carries ``alpha``, ``beta`` and ``phase_offsets = {m: alpha + m*beta}`` alongside the match. |

---

### `compute_strain_mismatch_averaged`

```python
compute_strain_mismatch_averaged(modes_a, modes_b, delta_t: float, inclination: float, n_azimuth: int = 12, **kwargs) -> dict
```

Frame-maximised strain mismatch averaged over azimuth.

:func:`compute_strain_match` is evaluated at a single azimuth, and its value
depends on that choice -- 1.55x between the best and worst azimuth for
SXS:BBH:0201.  Averaging removes both that dependence and the residual
asymmetry under swapping ``modes_a`` and ``modes_b``, since the two argument
orders differ only by which absolute azimuth the pair is evaluated at.  This
is the number to quote when the viewing geometry is a nuisance rather than
the subject.

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `dict` | ``mean``, ``median``, ``min``, ``max`` of the mismatch over the grid, plus ``per_azimuth`` (the individual :class:`StrainMatchResult` objects) so the spread can be inspected rather than assumed small. |

---

{% endraw %}
