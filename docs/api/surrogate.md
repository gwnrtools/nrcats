---
title: nrcats.surrogate
parent: API Reference
nav_order: 10
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.surrogate`

Surrogate waveform utilities for NR vs NRSur7dq4 comparisons.

Handles loading the surrogate, calling it with NR-compatible parameters,
and wrapping the output as physical-unit pycbc TimeSeries objects.

NRSur7dq4 notes
---------------
* Precessing model — takes full 3-vector spins, returns all modes up to ell=4.
* The ``ellMax`` parameter caps the output (max is 4); ``mode_list`` is not
  supported for precessing models.
* ``f_low`` controls waveform truncation only (start frequency); for NRSur7dq4
  the recommended value is 0 (return the entire waveform).
* ``f_ref`` sets the reference epoch at which the spins are defined; it is in
  **cycles/M** (= M * f_GW where M is in seconds).  For aligned-spin or
  non-spinning systems ``f_ref = f_lower_NR * M_seconds`` is sufficient.  For
  precessing SXS systems Phase 2 extracts the instantaneous spins from
  ``Horizons.h5`` at the epoch corresponding to the chosen ``f_ref``.
* ``dt`` argument is in dimensionless M units.
* Output ``h[(ell,em)]`` is a complex numpy array representing the spin-weight
  -2 spherical harmonic mode ``h_lm`` (same convention as SXS/RIT/MAYA NR
  data: ``h_+ - i h_×`` decomposed as ``Σ h_lm Y_{-2,lm}``).
* Time array ``t_sur`` is in dimensionless M units; ``t_sur[-1] ≈ +100 M``
  is near merger (peak amplitude).

Spin epoch conventions
----------------------
Two cases arise depending on whether the NR waveform starts before or after
the surrogate's minimum training frequency (~0.0165 M·Ω_orbital):

1. **NR shorter than surrogate** (``f_lower_NR > f_min_sur``):
   Pass ``f_low=0`` (full waveform) and ``f_ref=f_lower_NR_dimless``.  The
   surrogate backward-evolves the spins from the NR epoch to its start.

2. **NR longer than surrogate** (``f_lower_NR < f_min_sur``):
   The surrogate cannot extrapolate before its minimum frequency, so
   ``f_ref`` is clipped to ``f_min_sur``.  For **aligned-spin or non-spinning
   systems** the spin components are constant (no precession), so the metadata
   spins remain valid regardless of epoch.  For **precessing SXS systems**
   Phase 2 automatically re-extracts the spin vectors from ``Horizons.h5`` at
   the clipped epoch so that the spin-epoch is always physically consistent.


## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Constants

| Name | Value |
|---|---|
| `NR_MODES` | `[(2, 2), (2, 1), (3, 3), (4, 4), (5, 5), (3, 2), (4, 3)]` |
| `SURROGATE_MODES` | `[(ell, em) for ell, em in NR_MODES if ell <= 4]` |

---

### `load_nrsur7dq4`

```python
load_nrsur7dq4()
```

Load and cache the NRSur7dq4 surrogate model.

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `gwsurrogate surrogate object` |  |

---

### `generate_surrogate_modes`

```python
generate_surrogate_modes(params: dict, total_mass: float, distance: float = 1.0, delta_t_seconds: float = 1.0 / 4096, sim_name: str | None = None, catalog=None, nr_wfm=None, modes='all') -> tuple[dict, float]
```

Call NRSur7dq4 and return physical-unit modes as a pycbc TimeSeries dict.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `params` | `dict` | PyCBC-compatible binary parameter dict as returned by ``CatalogBase.get_parameters()``. Must contain: ``mass1``, ``mass2``, ``spin1x/y/z``, ``spin2x/y/z``, ``f_lower``. |
| `total_mass` | `float` | Total binary mass in solar masses (sets the physical time/frequency scale for the surrogate call). |
| `distance` | `float` | Luminosity distance in Mpc for amplitude scaling (default 1). |
| `delta_t_seconds` | `float` | Desired sample spacing in physical seconds (default 1/4096). |
| `sim_name` | `str` | Simulation name, used for epoch-aligned spin extraction on precessing SXS runs. |
| `catalog` | `CatalogBase` | Catalog instance; enables Phase 2 epoch alignment when the catalog is SXS. |
| `nr_wfm` | `WaveformModes` | Unused; kept for API compatibility. |
| `modes` | `list[tuple[int, int]] or 'all' or None` | Which modes to return. ``None`` (default) keeps the historical :data:`SURROGATE_MODES` subset, so existing per-mode results are unchanged. ``"all"`` returns every mode the model produced, which is what a sphere-averaged or BMS-maximized comparison of a *precessing* binary requires -- see the note on :data:`SURROGATE_MODES`. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `tuple[dict, float]` | ``({(ell, em): pycbc.types.TimeSeries}, f_lower_effective)`` — complex mode time series in physical units with epoch at peak (2,2) amplitude, plus the effective starting GW frequency in Hz. |

#### Raises

| Exception | Condition |
|---|---|
| `ValueError` | If ``f_lower`` is not positive or the surrogate call fails. |

---

### `check_surrogate_prior`

```python
check_surrogate_prior(params: dict, q_max: float = 4.0, chi_max: float = 0.8) -> bool
```

Return True if the binary parameters fall within the NRSur7dq4 prior volume.

NRSur7dq4 is valid for:
- ``q = m1/m2 ∈ [1, 4]``
- ``|χ₁|, |χ₂| ≤ 0.8``
- Waveform length ≥ ~4350 M (not checked here — handled by the surrogate call).

#### Parameters

| Name | Type | Description |
|---|---|---|
| `params` | `dict` | PyCBC-compatible parameter dict. |
| `q_max` | `float` | Maximum mass ratio (default 4). |
| `chi_max` | `float` | Maximum spin magnitude (default 0.8). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `bool` |  |

---

### `surrogate_dict_to_waveform_modes`

```python
surrogate_dict_to_waveform_modes(h_sur_dict: dict, ell_max: int = 4)
```

Wrap a dictionary of pycbc mode TimeSeries into a WaveformModes object.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h_sur_dict` | `dict` | Dict mapping (ell, em) -> pycbc.TimeSeries. |
| `ell_max` | `int` | Maximum ell value to support (default 4). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `WaveformModes` |  |

---

{% endraw %}
