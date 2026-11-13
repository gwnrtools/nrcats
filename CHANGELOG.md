# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **NRHJSur3dq8**: `nrcats.surrogate` can drive the aligned-spin NRHJSur3dq8 models of the `nrhjsurrogate` package (>= 1.4; model files on Zenodo, doi:10.5281/zenodo.22262339) by the names `NRHJSur3dq8_AA_v3` and `NRHJSur3dq8_AdA_v3`, through the same `load_surrogate`, `generate_surrogate_modes(..., model=)`, `$NRCATS_SURROGATE` and `surrogate_info()`. The call takes the model's full native span with no post-Newtonian extension, on the model's own grid at the requested spacing, and builds the m < 0 modes by reflection; a precessing binary (in-plane spin components at or above 1e-3, the model's own training bound) or a point outside the model's training box is refused rather than clamped. Every mode is multiplied by `NRHJ_MODE_SIGN = -1`: the model's native modes are in the SXS catalog's convention, an overall sign and a quarter-turn of azimuth away from the LAL convention that NRSur7dq4 (gwsurrogate), SEOBNRv4PHM and NRHybSur3dq8 (LALSimulation) share, measured 2026-10-15 in the inspiral through the relative phases of the modes. `surrogate_info()` records the package version and the SHA-256 of the inspiral file and of the merger-ringdown arm.
- **NRSur7dq4v2**: `nrcats.surrogate` can drive version 2 of the precessing surrogate (arXiv:2609.07873; gwsurrogate >= 1.2.0, data on Zenodo, doi:10.5281/zenodo.22257361). `load_surrogate(name)` loads and caches any model in `SURROGATE_MODELS`; `load_nrsur7dq4()` returns the configured one, chosen by `$NRCATS_SURROGATE` and defaulting to version 1, so every caller that does not choose is unchanged; `generate_surrogate_modes(..., model=)` calls a named model; `surrogate_info()` returns the model name, gwsurrogate version and the data file with its MD5 for manifests. An unknown name fails and says whether it came from the argument or the environment. Every call keeps `ellMax=4`, so version 2's added ell = 5 modes are not returned.
- **SXS resolution levels**: `SXSCatalog.get(..., lev=k)` loads resolution level `Levk` (default: the highest published, as before), fails loudly if another level is obtained, and records `lev_number` on the returned waveform; `SXSCatalog.available_levs()` lists a simulation's levels from the catalog metadata without a download. The mismatch between the two highest levels at one extrapolation order is the finite-resolution truncation estimate the extrapolation-order comparison does not see. Docstrings that said the v3 catalog publishes one level per simulation were wrong (3,964 of 4,150 BBH simulations publish two or more) and are corrected.
- **Current Cosmic Explorer curves**: `CE40` and `CE20` (CE-T2000017 v009, the baseline 40 km and 20 km detectors) join the observatory registry as data files under `nrcats/waveform/data/`; `detectors.psd_from_name` resolves a registry label to its file and falls back to LALSimulation, and every PSD in `matching.py` is built through it. `CE` (P1600143, 2016) is kept for published results and documented as superseded; `ET` is noted equal to the official ET-D curve.
- **Observatory registry**: `nrcats.waveform.detectors` maps an observatory name (`aLIGO`, `CE`, `ET`, `CE-pessimistic`, `CE-wideband`, and the `CE40`/`CE20` above) to a noise curve, a band and a citation, in one place. `detector(name)` resolves a name case-insensitively, `psd_for(name, length, delta_f)` builds its PSD on a given grid, and `band(name, f_waveform)` returns `(f_lower, f_upper, bound_by)`, where `bound_by` says whether the instrument or the length of the waveform set the low cutoff.
- **`nrcats.waveform.frame`**: puts a waveform into a stated BMS frame with no free parameters, using the operations a catalog records. `undo_com_correction()` applies the negated centre-of-mass translation and boost that SXS v3 records, `strip_memory()` removes the memory SXS adds, and `harmonize()` applies whatever a simulation's sidecar records and returns the waveform with a `FrameRecord` saying what was done (a requested treatment that cannot be applied is reported, not skipped). `apply_bms_to_mode_dict()` applies a given BMS element in geometric units, and `to_coprecessing()` rotates a waveform or mode dict into its own time-dependent coprecessing frame, refusing an incomplete ell block rather than zero-filling it. `undo_com_correction()`, `harmonize()` and `apply_bms_to_mode_dict()` need `scri` (the `bms` extra), imported lazily; `to_coprecessing()` does not.
- **Polarisation match at a fixed orientation**: `compute_polarization_match()` projects two mode sets at one inclination and coalescence phase and matches h_+ and h_x separately over a time shift alone, returning a `PolarizationMatchResult`. `compute_strain_match` maximises over a rotation of the polarisation basis and so cannot see the relative phase of h_+ and h_x; this function can. `n_psi > 0` scans basis rotations of `modes_b` only and reports the best angle and its h_+ match.
- **Per-catalog eccentricity threshold**: `NRCatalogClassifier(ecc_threshold=...)` accepts `{catalog: threshold}` as well as a float, since SXS, RIT and MAYA measure eccentricity by different procedures at different epochs; `ecc_threshold_for(catalog)` returns the one that applies. A catalog missing from the mapping raises `ValueError` instead of borrowing another catalog's threshold. The scalar form and its default are unchanged.
- **Lorentz boost in the BMS search**: `match_sphere_averaged_bms_maximized(..., boost_max=)` adds a boost (units of c) to the search, reported as `info["boost_velocity"]`. The default of 0 searches no boost and leaves the search unchanged.
- **Catalog Plugin Registry**: Introduced a dynamic `@register_catalog` decorator and catalog registry framework under `nrcats/registry.py` to allow runtime extensions and registration of custom catalog backends.
- **Waveform Sub-package Refactoring**: Fully split the massive 60 KB monolithic `waveform.py` into a highly maintainable `nrcats/waveform/` sub-package containing modular components:
  - `modes.py`: Defines the `WaveformModes` core interface.
  - `loaders.py`: Dedicated HDF5 and tarball loaders.
  - `units.py`: Waveform interpolation, physical unit scaling, and extraction tools.
  - `matching.py`: Sphere-averaged mismatch computation and Wigner rotation utilities.
  - Fully backward-compatible top-level `waveform.py` shim re-exports all members.
- **YAML Schema Mapping**: Extracted key mapping tables from `metadata.py` into separate catalog schemas (`rit_keys.yaml`, `sxs_keys.yaml`, `maya_keys.yaml`) located in `nrcats/schemas/` to ease catalog extension without touching core code.
- **Conda & PyProject Build Infrastructure**: Modernized package building using `pyproject.toml` with `setuptools >= 64` and `setuptools_scm` for automatic git-tag-based versioning.
- **Interactive Documentation**: Replaced tutorial stubs with two rich, executable step-by-step notebooks/tutorials covering waveform loading, mode visualization, and cross-catalog mismatch validation. Added plain-text HTML fallbacks for LaTeX math rendering and a "Building the docs locally" section in the README.

### Changed
- **Phase drift per cycle**: `compute_phase_diff_per_cycle` now defaults to `alignment="epoch"` (previously `"peak"`) and ends its window at `t_stop=0.0`, the (2,2) peak, so it measures the inspiral drift it is described as measuring. Run to the end of the common span, the numerator had been mostly post-merger (69-98% on the four simulations measured), while almost all the cycles in the denominator are inspiral. `t_stop=None` keeps the whole span. Values returned for the same inputs change; see also the fix below.
- **Sphere-averaged window**: `match_sphere_averaged` and `match_sphere_averaged_bms_maximized` locate the common window on the peak of the rotation-invariant amplitude sqrt(sum |h_lm|^2) instead of the last prominent peak of |h_22|. In a frame rotated by a large angle the (2,2) mode beats and its peak lands on a beat: a chirp rotated by Euler angles (0.7, 1.1, 2.3) matched itself at 0.4926 before this change and at 1.00000 after. The two peaks are close but not identical for an unrotated waveform, so its values can change slightly too.
- **Breaking Change**: `CatalogBase` decoupled from third-party package `sxs`. It no longer inherits from the deprecated `sxs.Catalog` object, removing unused SXS-specific internal metadata records (e.g. `_dict["records"]`, `_dict["modified"]`) from RIT and MAYA catalogs.
- **Breaking Change**: `SXSCatalog` backend migrated to use the new `sxs.Simulations` infrastructure natively (`sxs >= 2024.0.0` required).
- **Breaking Change**: `catalog.simulations_dataframe` for `SXSCatalog` now returns a rich `sxs.SimulationsDataFrame` supporting advanced filtering and properties (e.g. `.BBH`, `.noneccentric`).
- **Breaking Change**: Removed legacy flat metadata fields `catalog.files`, `catalog.select()`, and `catalog.select_files()`. Downstream users can load per-simulation file catalogs dynamically via `sxs.load(sim_id).files`.
- **Waveform Mode Access**: Accessing missing modes when loading a tarball (`load_from_targz`) now emits a descriptive `UserWarning` and keeps track of truly present modes in the `_present_modes` set, avoiding silent zero-padding errors.

### Deprecated
- **`delta_t` Magic Convention**: Deprecated the positional `delta_t` parameter with implicit physical-vs-dimensionless thresholding. Downstream callers are prompted via a `DeprecationWarning` to transition to the explicit parameters `delta_t_seconds` and `delta_t_Msun`.

### Fixed
- **Phase-per-cycle cycle count**: `compute_phase_diff_per_cycle` divided by the *net* phase advance, which counts cycles only while the phase is monotone (one eccentric RIT run read 288 rad/cycle from a count of 0.50 cycles), and cut each mode's window at its own last prominent amplitude peak. The count now comes from the total variation of the phase on the common time window, read only where both series exceed `PHASE_AMP_FLOOR = 1e-3` of their peak. A window with fewer than `MIN_CYCLES_FOR_PHASE = 3` cycles, or whose net advance is below `MIN_PHASE_MONOTONICITY = 0.9` of the total variation, returns NaN with the cycle count instead of a number. The returned tuple keeps its shape.
- **BMS search units**: `match_sphere_averaged_bms_maximized` gave scri physical strain (~1e-18) and a supertranslation in seconds while declaring them dimensionless, so the inhomogeneous term of any ell >= 2 supertranslation was ~1e14 too large; translations (ell = 1), the only case tested, were unaffected. The search now runs in geometric units, and `info["supertranslation"]` is in units of M, as `info["supertranslation_M"]` already was (it was in seconds).
- **Scripts' paths**: `bin/compare_catalogs_q1_nospin.py` no longer inserts one machine's checkout onto `sys.path`, and `bin/compute_matches_alignedspin_rit.py` keeps an existing `LAL_DATA_PATH` instead of overwriting it.
- **Scipy Compatibility**: Fixed compatibility failures with Scipy 1.11+ by replacing the deprecated `scipy.stats.mode` signature with a robust numpy-based unique count fallback during grid generation.
- **Stale LRU Cache**: Fixed a silent bug in `RITCatalog.load()` where `@lru_cache` returned stale cached results when toggling `download=True`. Replaced with a stateful `_rit_catalog_singleton` and a dedicated `reload()` method.
- **Sentinel-Key Heuristic**: Eliminated fragile sentinel key checks (`"relaxed_mass1"`, `"GTID"`) in metadata parsing by introducing an explicit `catalog_type` metadata injection.
- **Network Resilience**: Capped download retry counts to `5` (down from `100` hangs) and implemented an exponential backoff retry strategy with proper `ConnectionError` reporting.
- **RIT Catalog Metadata Keys**: Corrected RIT catalog parsing to match dash-based metadata keys (e.g. `relaxed-mass-ratio-1-over-2`, `relaxed-chi1x`, `freq-start-22`) in modern RIT catalog schema.

### Migration Guide for Downstream Users

#### Type Assertions
Downstream code checking `isinstance(catalog, sxs.Catalog)` will now evaluate to `False` for all catalog objects. Use `isinstance(catalog, nrcats.CatalogBase)` instead.

#### SXS Tooling Interoperability
If you need to pass a catalog directly to downstream `sxs` APIs (e.g. `closest_simulation`), call `.to_sxs()` on any `nrcats` catalog object to obtain a valid `sxs.Simulations` native instance:
```python
# Returns an sxs.Simulations object
sxs_native = catalog.to_sxs()
```

#### Per-Simulation File Resolution
Instead of checking the global flat map `catalog.files`, load files on-demand for a given simulation ID:
```python
# Modern way to resolve files:
sim_files = sxs.load(sim_id).files
```

#### Sampling Time Steps
Replace ambiguous `delta_t` parameters with explicit time steps:
```python
# Old (Deprecated)
modes = wfm.get_mode(2, 2, delta_t=1/4096)

# New (Explicit Seconds)
modes = wfm.get_mode(2, 2, delta_t_seconds=1/4096)

# New (Explicit Dimensionless M)
modes = wfm.get_mode(2, 2, delta_t_Msun=0.5)
```
