---
title: nrcats.sxs
parent: API Reference
nav_order: 4
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.sxs`

SXS catalog interface.

Wraps the ``sxs`` package (≥ 2025.0.0) to provide access to the
Simulating eXtreme Spacetimes (SXS) catalog of numerical-relativity BBH
waveforms stored on Zenodo.

Key design decisions
--------------------
* **Singleton pattern** – ``load()`` stores its result in ``_sxs_catalog_singleton``
  so that repeated calls don't re-parse the ~2000-entry catalog JSON.  Pass
  ``download=True`` or call ``reload()`` to force a fresh download.

* **Lazy path resolution** – individual simulation file paths are resolved on
  demand via ``waveform_filepath_from_simname()`` rather than upfront (which
  would require ~2000 ``sxs.load()`` calls at catalog-load time).  The
  ``_add_paths_to_metadata()`` helper seeds stub empty strings so downstream
  code that checks for key presence never sees a ``KeyError``.

* **Auto-supersede** – ``get()`` passes ``auto_supersede=True`` to
  ``sxs.load()`` so deprecated simulation IDs are resolved transparently.

Public classes
--------------
SXSCatalog
    Registered under the tag ``"SXS"`` in the catalog plugin registry.


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

## *class* `SXSCatalog`

Bases: `catalog.CatalogBase`

Catalog interface for the SXS (SpEC) NR waveform collection.

Wraps the ``sxs`` package to provide a ``CatalogBase``-compatible
interface over the SXS Zenodo-hosted catalog.  Key design points:

- Metadata is loaded via ``sxs.load("catalog", download=None)``,
  which returns a ``sxs.Catalog`` dict of all ~2000 simulations.
- Path columns in the metadata are set to empty strings at load
  time (lazy stub) because resolving real on-disk paths would
  require one ``sxs.load()`` call per simulation (~2000 requests).
  Paths are resolved on demand inside ``get()``.
- ``get()`` delegates to ``sxs.load(sim_name, auto_supersede=True)``
  and wraps the returned ``sxs.WaveformModes`` in the local
  ``WaveformModes`` subclass.
- A module-level singleton prevents redundant catalog loads when
  ``load()`` is called multiple times in the same process.

> **Example**
> >>> import nrcats as nrcat
> >>> cat = nrcat.SXSCatalog.load(download=False)
> >>> wfm = cat.get("SXS:BBH:0001")


### *classmethod* `load`

```python
load(download: bool | None = None, verbosity: int = 0, **kwargs) -> SXSCatalog
```

Load the SXS catalog.

This is a wrapper around ``sxs.load``.  The result is cached in a
module-level singleton; pass ``download=True`` or call
``SXSCatalog.reload()`` to force a fresh download.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `download` | `(None, bool)` | If False, this function will look for the catalog in the cache and raise an error if it is not found. If True, this function will download the catalog and raise an error if the download fails. If None (the default), it will try to download the file, warn but fall back to the cache if that fails, and only raise an error if the catalog is not found in the cache. |

---

### *classmethod* `reload`

```python
reload(**kwargs) -> SXSCatalog
```

Force a fresh download and replace the cached singleton.

Equivalent to calling ``sxs.Simulations.reload()`` and then reloading.

---

### `waveform_filename_from_simname`

```python
waveform_filename_from_simname(sim_name: str) -> str
```

Return the bare filename for the strain waveform file.

Resolves the path on demand by calling ``sxs.load(sim_name,
download=False)`` and extracting ``sim.strain_path[0]``.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Filename component only (no directory). |

---

### `waveform_filepath_from_simname`

```python
waveform_filepath_from_simname(sim_name: str) -> str
```

Return the absolute local path for the strain waveform file.

Path resolution is lazy: this calls ``sxs.load(sim_name,
download=False)`` each time it is invoked.  The file may not yet
exist locally if it has never been downloaded.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | POSIX-style absolute path string. |

---

### `metadata_filename_from_simname`

```python
metadata_filename_from_simname(sim_name: str) -> str
```

Return the bare filename for the per-simulation metadata file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Filename component only (no directory). |

---

### `metadata_filepath_from_simname`

```python
metadata_filepath_from_simname(sim_name: str) -> str
```

Return the absolute local path for the per-simulation metadata file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | POSIX-style absolute path string. |

---

### `psi4_filename_from_simname`

```python
psi4_filename_from_simname(sim_name: str) -> str
```

Return the bare filename for the psi4 data file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Filename component only (no directory). |

---

### `psi4_filepath_from_simname`

```python
psi4_filepath_from_simname(sim_name: str) -> str
```

Return the absolute local path for the psi4 data file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | POSIX-style absolute path string. |

---

### `psi4_url_from_simname`

```python
psi4_url_from_simname(sim_name: str) -> str
```

Return the remote download URL for the psi4 data file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Full HTTP(S) URL from the SXS file manifest. |

---

### `download_psi4_data`

```python
download_psi4_data(sim_name: str)
```

Download the psi4 data for *sim_name* via the ``sxs`` package.

Accesses ``sim.psi4`` to trigger the download; the file is cached
in the ``sxs`` package's own cache directory.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

---

### `get`

```python
get(sim_name: str, extrapolation_order: int = 2, download: bool = True) -> waveform.WaveformModes
```

Load the strain waveform for *sim_name* and return a WaveformModes object.

Overrides ``CatalogBase.get()`` because SXS waveform loading goes
entirely through the ``sxs`` package rather than via direct HDF5 reads.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. Deprecated IDs are resolved automatically via ``auto_supersede=True``. |
| `extrapolation_order` | `int` | Order of the polynomial extrapolation to future null infinity. Defaults to 2, which is the SXS default and the only order stored in ``Strain_N2.h5``; orders 3 and 4 are read from ``ExtraWaveforms.h5``. Comparing waveforms across orders bounds the systematic error of the extraction to null infinity, which is a distinct component of the numerical error budget from finite-resolution truncation error -- the v3 catalog publishes a single Lev, so the latter cannot be measured from it. Raises if the requested order is not the one obtained. |
| `download` | `bool` | Whether to download the waveform if not cached. Defaults to True. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `waveform.WaveformModes` | nrcats.waveform.WaveformModes: Waveform object with the |
|  | `waveform.WaveformModes` | catalog metadata attached. |

---

### `available_extrapolation_orders`

```python
available_extrapolation_orders(sim_name: str, download: bool = False) -> list[int]
```

Return the extrapolation orders available for *sim_name*.

Determined from the simulation's **file listing**, which is already
cached with the catalog, so this costs no download and can be called
over a whole batch cheaply.

The v3 layout stores the default order in ``Strain_N{k}.h5`` and every
other order in ``ExtraWaveforms.h5`` under groups ``/Strain_N{k}.dir``.
Group names inside that file cannot be read without fetching it (~21 MB
per simulation), so when ``ExtraWaveforms.h5`` is present the orders it
conventionally carries are reported without opening it.  Pass
``download=True`` to verify against the actual groups instead of the
convention -- worth doing once when surveying a new catalog release,
not per simulation in a loop.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name. |
| `download` | `bool` | Open ``ExtraWaveforms.h5`` and read its real group names rather than assuming the convention. Defaults to False. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `list[int]` | list[int]: Sorted extrapolation orders, e.g. ``[2, 3, 4]``. Only |
|  | `list[int]` | the default order is returned when no ``ExtraWaveforms.h5`` exists. |

> **Note**
> The extrapolation order is **not** the resolution.  The v3 catalog
> publishes a single Lev per simulation (``lev_numbers == [3]``), so
> finite-resolution truncation error cannot be estimated from it;
> varying the order bounds the error of the extraction to null
> infinity, which is a different component of the error budget.

---

### *staticmethod* `clear_cache`

```python
clear_cache(sim_names: list[str] | None = None, dry_run: bool = False) -> dict
```

Delete cached waveform files, freeing disk between processing batches.

A full pass over the catalog needs far more disk than the waveforms are
worth keeping: 637 quasi-circular simulations at ~21 MB of
``ExtraWaveforms.h5`` each is ~13 GB, and that is before strain files.
Processing in chunks and clearing between them turns an impossible run
into a bounded one, at the cost of re-downloading anything needed twice.

**Only per-simulation waveform directories are removed.**  The catalog
index (``simulations_*.bz2``, ``catalog.zip``) is deliberately kept: it
is a few megabytes, it is expensive to rebuild, and every call into the
catalog needs it.  Deleting it would turn a disk-space optimisation into
a repeated multi-minute stall.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_names` | `list[str] or None` | Simulations to clear, e.g. ``["SXS:BBH:1419"]``. Version suffixes are matched automatically, so ``SXS:BBH:1419`` clears ``SXS:BBH:1419v3.0``. ``None`` clears **every** cached simulation. |
| `dry_run` | `bool` | Report what would be freed without deleting. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `dict` | `dict` | ``{"removed": [paths], "bytes_freed": int, "dry_run": bool}``. |

> **Note**
> Cached data is a pure download cache -- everything removed can be
> fetched again -- so this is safe to call between chunks.  It is not
> safe to call *while* another process is reading the same cache.

---

### `download_waveform_data`

```python
download_waveform_data(sim_name: str) -> object
```

Download the strain waveform data for *sim_name* via ``sxs.load()``.

First fetches the simulation metadata JSON, then downloads the
``rhOverM`` strain file.  Both are handled by the ``sxs`` package's
download and caching machinery.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | SXS simulation name, e.g. ``"SXS:BBH:0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | sxs.WaveformModes: Raw waveform object returned by ``sxs.load()``. |

---

### `waveform_url_from_simname`

```python
waveform_url_from_simname(_sim_name: str) -> str
```

Not implemented for SXS; downloads are managed by ``sxs.load()``.

#### Raises

| Exception | Condition |
|---|---|
| `NotImplementedError` | Always. |

---

### `metadata_url_from_simname`

```python
metadata_url_from_simname(_sim_name: str) -> str
```

Not implemented for SXS; use ``sxs.load()`` instead.

#### Raises

| Exception | Condition |
|---|---|
| `NotImplementedError` | Always. |

---

### `to_sxs`

```python
to_sxs() -> 'sxs.Simulations'
```

Return the live ``sxs.Simulations`` object backing this catalog.

---

### *property* `simulations_dataframe`

Return the sxs.SimulationsDataFrame for this catalog.

---

### *property* `table`

Alias for simulations_dataframe.

---

### *property* `tag`

Return the git tag of the catalog snapshot.

---

### *property* `published_at`

Return the ISO timestamp from GitHub Releases.

---

### *property* `modified`

Approximate the modified property using published_at.

---

{% endraw %}
