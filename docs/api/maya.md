---
title: nrcats.maya
parent: API Reference
nav_order: 5
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.maya`

MAYA/GT catalog interface.

Wraps the Georgia Tech MAYA catalog of numerical-relativity BBH waveforms.
Metadata is distributed as a pickled Pandas DataFrame
(``MAYAmetadata.pkl``) downloaded from the UT Austin CGP storage server
and cached locally as a bzip2-compressed ZIP archive.  Waveform data files
are individual HDF5 files downloaded on demand.

Key design decisions
--------------------
* **Singleton pattern** – ``load()`` stores its result in
  ``_maya_catalog_singleton`` so that repeated calls reuse the parsed
  metadata.  Pass ``download=True`` or call ``reload()`` to force a fresh
  download.

* **LVCNR conversion** – ``download_waveform_data()`` can optionally
  convert MAYA-format HDF5 files to LVCNR catalog format via the
  ``mayawaves`` package before caching.

* **PSI4 not available** – the MAYA catalog only distributes strain
  waveforms; all ``psi4_*`` methods raise ``NotImplementedError``.

Public classes
--------------
MayaCatalog
    Registered under the tag ``"MAYA"`` in the catalog plugin registry.


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

## *class* `MayaCatalog`

Bases: `catalog.CatalogBase`

Catalog interface for the MAYA / Georgia-Tech NR waveform collection.

Downloads a pickle-format metadata file (``MAYAmetadata.pkl``) from
the UT Austin CGP storage and uses the ``mayawaves`` package to load
individual waveform HDF5 files.  Key design points:

- Metadata is cached as ``~/.cache/MAYA/data/catalog.zip`` on first
  load.  Subsequent calls read from cache unless ``download=True``.
- Waveform files are loaded via ``mayawaves.Coalescence`` and then
  wrapped in a ``WaveformModes`` object.
- Psi4 is not available for this catalog; the corresponding methods
  raise ``NotImplementedError``.
- A module-level singleton prevents redundant catalog loads when
  ``load()`` is called multiple times in the same process.

Example:
    >>> import nrcats as nrcat
    >>> cat = nrcat.MayaCatalog.load()
    >>> wfm = cat.get("GT0001")


### `clear_cache`

```python
clear_cache()
```

Remove the cached catalog ZIP file so the next ``load()`` re-downloads it.

The file removed is ``~/.cache/MAYA/catalog.zip`` (or the path
configured via ``NR_CATALOG_CACHE``).  The module-level singleton is
*not* cleared by this method; call ``MayaCatalog.reload()`` to force a
full re-download and replace the singleton.

---

### *classmethod* `load`

```python
load(download: bool | None = None, verbosity: int = 0, show_progress: bool = True) -> MayaCatalog
```

Load the MAYA catalog.

Downloads the pickled metadata from the UT Austin CGP storage server,
compresses it to a local bzip2 ZIP cache, and parses it into a
``MayaCatalog`` singleton.  Subsequent calls return the singleton
without re-parsing, unless ``download=True``.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `download` | `bool or None` | If ``False``, only the local cache is used and an error is raised if it is absent. If ``True``, a fresh download is always attempted. If ``None`` (default), a download is attempted and the cache is used as a fallback. |
| `verbosity` | `int` | Verbosity level (0 = quiet). Defaults to 0. |
| `show_progress` | `bool` | Show a download progress bar. Defaults to True. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `MayaCatalog` | `MayaCatalog` | The loaded (possibly cached) catalog instance. |

#### Raises

| Exception | Condition |
|---|---|
| `RuntimeError` | If ``download=True`` and the download fails. |
| `ValueError` | If the catalog cache file is missing or corrupt. |

---

### *classmethod* `reload`

```python
reload(**kwargs) -> MayaCatalog
```

Force a fresh download and replace the cached singleton.

Equivalent to ``MayaCatalog.load(download=True, **kwargs)``.

---

### *property* `simulations_dataframe`

All simulations as a Pandas DataFrame indexed by simulation name.

Columns correspond to the MAYA metadata fields (mass ratio, spins,
orbital frequency, etc.) plus the path/link columns added by
``_add_paths_to_metadata()``.

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | pandas.DataFrame: DataFrame with one row per simulation. |

---

### *property* `files`

Map of waveform filenames to file-info dicts.

Each value is a dict with keys:
``checksum`` (None), ``filename``, ``filesize`` (bytes, 0 if not
cached), ``download`` (remote URL), and ``truepath`` (canonical
local filename after deduplication).

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `dict` | dict[str, dict]: Mapping from bare filename to file-info dict. |

---

### `metadata_filename_from_simname`

```python
metadata_filename_from_simname(sim_name: str) -> str
```

Return the bare filename for the per-simulation metadata file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | MAYA simulation name, e.g. ``"GT0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Filename, e.g. ``"GT0001.txt"``. |

---

### `metadata_filepath_from_simname`

```python
metadata_filepath_from_simname(sim_name: str, ext: str = 'txt') -> str
```

Return the absolute local path for the per-simulation metadata file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | MAYA simulation name, e.g. ``"GT0001"``. |
| `ext` | `str` | File extension. Defaults to ``"txt"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Absolute path under ``~/.cache/MAYA/metadata/``. |

---

### `metadata_url_from_simname`

```python
metadata_url_from_simname(_sim_name: str)
```

MAYA does not expose per-simulation metadata URLs; returns None.

The parameter is accepted for interface compatibility with
``CatalogABC`` but is not used.

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `None` | None |

---

### `waveform_filename_from_simname`

```python
waveform_filename_from_simname(sim_name: str) -> str
```

Return the bare HDF5 filename for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | MAYA simulation name, e.g. ``"GT0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Filename, e.g. ``"GT0001.h5"``. |

---

### `waveform_filepath_from_simname`

```python
waveform_filepath_from_simname(sim_name: str) -> str
```

Return the absolute local path for the waveform HDF5 file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | MAYA simulation name, e.g. ``"GT0001"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | POSIX-style absolute path under ``~/.cache/MAYA/data/``. |

---

### `waveform_url_from_simname`

```python
waveform_url_from_simname(sim_name: str, maya_format: bool = False) -> str
```

Return the remote URL for the waveform HDF5 file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | MAYA simulation name, e.g. ``"GT0001"``. |
| `maya_format` | `bool` | If True, return the URL for the native MAYA format file instead of the LVCNR-format file. Defaults to False. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Full HTTP(S) URL. |

---

### `download_waveform_data`

```python
download_waveform_data(sim_name: str, maya_format: bool = False, use_cache: bool | None = None)
```

Download the waveform HDF5 file for *sim_name* into the local cache.

By default downloads the LVCNR-format file.  If ``maya_format=True``
the native MAYA format is downloaded and then converted to LVCNR
format using the ``mayawaves`` package before the original is removed.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | MAYA simulation name, e.g. ``"GT0001"``. |
| `maya_format` | `bool` | Download in native MAYA format and convert to LVCNR. Requires ``mayawaves`` to be installed. Defaults to False. |
| `use_cache` | `bool or None` | Whether to skip download if a non-empty local file already exists. If ``None``, falls back to the instance-level ``self.use_cache`` setting. |

---

### `psi4_filename_from_simname`

```python
psi4_filename_from_simname(_sim_name: str) -> str
```

Not implemented; MAYA distributes strain waveforms only.

#### Raises

| Exception | Condition |
|---|---|
| `NotImplementedError` | Always. Use ``get(sim_name)`` for strain data. |

---

### `psi4_filepath_from_simname`

```python
psi4_filepath_from_simname(_sim_name: str) -> str
```

Not implemented; MAYA distributes strain waveforms only.

#### Raises

| Exception | Condition |
|---|---|
| `NotImplementedError` | Always. Use ``get(sim_name)`` for strain data. |

---

### `psi4_url_from_simname`

```python
psi4_url_from_simname(_sim_name: str) -> str
```

Not implemented; MAYA distributes strain waveforms only.

#### Raises

| Exception | Condition |
|---|---|
| `NotImplementedError` | Always. Use ``get(sim_name)`` for strain data. |

---

### `download_psi4_data`

```python
download_psi4_data(_sim_name: str)
```

Not implemented; MAYA distributes strain waveforms only.

#### Raises

| Exception | Condition |
|---|---|
| `NotImplementedError` | Always. Use ``download_waveform_data()`` instead. |

---

{% endraw %}
