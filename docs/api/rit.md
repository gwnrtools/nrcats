---
title: nrcats.rit
parent: API Reference
nav_order: 3
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.rit`

RIT catalog interface.

Provides access to the Rochester Institute of Technology (RIT) catalog of
numerical-relativity BBH waveforms generated with the LazEv code.

Metadata is scraped from the RIT web server
(``https://ccrgpages.rit.edu/~RITCatalog/Metadata/``) as individual
``*_Metadata.txt`` files, aggregated into a Pandas DataFrame, and cached
at ``~/.cache/RIT/metadata/metadata.csv``.  Waveform (HDF5) and psi4
(tar.gz) data are downloaded on demand.

Both quasicircular BBH (``RIT:BBH:NNNN-nRRR-idI``) and eccentric BBH
(``RIT:eBBH:NNNN-nRRR-ecc``) simulation names are supported.

Key design decisions
--------------------
* **Singleton pattern** – ``RITCatalog.load()`` stores its result in a
  module-level singleton to avoid the ``lru_cache`` stale-result bug
  (keyed on all arguments, a ``load(download=True)`` call after an earlier
  ``load(download=False)`` would have returned the stale cached result).

* **Two-class design** – ``RITCatalog`` exposes the public
  ``CatalogBase`` interface and delegates all scraping/caching/file-naming
  logic to ``RITCatalogHelper``, which can be instantiated and tested
  independently.

Public classes
--------------
RITCatalog
    Registered under the tag ``"RIT"`` in the catalog plugin registry.
RITCatalogHelper
    Internal helper; handles metadata scraping, caching, and filename
    conventions.  Not part of the public API but documented here because
    it is the main complexity in this module.


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

## *class* `RITCatalog`

Bases: `catalog.CatalogBase`

Catalog interface for the RIT (LazEv) NR waveform collection.

Delegates all file-naming, URL-construction, caching, and web-crawling
logic to ``RITCatalogHelper``.  ``RITCatalog`` itself provides the
``CatalogBase``-compatible public API (``load()``, ``get()``,
``get_metadata()``, ``get_parameters()``, etc.).

Key design points:

- Metadata is scraped from ``https://ccrgpages.rit.edu/~RITCatalog/``
  on first load and aggregated into ``~/.cache/RIT/metadata/metadata.csv``.
- Waveform HDF5 files (``ExtrapStrain_RIT-BBH-*.h5``) are downloaded
  to ``~/.cache/RIT/data/`` on demand.
- Psi4 data is available as ``.tar.gz`` archives via
  ``download_psi4_data()`` / ``psi4_filepath_from_simname()``.
- A module-level singleton prevents redundant catalog loads when
  ``load()`` is called multiple times in the same process.

Example:
    >>> import nrcats as nrcat
    >>> cat = nrcat.RITCatalog.load(verbosity=0)
    >>> wfm = cat.get("RIT:BBH:0001-n100-id3")


### *classmethod* `load`

```python
load(download: bool | None = None, num_sims_to_crawl: int = 2000, acceptable_scraping_fraction: float = 0.7, verbosity: int = 0) -> RITCatalog
```

Load the RIT catalog.

The result is cached in a module-level singleton after the first
successful load.  Subsequent calls return the cached instance without
re-reading disk or the network, unless ``download=True`` is passed.

Pass ``download=True`` to force a fresh download and replace the
singleton, or call ``RITCatalog.reload()`` for the same effect.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `download` | `(None, bool)` | If False, this function will look for the catalog in the cache and raise an error if it is not found. If True, this function will download the catalog and raise an error if the download fails. If None (the default), it will try to download the file, warn but fall back to the cache if that fails, and only raise an error if the catalog is not found in the cache. |

> **See Also**
> RITCatalog.reload : Force a fresh download and replace the singleton.
> nrcats.utils.rit_catalog_info : Catalog info, including cache directory.

---

### *classmethod* `reload`

```python
reload(**kwargs) -> RITCatalog
```

Force a fresh download and replace the cached singleton.

Equivalent to ``RITCatalog.load(download=True, **kwargs)``.

---

### *property* `simulations_dataframe`

All simulations as a Pandas DataFrame indexed by simulation name.

Removes any unnamed index columns left over from CSV round-trips and
sets ``simulation_name`` as both the index and an explicit column.

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | pandas.DataFrame: DataFrame with one row per simulation and |
|  | `object` | metadata fields as columns. |

---

### *property* `files`

Map of waveform and psi4 filenames to file-info dicts.

Each value is a dict with keys: ``checksum`` (None), ``filename``,
``filesize`` (bytes; 0 if not cached), ``download`` (remote URL), and
``truepath`` (canonical local filename after deduplication).

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `dict` | dict[str, dict]: Mapping from bare filename to file-info dict. |

---

### `metadata_filename_from_simname`

```python
metadata_filename_from_simname(sim_name: str) -> str
```

Return the bare filename of the RIT metadata text file for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag, e.g. ``"RIT:BBH:0001-n100-id3"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Filename string, e.g. ``"RIT:BBH:0001-n100-id3_Metadata.txt"``. |

---

### `metadata_filepath_from_simname`

```python
metadata_filepath_from_simname(sim_name: str) -> str
```

Return the absolute local path to the metadata file for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag, e.g. ``"RIT:BBH:0001-n100-id3"``. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Absolute path string to the cached ``.txt`` metadata file. |

#### Raises

| Exception | Condition |
|---|---|
| `RuntimeError` | If the path stored in the metadata dict does not exist on disk. |

---

### `metadata_url_from_simname`

```python
metadata_url_from_simname(sim_name: str) -> str
```

Return the remote URL for the metadata text file for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Full URL string on the RIT web server. |

---

### `waveform_filename_from_simname`

```python
waveform_filename_from_simname(sim_name: str) -> str
```

Return the bare filename of the RIT waveform HDF5 file for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Filename string, e.g. ``"ExtrapStrain_RIT-BBH-0001-n100.h5"``. |

---

### `waveform_filepath_from_simname`

```python
waveform_filepath_from_simname(sim_name: str) -> str
```

Return the absolute local cache path to the waveform HDF5 for *sim_name*.

Falls back to re-anchoring the filename under the current cache directory
when the stored path (from a shared CSV) belongs to a different machine.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Absolute path string to the local ``.h5`` waveform file. Returns the |
|  | `str` | stored (possibly stale) path if the file is not yet downloaded. |

---

### `waveform_url_from_simname`

```python
waveform_url_from_simname(sim_name: str) -> str
```

Return the remote URL of the waveform HDF5 file for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Full URL string on the RIT web server. |

---

### `refresh_metadata_df_on_disk`

```python
refresh_metadata_df_on_disk(num_sims_to_crawl: int = 2000) -> object
```

Delegate to ``RITCatalogHelper.refresh_metadata_df_on_disk()``.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `num_sims_to_crawl` | `int` | Upper bound on the simulation index. Defaults to 2000. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | pandas.DataFrame: Refreshed aggregated metadata DataFrame. |

---

### `download_data_for_catalog`

```python
download_data_for_catalog(num_sims_to_crawl: int = 2000, which_data: str = 'waveform', possible_res: list | None = None, max_id_in_name: int = -1, use_cache: bool = True) -> dict
```

Download waveform or psi4 data for all simulations in the catalog.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `num_sims_to_crawl` | `int` | Maximum number of simulations to process. Defaults to 2000. |
| `which_data` | `str` | ``"waveform"`` or ``"psi4"``. Defaults to ``"waveform"``. |
| `possible_res` | `list[int] or None` | Resolution values to try. Defaults to the list in ``utils.rit_catalog_info``. |
| `max_id_in_name` | `int` | Maximum ID suffix to search for. Defaults to ``-1`` (uses the value in ``utils.rit_catalog_info``). |
| `use_cache` | `bool` | Skip download if a non-empty file exists locally. Defaults to True. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `dict` | dict[str, pathlib.Path]: Mapping from simulation name to the |
|  | `dict` | local file path for each successfully downloaded file. |

---

### `write_metadata_df_to_disk`

```python
write_metadata_df_to_disk()
```

Write the current aggregated metadata DataFrame to ``metadata.csv``.

Delegates to ``RITCatalogHelper.write_metadata_df_to_disk()``.

---

### `download_waveform_data`

```python
download_waveform_data(sim_name: str, use_cache: bool | None = None) -> bool
```

Download the waveform HDF5 file for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag, e.g. ``"RIT:BBH:0001-n100-id3"``. |
| `use_cache` | `bool or None` | Use the cached file if present. Defaults to None (uses helper's default). |

#### Returns

| Name | Type | Description |
|---|---|---|
| `bool` | `bool` | True if the file is available locally after the call. |

---

### `psi4_filename_from_simname`

```python
psi4_filename_from_simname(sim_name: str) -> str
```

Return the bare filename of the RIT psi4 tar.gz archive for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Filename string, e.g. |
|  | `str` | ``"ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz"``. |

---

### `psi4_filepath_from_simname`

```python
psi4_filepath_from_simname(sim_name: str) -> str
```

Return the absolute local path to the psi4 archive for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Absolute path string, or an empty string if the file is not yet |
|  | `str` | downloaded. |

---

### `psi4_url_from_simname`

```python
psi4_url_from_simname(sim_name: str) -> str
```

Return the remote URL of the psi4 archive for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Full URL string on the RIT web server. |

---

### `download_psi4_data`

```python
download_psi4_data(sim_name: str, use_cache: bool | None = None) -> bool
```

Download the psi4 tar.gz archive for *sim_name*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag. |
| `use_cache` | `bool | None` | Skip download if a non-empty file exists locally. Defaults to None (uses helper's default). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `bool` | True if the file is available locally after the call. |

---

## *class* `RITCatalogHelper`

Bases: `object`

Internal helper for RIT catalog scraping, caching, and file naming.

Handles all the catalog-specific complexity that ``RITCatalog`` delegates:

- **File naming** – converts simulation name tags (e.g.
  ``"RIT:BBH:0001-n100-id3"``) to metadata filenames, waveform filenames,
  and psi4 filenames in the formats used on the RIT web server.
- **Metadata scraping** – downloads ``*_Metadata.txt`` files from the RIT
  web server one-by-one, parses them with ``parse_metadata_txt()``, and
  aggregates them into a Pandas DataFrame.
- **Disk caching** – reads/writes the aggregated DataFrame as
  ``~/.cache/RIT/metadata/metadata.csv`` and stores individual
  ``*_Metadata.txt`` files in ``~/.cache/RIT/metadata/``.
- **Data downloads** – downloads waveform HDF5 and psi4 tar.gz files via
  ``wget`` into ``~/.cache/RIT/data/``.

This class is not part of the public API; use ``RITCatalog`` instead.


### `metadata_filenames`

```python
metadata_filenames(idx: int, res: int, id_val: int) -> list
```

Return all candidate metadata filenames for simulation *idx*.

Returns one name for the quasicircular BBH format and one for the
eccentric BBH format, since a given index may correspond to either.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index (e.g. ``1`` for ``0001``). |
| `res` | `int` | Numerical resolution tag (e.g. ``100``). |
| `id_val` | `int` | ID suffix for quasicircular simulations (e.g. ``3``). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `list` | list[str]: Two candidate filenames: |
|  | `list` | ``["RIT:BBH:NNNN-nRRR-idI_Metadata.txt", |
|  | `list` | "RIT:eBBH:NNNN-nRRR-ecc_Metadata.txt"]``. |

---

### `sim_info_from_metadata_filename`

```python
sim_info_from_metadata_filename(file_name: str) -> tuple
```

Input:
------
file_name: name (not path) of metadata file as hosted on the web

Output:
-------
- simulation number
- resolution as indicated with an integer
- ID value (only for non-eccentric simulations)

---

### `simname_from_metadata_filename`

```python
simname_from_metadata_filename(filename: str) -> str
```

Input:
------
- filename: name (not path) of metadata file as hosted on the web

Output:
-------
- Simulation Name Tag (Class uses this tag for internal indexing)

---

### `metadata_filename_from_simname`

```python
metadata_filename_from_simname(sim_name: str) -> str
```

We assume the sim names are either of the format:
(1) RIT:eBBH:1109-n100-ecc
(2) RIT:BBH:1109-n100-id1

---

### `metadata_filename_from_cache`

```python
metadata_filename_from_cache(idx: int) -> str
```

Return the path of the cached metadata file for simulation *idx*.

Searches the local metadata cache directory for any file whose name
starts with either the BBH or eBBH sim-tag prefix for this index.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Full path of the first matching cached file, or ``""`` if no |
|  | `str` | cached file is found. |

---

### `psi4_filename_from_simname`

```python
psi4_filename_from_simname(sim_name: str) -> str
```

We assume the sim names are either of the format:
(1) RIT:eBBH:1109-n100-ecc
(2) RIT:BBH:1109-n100-id1

---

### `psi4_filename_from_cache`

```python
psi4_filename_from_cache(idx: int) -> str
```

Return the psi4 filename for simulation *idx* via the cache.

Looks up the cached metadata filename for *idx*, derives the
simulation name from it, and then computes the psi4 filename.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Psi4 filename, e.g. |
|  | `str` | ``"ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz"``. |

---

### `waveform_filename_from_simname`

```python
waveform_filename_from_simname(sim_name: str) -> str
```

ExtrapStrain_RIT-BBH-0005-n100.h5 -->
ExtrapStrain_RIT-eBBH-1843-n100.h5
RIT:eBBH:1843-n100-ecc_Metadata.txt

---

### `waveform_filename_from_cache`

```python
waveform_filename_from_cache(idx: int) -> str
```

Return the waveform HDF5 filename for simulation *idx* via the cache.

Looks up the cached metadata filename for *idx*, derives the
simulation name from it, and then computes the waveform filename.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Waveform filename, e.g. |
|  | `str` | ``"ExtrapStrain_RIT-BBH-0001-n100.h5"``. |

---

### `simname_from_cache`

```python
simname_from_cache(idx: int) -> str
```

Return the simulation name tag for *idx* by inspecting the cache.

Searches the metadata cache directory for a file matching either the
BBH or eBBH prefix, then derives the simulation name from the
filename.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `str` | `str` | Simulation name tag (e.g. ``"RIT:BBH:0001-n100-id3"``), or |
|  | `str` | ``""`` if no cached metadata file is found for *idx*. |

---

### `simnames`

```python
simnames(idx: int, res: int, id_val: int) -> list
```

Return candidate simulation name tags for *idx* at *res* and *id_val*.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |
| `res` | `int` | Numerical resolution tag. |
| `id_val` | `int` | ID suffix for quasicircular simulations. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `list` | list[str]: Two candidate simulation names (BBH and eBBH formats). |

---

### `simtags`

```python
simtags(idx: int) -> list
```

Return the filename-prefix tags used to glob-search for *idx*.

Returns one prefix for the BBH format and one for the eBBH format,
which are used to search for cached metadata files with
``glob(prefix + "*")``.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `list` | list[str]: Two prefix strings, e.g. |
|  | `list` | ``["RIT:BBH:0001", "RIT:eBBH:0001"]``. |

---

### `parse_metadata_txt`

```python
parse_metadata_txt(raw: list) -> tuple
```

Parses raw RIT metadata

#### Parameters

| Name | Type | Description |
|---|---|---|
| `raw` | `list(str)` | List of lines read in from RIT metadata |

#### Returns

| Name | Type | Description |
|---|---|---|
| `list` | `str` | Original metadata with empty lines removed |
| `dict` | `tuple` | Parsed metadata as a dictionary |

---

### `metadata_from_link`

```python
metadata_from_link(link: str, save_to: object = None, num_retries: int = 5) -> tuple
```

Fetch and parse a single RIT metadata file from a URL.

If *save_to* is given, downloads the file to disk and then parses it
with ``metadata_from_file()``.  Otherwise performs an in-memory HTTP
GET and parses the response body directly.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `link` | `str` | Full HTTP(S) URL to the ``*_Metadata.txt`` file. |
| `save_to` | `str or pathlib.Path or None` | If provided, save the downloaded text to this path before parsing. Defaults to None. |
| `num_retries` | `int` | Number of request attempts with exponential backoff. Defaults to 5. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `tuple` | tuple[list[str], dict]: The output of ``parse_metadata_txt()``: |
|  | `tuple` | a list of non-empty metadata lines and a dict of parsed fields. |

#### Raises

| Exception | Condition |
|---|---|
| `ConnectionError` | If all retry attempts fail. |

---

### `metadata_from_file`

```python
metadata_from_file(file_path: object) -> tuple
```

Parse a locally cached RIT metadata text file.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `file_path` | `str or pathlib.Path` | Path to the ``*_Metadata.txt`` file on disk. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `tuple` | tuple[list[str], dict]: The output of ``parse_metadata_txt()``: |
|  | `tuple` | a list of non-empty metadata lines and a dict of parsed fields. |

---

### `metadata_from_cache`

```python
metadata_from_cache(idx: int) -> object
```

Build a single-row DataFrame from a cached metadata file for *idx*.

Searches the local metadata cache for any file matching the BBH or
eBBH prefix for *idx*.  If found, parses it and enriches the result
with the simulation name, metadata/psi4/waveform URLs and local paths.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | pandas.DataFrame: Single-row DataFrame with all metadata fields |
|  | `object` | plus ``simulation_name``, ``metadata_link``, ``metadata_location``, |
|  | `object` | ``psi4_data_link``, ``psi4_data_location``, ``waveform_data_link``, |
|  | `object` | and ``waveform_data_location`` columns. Returns an empty DataFrame |
|  | `object` | if no cached file is found for *idx*. |

---

### `download_metadata`

```python
download_metadata(idx: int, res: int, id_val: int = -1) -> object
```

Download (or read from cache) metadata for one simulation.

Tries the BBH filename format first, then the eBBH format.  For each
candidate filename, checks the local cache first (if ``use_cache`` is
True) before making an HTTP request.  The downloaded file is saved
to the metadata cache directory.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |
| `res` | `int` | Numerical resolution tag (e.g. ``100``). |
| `id_val` | `int` | ID suffix for quasicircular simulations. Defaults to ``-1`` (triggers the eccentric filename format as fallback). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | pandas.DataFrame: Single-row DataFrame (same schema as |
|  | `object` | ``metadata_from_cache()``), or an empty DataFrame if neither |
|  | `object` | filename is found locally or remotely. |

---

### `download_metadata_for_catalog`

```python
download_metadata_for_catalog(num_sims_to_crawl: int = 2000, possible_res: list = [], max_id_in_name: int = -1) -> object
```

We crawl the webdirectory where RIT metadata usually lives,
and try to read metadata for as many simulations as we can

---

### `write_metadata_df_to_disk`

```python
write_metadata_df_to_disk()
```

Write the current ``self.metadata`` DataFrame to ``metadata.csv``.

Saves to ``~/.cache/RIT/metadata/metadata.csv`` (or the path
configured via ``NR_CATALOG_CACHE``).  Called automatically after
each simulation's metadata is downloaded during a catalog crawl.

---

### `refresh_metadata_df_on_disk`

```python
refresh_metadata_df_on_disk(num_sims_to_crawl: int = 2000) -> object
```

Rebuild the metadata CSV from cached ``*_Metadata.txt`` files.

Iterates over simulation indices 1 … *num_sims_to_crawl*, reads each
simulation's metadata from the local file cache (does **not** make
network requests), concatenates the results, and writes the aggregated
DataFrame to ``metadata.csv``.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `num_sims_to_crawl` | `int` | Upper bound on the simulation index to scan. Defaults to 2000. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | pandas.DataFrame: The refreshed aggregated metadata DataFrame. |

---

### `read_metadata_df_from_disk`

```python
read_metadata_df_from_disk() -> object
```

Load the aggregated metadata DataFrame from ``metadata.csv``.

If the CSV file does not exist or is empty, sets ``self.metadata`` to
an empty DataFrame and returns it.

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | pandas.DataFrame: The previously saved aggregated metadata, or an |
|  | `object` | empty DataFrame if the cache file is absent. |

---

### `download_psi4_data`

```python
download_psi4_data(sim_name: str, use_cache: bool | None = True) -> bool
```

Download the psi4 tar.gz file for *sim_name* via ``wget``.

Skips the download if ``use_cache`` is True and a non-empty local
file already exists.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag, e.g. ``"RIT:BBH:0001-n100-id3"``. |
| `use_cache` | `bool or None` | Use cached file if present. If ``None``, falls back to the instance-level ``self.use_cache``. Defaults to True. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `bool` | `bool` | True if the file is available locally (either from cache or |
|  | `bool` | after a successful download), False if the URL was not found. |

---

### `download_waveform_data`

```python
download_waveform_data(sim_name: str, use_cache: bool | None = True) -> bool
```

Download the waveform HDF5 file for *sim_name* via ``wget``.

Skips the download if ``use_cache`` is True and a non-empty local
file already exists.

Possible file formats:

- ``https://ccrgpages.rit.edu/~RITCatalog/Data/ExtrapStrain_RIT-BBH-0193-n100.h5``
- ``https://ccrgpages.rit.edu/~RITCatalog/Data/ExtrapStrain_RIT-eBBH-1911-n100.h5``

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_name` | `str` | RIT simulation name tag, e.g. ``"RIT:BBH:0001-n100-id3"``. |
| `use_cache` | `bool or None` | Use cached file if present. If ``None``, falls back to the instance-level ``self.use_cache``. Defaults to True. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `bool` | `bool` | True if the file is available locally (either from cache or |
|  | `bool` | after a successful download), False if the URL was not found. |

---

### `fetch_waveform_data_from_cache`

```python
fetch_waveform_data_from_cache(idx: int) -> object
```

Not yet implemented.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `idx` | `int` | Four-digit simulation index. |

#### Raises

| Exception | Condition |
|---|---|
| `NotImplementedError` | Always. |

---

### `download_data_for_catalog`

```python
download_data_for_catalog(num_sims_to_crawl: int = 2000, which_data: str = 'waveform', possible_res: list = [], max_id_in_name: int = -1, use_cache: bool = True) -> dict
```

Download waveform or psi4 data for all simulations in the catalog.

Crawls the RIT web directory for waveform or psi4 data files and
downloads each one.  Refreshes the on-disk metadata DataFrame first
if it is out of date.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `num_sims_to_crawl` | `int` | Maximum number of simulations to process. Defaults to 2000. |
| `which_data` | `str` | ``"waveform"`` or ``"psi4"``. Defaults to ``"waveform"``. |
| `possible_res` | `list` | Resolution values to try. Defaults to the list in ``utils.rit_catalog_info``. |
| `max_id_in_name` | `int` | Maximum ID suffix to search. Defaults to ``-1`` (uses the value in ``utils.rit_catalog_info``). |
| `use_cache` | `bool` | Skip download if a non-empty file exists locally. Defaults to True. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `dict` | dict[str, pathlib.Path]: Mapping from simulation name to the |
|  | `dict` | local file path for each successfully downloaded file. |

---

{% endraw %}
