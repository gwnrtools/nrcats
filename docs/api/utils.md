---
title: nrcats.utils
parent: API Reference
nav_order: 16
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.utils`

Utility constants, cache-path definitions, and download helpers.

Module-level constants
----------------------
nrcatalog_cache_dir : pathlib.Path
    Root cache directory; defaults to ``~/.cache/`` but can be overridden
    by setting the ``NR_CATALOG_CACHE`` environment variable.

rit_catalog_info : dict
    Cache paths, base URLs, filename format strings, and parameter ranges
    for the RIT catalog.

maya_catalog_info : dict
    Cache paths and base URLs for the MAYA/GT catalog.

sxs_catalog_info : dict
    Cache paths and base URLs for the SXS catalog (supplemental; the
    ``sxs`` package manages its own cache internally).

Public functions
----------------
url_exists(link, num_retries, verbosity)
    HEAD-request check with exponential-backoff retries.

download_file(url, path, progress, if_newer, num_retries, verbosity)
    Download a URL to a local path; tries the ``sxs`` downloader first and
    falls back to ``requests``.

call_with_timeout(myfunc, args, kwargs, timeout)
    Run a callable in a subprocess with a hard wall-clock timeout.

time_to_physical(M)
    Convert dimensionless NR time units to seconds.

amp_to_physical(M, D)
    Scale dimensionless NR strain amplitude to SI units.

amplitude_phase_frequency_from_complex_mode(hlm)
    Compute instantaneous amplitude, phase, and frequency from a complex
    PyCBC TimeSeries.


## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Constants

| Name | Value |
|---|---|
| `logger` | `logging.getLogger(__name__)` |
| `nrcatalog_cache_dir` | `pathlib.Path(os.getenv('NR_CATALOG_CACHE')).expanduser().resolve()` |
| `nr_group_tags` | `{}` |
| `rit_catalog_info` | `{}` |
| `maya_catalog_info` | `{'cache_dir': nrcatalog_cache_dir / 'MAYA', 'data_url': 'https://cgpstorage.ph.utexas.edu/', 'metadata_url': 'https://cgpstorage.ph.utexas.edu/MAYAmetadata.pkl'}` |
| `sxs_catalog_info` | `{'cache_dir': nrcatalog_cache_dir / 'SXS', 'data_url': 'https://www.black-holes.org/waveforms/', 'metadata_url': 'https://www.black-holes.org/waveforms/metadata.json'}` |

---

### `url_exists`

```python
url_exists(link: str, num_retries: int = 5, verbosity: int = 0) -> bool
```

Check if a given URL exists on the web.

Retries up to ``num_retries`` times on network errors, with exponential
backoff (``2**attempt`` seconds, capped at 30 s).  A non-OK HTTP status
is returned immediately as ``False`` without retrying (the URL exists but
is not accessible / not found — no point retrying).

#### Parameters

| Name | Type | Description |
|---|---|---|
| `link` | `str` | Complete web URL. |
| `num_retries` | `int` | Maximum number of attempts. Defaults to 5. |
| `verbosity` | `int` | Print retry progress when > 0. Defaults to 0. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `bool` | `bool` | True if the URL returned HTTP 200, False otherwise. |

---

### `download_file`

```python
download_file(url: str, path: str | pathlib.Path, progress: bool = False, if_newer: bool = True, num_retries: int = 5, verbosity: int = 0) -> pathlib.Path
```

Download a file from the given URL to the specified local path.

This function attempts to download the file at `url` and save it to `path`.
It first tries to use the `sxs.utilities.downloads.download_file` utility (if available).
If that fails, it falls back to using the Python `requests` package, with SSL
verification disabled and up to ``num_retries`` attempts with exponential
backoff (``2**attempt`` seconds, capped at 30 s).

#### Parameters

| Name | Type | Description |
|---|---|---|
| `url` | `str` | The URL to download the file from. |
| `path` | `str or pathlib.Path` | The destination path where the file should be saved. |
| `progress` | `bool` | Whether to show a progress bar if supported. |
| `if_newer` | `bool` | Only download if the remote file is newer than the local file. |
| `num_retries` | `int` | Maximum number of fallback attempts. Defaults to 5. |
| `verbosity` | `int` | Print retry progress when > 0. Defaults to 0. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `pathlib.Path` | The path (as a pathlib.Path object) to the downloaded file. |

#### Raises

| Exception | Condition |
|---|---|
| `ConnectionError` | If the file could not be fetched after all retry attempts. |
| `RuntimeError` | If the server returned a non-200 status. |

---

### `call_with_timeout`

```python
call_with_timeout(myfunc: object, args: tuple = (), _kwargs: dict = {}, timeout: float = 5) -> object
```

Call a function with a time limit in a separate process.

Executes the provided function `myfunc` with given positional (`args`) and keyword
arguments (`kwargs`) in a separate process. If the function does not complete
within `timeout` seconds, the process is terminated and an exception is raised.
If the function completes within the timeout, its result is returned.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `myfunc` | `callable` | The function to execute. |
| `args` | `tuple` | Positional arguments to pass to `myfunc`. Defaults to (). |
| `_kwargs` | `dict` | Reserved; keyword arguments are not currently forwarded to the subprocess. Defaults to {}. |
| `timeout` | `int or float` | Maximum allowed execution time in seconds. Defaults to 5. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `object` | The result of `myfunc(*args, **kwargs)` if completed within the timeout. |

#### Raises

| Exception | Condition |
|---|---|
| `Exception` | If the function does not complete within the specified timeout. |

---

### `time_to_physical`

```python
time_to_physical(M: float) -> float
```

Factor to convert time from dimensionless units to SI units

#### Parameters

| Name | Type | Description |
|---|---|---|
| `M` | `float` |  |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `converting factor` |  |

---

### `amp_to_physical`

```python
amp_to_physical(M: float, D: float) -> float
```

Factor to rescale strain to mass M and distance D convert from
dimensionless units to SI units

#### Parameters

| Name | Type | Description |
|---|---|---|
| `M` | `float` |  |
| `D` | `float` |  |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `Scaling factor` |  |

---

### `amplitude_phase_frequency_from_complex_mode`

```python
amplitude_phase_frequency_from_complex_mode(hlm: object) -> tuple
```

Compute amplitude, phase, and instantaneous frequency from a complex mode time series.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `hlm` | `tuple of (real, imag) pycbc.types.TimeSeries, or a single complex pycbc.types.TimeSeries` | Either a tuple of real and imaginary parts of a mode (as PyCBC TimeSeries with matching sample_times), or a single complex-valued PyCBC TimeSeries. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `amp` | `pycbc.types.TimeSeries` | The instantaneous amplitude as a function of time. |
| `phase` | `pycbc.types.TimeSeries` | The instantaneous phase as a function of time. |
| `freq` | `pycbc.types.TimeSeries` | The instantaneous frequency (cycles per unit time) as a function of time. |

---

{% endraw %}
