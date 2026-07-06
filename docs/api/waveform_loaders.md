---
title: nrcats.waveform.loaders
parent: API Reference
nav_order: 8
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.waveform.loaders`

Standalone loader functions for WaveformModes.

Each function accepts ``cls`` as its first argument (classmethod pattern)
so the caller in ``modes.py`` can do::

    @classmethod
    def load_from_h5(cls, ...):
        from nrcats.waveform.loaders import load_from_h5 as _impl
        return _impl(cls, ...)

No import of ``WaveformModes`` is needed here, avoiding circular imports.


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

### `load_from_h5`

```python
load_from_h5(cls, file_path_or_open_file, metadata={}, verbosity=0)
```

Load SWSH waveform modes from an HDF5 file (RIT/MAYA catalog format).

#### Parameters

| Name | Type | Description |
|---|---|---|
| `cls` | `type` | The ``WaveformModes`` class (or subclass) to instantiate. |
| `file_path_or_open_file` | `str or h5py.File` | Path to the HDF5 file, or an already-open file object. |
| `metadata` | `dict` | Simulation metadata dict. |
| `verbosity` | `int` | Verbosity level (0 = quiet). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `WaveformModes` |  |

---

### `load_from_targz`

```python
load_from_targz(cls, file_path, metadata={}, verbosity=0)
```

Load SWSH waveform modes from a ``.tar.gz`` archive (RIT psi4 format).

#### Parameters

| Name | Type | Description |
|---|---|---|
| `cls` | `type` | The ``WaveformModes`` class (or subclass) to instantiate. |
| `file_path` | `str` | Path to the ``.tar.gz`` archive. |
| `metadata` | `dict` | Simulation metadata dict. |
| `verbosity` | `int` | Verbosity level (0 = quiet). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `WaveformModes` |  |

---

{% endraw %}
