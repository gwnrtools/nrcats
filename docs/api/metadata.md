---
title: nrcats.metadata
parent: API Reference
nav_order: 13
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.metadata`

Cross-catalog metadata key mappings and source-parameter extraction.

This module bridges the three catalog-specific metadata schemas (RIT, SXS,
MAYA) and the PyCBC parameter convention used downstream by LVK analysis
pipelines.

Module-level constants
----------------------
RIT_KEYS : dict
    Canonical quantity name → RIT metadata key (hyphenated).
SXS_KEYS : dict
    Canonical quantity name → SXS metadata key (snake_case).  Spin
    components are stored as 3-element list vectors; per-component keys map
    to ``None`` to indicate vector access.
MAYA_KEYS : dict
    Canonical quantity name → MAYA/GT metadata key.
CANONICAL_TO_CATALOG : dict
    Unified lookup: ``canonical_name → {"RIT": key, "SXS": key, "MAYA": key}``.
CANONICAL_TO_PYCBC : dict
    Maps canonical names to their PyCBC output parameter names.
PYCBC_KEYS : dict
    Identity mapping of PyCBC parameter names; documents which keys are
    output by ``get_source_parameters_from_metadata()``.

Public functions
----------------
get_source_parameters_from_metadata(metadata, total_mass)
    Convert a raw catalog metadata dict (with injected ``catalog_type`` key)
    into a PyCBC-compatible binary parameter dict.


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

### `RIT_KEYS`

Mapping from canonical quantity names to RIT metadata keys (hyphenated).

In the `simulations_dataframe` these keys appear as-is (with hyphens).
`get_source_parameters_from_metadata()` accesses them with underscores after
`parse_metadata_txt()` converts hyphens to underscores during DataFrame
construction.

---

### `SXS_KEYS`

Mapping from canonical quantity names to SXS metadata keys (snake_case).

Spin components are stored as 3-element lists under ``spin1_vector`` /
``spin2_vector``; the individual ``spin1x`` etc. entries are ``None`` to
signal that they must be accessed by index::

    spin1 = metadata[SXS_KEYS["spin1_vector"]]   # [chi_x, chi_y, chi_z]
    chi1x = spin1[0]

---

### `MAYA_KEYS`

Mapping from canonical quantity names to MAYA/GT metadata keys.

Note that MAYA does not record a dedicated relaxation time, initial ADM
quantities, or a separate reference epoch distinct from the simulation start.

---

### `CANONICAL_TO_CATALOG`

Dict mapping each canonical quantity name to its key in every catalog.

Example::

    >>> from nrcats.metadata import CANONICAL_TO_CATALOG
    >>> CANONICAL_TO_CATALOG["mass_ratio"]
    {'RIT': 'relaxed-mass-ratio-1-over-2', 'SXS': 'reference_mass_ratio', 'MAYA': 'q'}
    >>> CANONICAL_TO_CATALOG["spin1x"]
    {'RIT': 'relaxed-chi1x', 'SXS': None, 'MAYA': 'a1x'}

A value of ``None`` means the quantity is not stored as a scalar key in that
catalog (see the per-catalog dict docstring for the access pattern).

---

### `PYCBC_KEYS`

PyCBC-compatible parameter names output by ``get_source_parameters_from_metadata()``.

These are the keys accepted by ``pycbc.waveform.get_td_waveform_modes()`` and
related functions.  All catalog-specific keys are normalised to these names by
``get_source_parameters_from_metadata()``.

---

### `CANONICAL_TO_PYCBC`

Maps canonical quantity names to their PyCBC output parameter name.

Quantities absent from this dict are not directly exposed as PyCBC parameters
(e.g. remnant properties, ADM quantities, numerical method flags).

---

### `get_source_parameters_from_metadata`

```python
get_source_parameters_from_metadata(metadata: dict, total_mass: float = 1.0) -> dict
```

Return the initial physical parameters for the simulation. Only for
quasicircular simulations are supported, orbital eccentricity is ignored

#### Parameters

| Name | Type | Description |
|---|---|---|
| `metadata` | `dict` | Simulation metadata dict. Must contain a ``"catalog_type"`` key with value ``"RIT"``, ``"SXS"``, or ``"MAYA"``. This key is injected automatically by ``CatalogBase.get_metadata()``. |
| `total_mass` | `float` | Total Mass of Binary (solar masses). Defaults to 1.0. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `dict` | `dict` | Initial binary parameters with names compatible with PyCBC. |

#### Raises

| Exception | Condition |
|---|---|
| `ValueError` | If ``catalog_type`` is absent or not one of the known values. |

---

{% endraw %}
