---
title: API Reference
nav_order: 8
has_children: true
has_toc: false
---

# API Reference

Generated from source docstrings by [`bin/generate_api_docs.py`](https://github.com/gwnrtools/nrcats/blob/master/bin/generate_api_docs.py),
which statically analyses the package with [griffe](https://mkdocstrings.github.io/griffe/)
(no runtime imports, so heavy dependencies are not needed to build the docs).

| Module | Description |
|--------|-------------|
| [catalog](catalog.md) | Abstract base classes `CatalogABC` and `CatalogBase` |
| [rit](rit.md) | `RITCatalog` and `RITCatalogHelper` for the RIT catalog |
| [sxs](sxs.md) | `SXSCatalog` wrapping the `sxs` package |
| [maya](maya.md) | `MayaCatalog` for the Georgia Tech MAYA catalog |
| [waveform.modes](waveform_modes.md) | The central `WaveformModes` object |
| [waveform.matching](waveform_matching.md) | Mode matching, rotation, and PSD helpers |
| [waveform.loaders](waveform_loaders.md) | HDF5 / tar.gz loaders for `WaveformModes` |
| [waveform.units](waveform_units.md) | Waveform-level constants |
| [surrogate](surrogate.md) | NRSur7dq4, NRSur7dq4v2 and NRHJSur3dq8 loading, evaluation, and prior check |
| [comparisons](comparisons.md) | End-to-end NR vs surrogate comparison pipeline |
| [classification](classification.md) | Spin/eccentricity classification of catalog simulations |
| [metadata](metadata.md) | Cross-catalog key mappings and `get_source_parameters_from_metadata` |
| [registry](registry.md) | Catalog plugin registry |
| [lvc](lvc.md) | Frame-rotation helpers and LVCNR format utilities |
| [utils](utils.md) | Cache paths, download helpers, unit conversions |
| [waveform.frame](waveform_frame.md) | BMS frame operations a catalog records, and the coprecessing frame |
| [waveform.detectors](waveform_detectors.md) | Observatory registry: noise curves, bands, PSD resolution |

## Quick navigation

All public classes and functions are also importable directly from the top-level package:

```python
import nrcats as nrcat

# Catalogs
nrcat.RITCatalog
nrcat.SXSCatalog
nrcat.MayaCatalog

# Catalog helpers
nrcat.load_catalog          # load by name tag: "SXS", "RIT", or "MAYA"
nrcat.filter_by_surrogate_prior  # filter sims by NRSur7dq4 validity volume

# Waveform
nrcat.WaveformModes
nrcat.apply_wigner_rotation_to_mode_dict

# Registry
nrcat.register_catalog
nrcat.get_catalog
nrcat.list_catalogs

# Metadata key maps
nrcat.RIT_KEYS
nrcat.SXS_KEYS
nrcat.MAYA_KEYS
nrcat.CANONICAL_TO_CATALOG
nrcat.CANONICAL_TO_PYCBC
nrcat.PYCBC_KEYS
```

### Surrogate and matching (sub-module imports)

```python
from nrcats.surrogate import (
    load_nrsur7dq4,
    generate_surrogate_modes,
    check_surrogate_prior,
    SURROGATE_MODES,
    NR_MODES,
)

from nrcats.waveform.matching import (
    load_psd,
    compute_mode_match,
    compute_phase_diff_per_cycle,
    mode_f_lower,
)

from nrcats.comparisons import compare_sim_vs_surrogate, DELTA_T
```

## Regenerating these pages

The per-module pages in this section are generated artifacts — do not edit them by
hand. To refresh them after changing docstrings:

```bash
python bin/generate_api_docs.py          # rewrite docs/api/*.md
python bin/generate_api_docs.py --check  # CI freshness check
```
