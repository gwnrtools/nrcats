---
title: nrcats.classification
parent: API Reference
nav_order: 12
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.classification`

Dynamic threshold-based classification and organization of numerical relativity catalogs.

Exposes the ``NRCatalogClassifier`` class which categorises simulations in
SXS, RIT, and MAYA catalogs into six mutually exclusive categories based on spin and
eccentricity thresholds, and supports filtering by NRSur7dq4 training simulations.


## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## *class* `NRCatalogClassifier`

Classifies simulations in SXS, RIT, and MAYA catalogs based on spin and eccentricity thresholds.

Provides methods to query simulations belonging to different physical categories dynamically.


### `ecc_threshold_for`

```python
ecc_threshold_for(catalog_name: str) -> float
```

The eccentricity threshold applied to *catalog_name*.

Returns the scalar unchanged when a scalar was configured, so the
uniform case costs nothing and reads the same.

---

### `load_catalog`

```python
load_catalog(catalog_name: str)
```

Lazy load a catalog by name tag.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `catalog_name` | `str` | One of 'SXS', 'RIT', 'MAYA' (case-insensitive). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `CatalogBase subclass` |  |

---

### `load_nrsur_calibration_sims`

```python
load_nrsur_calibration_sims() -> set[str]
```

Load and return the set of SXS simulations used to train/calibrate NRSur7dq4.

Extracts the calibration set from `catalog_organization/sxs_classification.json`.

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `set[str]` | Simulation name tags (e.g. 'SXS:BBH:0001'). |

---

### `classify_simulation`

```python
classify_simulation(catalog_name: str, sim_name: str) -> str
```

Classify a single simulation into one of the six categories.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `catalog_name` | `str` | One of 'SXS', 'RIT', 'MAYA'. |
| `sim_name` | `str` | Name tag of the simulation. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `str` | Category name. |

---

### `classify_all`

```python
classify_all(catalog_name: str)
```

Precompute and cache classifications for all simulations in a catalog.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `catalog_name` | `str` | One of 'SXS', 'RIT', 'MAYA'. |

---

### `get_simulations`

```python
get_simulations(catalog_name: str, category: str, only_nrsur_calibration: bool = False) -> list[str]
```

Get the list of simulation name tags under a given catalog and category.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `catalog_name` | `str` | One of 'SXS', 'RIT', 'MAYA' (case-insensitive). |
| `category` | `str` | One of 'a', 'b', 'c', 'd', 'e', 'f' or their full name values. |
| `only_nrsur_calibration` | `bool` | If True, only returns simulations used to calibrate NRSur7dq4 (SXS only). |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `list[str]` | Simulation name tags matching the category. |

---

{% endraw %}
