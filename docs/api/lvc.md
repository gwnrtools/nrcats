---
title: nrcats.lvc
parent: API Reference
nav_order: 15
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.lvc`

Frame-rotation helpers and LVCNR format utilities.

This module provides the low-level functions needed to:

1. **Read LVCNR HDF5 files** – parse mode data and strain from the LVK
   NR format (``get_modes_from_lvcnr_file``, ``get_strain_from_lvcnr_file``).

2. **Compute the LAL source frame** – derive the NR orbital normal
   (``LNhat``) and binary-separation unit vector (``nhat``) from HDF5
   file attributes, time-series data, or SXS-style simulation metadata
   (``compute_lal_source_frame_from_sxs_metadata``,
   ``compute_lal_source_frame_by_interp``).

3. **Rotate NR waveforms into the LAL frame** – compute the SO(3) rotation
   angles (``theta``, ``psi``, ``alpha``) needed to project the NR
   waveform onto the detector line of sight at a given inclination and
   reference phase (``get_nr_to_lal_rotation_angles``).

4. **Build LAL mode dictionaries** – create LAL parameter dictionaries
   with a specified set of (ell, m) modes for use with LALSimulation
   waveform routines (``get_lal_mode_dictionary``,
   ``get_lal_mode_dictionary_from_lmax``).

Public functions
----------------
get_lal_mode_dictionary(mode_array)
get_lal_mode_dictionary_from_lmax(lmax)
get_modes_from_lvcnr_file(path_to_file, Mtot, distance, srate, lmax, f_low)
get_strain_from_lvcnr_file(path_to_file, Mtot, distance, inclination, phi_ref, srate, mode_array)
check_interp_req(h5_file, metadata, ref_time, avail_ref_time)
get_ref_freq_from_ref_time(h5_file, ref_time)
get_ref_time_from_ref_freq(h5_file, ref_freq)
check_nr_attrs(sim_metadata_object, req_attrs)
get_interp_ref_values_from_h5_file(h5_file, req_ts_attrs, ref_time)
get_ref_vals(sim_metadata_object, req_attrs)
compute_lal_source_frame_from_sxs_metadata(sim_metadata)
compute_lal_source_frame_by_interp(h5_file, req_ts_attrs, t_ref)
normalize_metadata(sim_metadata)
get_ref_time_from_metadata(sim_metadata)
transform_spins_nr_to_lal(nrSpin1, nrSpin2, n_hat, ln_hat)
get_nr_to_lal_rotation_angles(h5_file, sim_metadata, inclination, phi_ref, f_ref, t_ref, tol)


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

### `get_lal_mode_dictionary`

```python
get_lal_mode_dictionary(mode_array: list) -> object
```

Get LALDict with all specified modes.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `mode_array` | `list` |  |

#### Returns

| Name | Type | Description |
|---|---|---|
| `waveform_dictionary` | `LALDict with all modes included` |  |

---

### `get_lal_mode_dictionary_from_lmax`

```python
get_lal_mode_dictionary_from_lmax(lmax: int) -> object
```

Get LALDict with modes derived from `lmax`.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `lmax` | `int` |  |

#### Returns

| Name | Type | Description |
|---|---|---|
| `waveform_dictionary` | `LALDict with all modes upto `lmax` included` |  |

---

### `get_modes_from_lvcnr_file`

```python
get_modes_from_lvcnr_file(path_to_file: str, Mtot: float, distance: float, srate: int, lmax: int = 4, f_low: float | None = None) -> tuple
```

Get individual modes from LVCNR format file.


Parameters
==========
path_to_file: string
    Path to LVCNR file
Mtot: float
    Total mass (in units of MSUN) to scale the waveform to
distance: float
    Luminosity Distance (in units of MPc) to scale the waveform to
srate: int
    Sampling rate for the waveform
lmax: int
    Max value of :math:`\ell` to use
    (Default: 4)
f_low: float
    Value of the low frequency to start waveform generation
    Uses value given from the LVCNR file if `None` is provided
    (Default: None)

Returns
=======
mass_ratio: float
    Mass ratio derived from the LVCNR file
spins_args: list
    List of spins derived from the LVCNR file
eccentricity: float
    Eccentricty derived from the LVCNR file.
    Returns `None` is eccentricity is not a number.
f_low: float
    Low Frequency derived either from the file, or provided
    in the call
f_ref: float
    Reference Frequency derived from the file
modes: dict of pycbc TimeSeries objects
    dict containing all the read in modes

---

### `get_strain_from_lvcnr_file`

```python
get_strain_from_lvcnr_file(path_to_file: str, Mtot: float, distance: float, inclination: float, phi_ref: float, srate: int, mode_array: list | None = None) -> tuple
```

Get full strain from LVCNR format file.


Parameters
==========
path_to_file: string
    Path to LVCNR file
Mtot: float
    Total mass (in units of MSUN) to scale the waveform to
distance: float
    Luminosity Distance (in units of MPc) to scale the waveform to
srate: int
    Sampling rate for the waveform
mode_array: list
    list of modes to be included. `None` means all modes are included.
    (Default:None)

Returns
=======
UNDER CONSTRUCTION

---

### `check_interp_req`

```python
check_interp_req(h5_file: object = None, metadata: dict | None = None, ref_time: float | None = None, avail_ref_time: float | None = None) -> tuple
```

Check if the required reference time is different from
the available reference time in the NR HDF5 file or the
simulation metadata.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h5_file` | `file object` | The waveform h5 file handle. |
| `metadata` | `dict | None` |  |
| `ref_time` | `float` | The use and available nr reference time. |
| `avail_ref_time` | `float` | The use and available nr reference time. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `interp` | `bool` | Whether interpolation across time is required. |
| `avail_ref_time` | `float` | The ref_time available in the NR HDF5 file. |

---

### `get_ref_freq_from_ref_time`

```python
get_ref_freq_from_ref_time(h5_file: object, ref_time: float) -> float
```

Get the reference frequency from reference time

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h5_file` | `file object` | The waveform h5 file handle. |
| `ref_time` | `float` | Reference time. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `f_ref` | `float` | Reference frequency. |

---

### `get_ref_time_from_ref_freq`

```python
get_ref_time_from_ref_freq(h5_file: object, ref_freq: float) -> float
```

Get the reference time from reference frequency

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h5_file` | `file object` | The waveform h5 file handle. |
| `ref_freq` | `float` | The reference frequency. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `fTime` | `float` | Reference time. |

---

### `check_nr_attrs`

```python
check_nr_attrs(sim_metadata_object: object, req_attrs: list = ['LNhatx', 'LNhaty', 'LNhatz', 'nhatx', 'nhaty', 'nhatz']) -> bool
```

Check if the NR h5 file or a simulation metadata dictionary
    contains all the attributes required.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_metadata_object` | `h5 file object, dict` | The NR h5py file handle or simulation metadata. |
| `req_attrs` | `list` | A list of attribute keys. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `present` | `bool` | Whether or not all specified attributes are present. |
| `absent_attrs` | `list` | The attributes that are absent. |

---

### `get_interp_ref_values_from_h5_file`

```python
get_interp_ref_values_from_h5_file(h5_file: object, req_ts_attrs: list, ref_time: float) -> dict
```

Get the interpolated reference values at a given reference time
from the NR HDF5 File

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h5_file` | `file object` | The waveform h5 file handle. |
| `req_ts_attrs` | `list` | A list of attribute keys. |
| `ref_time` | `float` | Reference time. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `params` | `dict` | The parameter values at the reference time. |

---

### `get_ref_vals`

```python
get_ref_vals(sim_metadata_object: object, req_attrs: list = ['LNhatx', 'LNhaty', 'LNhatz', 'nhatx', 'nhaty', 'nhatz']) -> dict
```

Get the reference values from a NR HDF5 file
or a simulation metadata dictionary.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_metadata_object` | `h5 file object, dict` | The NR h5py file handle or the simulation metadata. |
| `req_attrs` | `list` | A list of attribute keys. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `params` | `dict` | The parameter values at the reference time. |

---

### `compute_lal_source_frame_from_sxs_metadata`

```python
compute_lal_source_frame_from_sxs_metadata(sim_metadata: dict) -> tuple
```

Compute the LAL source frame vectors at the
available reference time from the SXS simulation
metadata.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_metadata` | `dict` | The NR sim_metadata. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `params` | `dict` | A dictionary containing the LAL source frame vectors. |

---

### `compute_lal_source_frame_by_interp`

```python
compute_lal_source_frame_by_interp(h5_file: object, req_ts_attrs: list, t_ref: float) -> tuple
```

Compute the LAL source frame vectors at a given reference time
by interpolation of time series data.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h5_file` | `h5 file object` | The NR h5py file handle that contains the simulation metadata. |
| `t_ref` | `float` | The reference time. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `params` | `dict` | The LAL source frame vectors at the reference time. |

---

### `normalize_metadata`

```python
normalize_metadata(sim_metadata: dict) -> dict
```

Ensure that the keys of the metadata are
as required.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_metadata` | `dict` | The NR sim_metadata. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `norm_sim_metadata` | `dict` | The normalized simulation metadata |

---

### `get_ref_time_from_metadata`

```python
get_ref_time_from_metadata(sim_metadata: dict) -> float
```

Get the reference time of definition of the LAL
frame from the simulation metadata, if available.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `sim_metadata` | `dict` | The simulation metadata. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `t_ref` | `float` | The reference time |

---

### `transform_spins_nr_to_lal`

```python
transform_spins_nr_to_lal(nrSpin1: object, nrSpin2: object, n_hat: object, ln_hat: object) -> tuple
```

Trnasform the spins of the NR simulation from the
NR frame to the  frame.

#### Parameters

| Name | Type | Description |
|---|---|---|
| `nrSpin1` | `list` | A list of the components of the spins of the objects. |
| `nrSpin2` | `list` | A list of the components of the spins of the objects. |
| `n_hat` | `list` | A list of the components of the unit vectors of the objects, against which the components of the spins are specified. |
| `ln_hat` | `list` | A list of the components of the unit vectors of the objects, against which the components of the spins are specified. |

#### Returns

| Name | Type | Description |
|---|---|---|
|  | `S1, S2 : list` | The transformed spins in LAL frame. |

---

### `get_nr_to_lal_rotation_angles`

```python
get_nr_to_lal_rotation_angles(h5_file: object, sim_metadata: dict, inclination: float, phi_ref: float = 0, f_ref: float | None = None, t_ref: float | None = None, tol: float = 1e-06) -> tuple
```

Get the angular coordinates :math:`\theta, \phi`
and the rotation angle :math:`\alpha` from the H5 file

#### Parameters

| Name | Type | Description |
|---|---|---|
| `h5_file` | `file object` | The waveform h5 file handle. |
| `inclination` | `float` | The inclination angle. |
| `phi_ref` | `float` | The orbital phase at reference time. |
| `f_ref` | `float` | The reference orbital frequency or time |
| `t_ref` | `float` | The reference orbital frequency or time |
| `sim_metadata` | `dict` | The sim_metadata of the waveform file. |
| `tol` | `float` | The tolerance to use to allow floating point representation errors. |

#### Returns

| Name | Type | Description |
|---|---|---|
| `angles` | `dict` | The angular corrdinates Theta, Psi, and the rotation angle Alpha. If available, this also contains the reference time and frequency. |

> **Notes**
> 
> Variable definitions.
> 
> theta : Returned inclination angle of source in NR coordinates.
> psi :   Returned azimuth angle of source in NR coordinates.
> alpha: Returned polarisation angle.
> h5_file: h5py object of the NR HDF5 file.
> inclination: inclination of source in LAL source frame.
> phi_ref: Orbital reference phase.
> t_ref : Reference time. -1 or None indicates it was not found in the sim_metadata.
> f_ref: Reference frequency.
> 
> The reference epoch is defined close to the beginning of the simulation.

---

{% endraw %}
