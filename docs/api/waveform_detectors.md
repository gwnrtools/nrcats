---
title: nrcats.waveform.detectors
parent: API Reference
nav_order: 18
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.waveform.detectors`

Observatories: one place where a name resolves to a noise curve and a band.

Every quantity in these comparisons is a noise-weighted overlap, so it depends
on the observatory twice -- through the weighting across the band, and through
the band's edges.  Measuring the same waveforms in three observatories is then
one argument changed, *provided* the mapping from a name to a curve lives in
exactly one place.  Two copies of that mapping drift, and a comparison whose
noise curve differs between two code paths is not a comparison.

Curves come from ``pycbc.psd``'s LALSimulation set where it has them, so none
is hand-entered; the two it does not have -- Cosmic Explorer's current official
40 km and 20 km baselines (CE-T2000017 v009, 2025) -- are shipped as data files
under ``waveform/data/`` with their origin in ``psd_file`` and resolved by
:func:`psd_from_name`, so a matcher given a registry label builds the right
curve without knowing where it lives.

Measured sensitivity, ``sqrt(S_n)`` in strain per root-Hz, on the grid these
comparisons use::

                                      5 Hz     10 Hz     20 Hz    100 Hz
    aLIGOZeroDetHighPower           4.9e-22   9.7e-23   2.1e-23   3.9e-24
    CosmicExplorerP1600143          1.2e-22   7.3e-25   2.0e-25   1.6e-25
    CE40 (T2000017 v009, 40 km)     8.0e-24   6.8e-25   2.6e-25   2.1e-25
    CE20 (T2000017 v009, 20 km)     1.4e-23   1.3e-24   4.6e-25   3.2e-25
    EinsteinTelescopeP1600143       3.6e-24   1.2e-24   9.4e-25   3.9e-25
    CosmicExplorerPessimistic...    2.6e-20   2.1e-24   3.9e-25   2.5e-25
    CosmicExplorerWideband...       1.2e-22   7.4e-25   2.5e-25   2.1e-25

``CosmicExplorerP1600143`` is the 2016 curve; the collaboration has superseded
it, and against the current 40 km baseline it is 1.3x *more* sensitive over
20-100 Hz and worse only below 7 Hz.  It is kept as ``CE`` because results were
published with it; new work should name ``CE40``.  ``EinsteinTelescopeP1600143``
equals the official ET-D single-interferometer curve (ET-0000A-18) to four
figures at every frequency checked, so ``ET`` needs no file.

Two things in that table decide how the results should be read.  **Cosmic
Explorer is no more sensitive at 5 Hz than Advanced LIGO is**: its wall sits
near 10 Hz, and the pessimistic variant has effectively nothing below 10 Hz at
all.  **The Einstein Telescope is**, by a factor of thirty over CE at 5 Hz.  So
CE's advantage is breadth and depth above 10 Hz while ET's includes genuine
low-frequency reach, and any statement about which observatory a numerical
waveform's error matters most for has to say at which mass -- because the
frequency a numerical waveform *starts* at, in Hz, falls as 1/M.

The low cutoff is therefore not simply the detector's.  A numerical waveform has
finite length, so its own starting frequency bounds the band from below; at
40 Msun an SXS waveform begins near 20 Hz and reaching 5 Hz would need of order
forty times more inspiral than the catalogs publish.  :func:`band` returns the
binding constraint along with the value, so every result can say which one set
its band rather than leaving the reader to assume.


## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Constants

| Name | Value |
|---|---|
| `DETECTORS` | `{'aLIGO': Detector(name='aLIGO', psd_name='aLIGOZeroDetHighPower', f_lower=20.0, f_upper=None, reference='Advanced LIGO design sensitivity, zero-detuned high power', note='the default throughout this work; every published number uses it'), 'CE': Detector(name='CE', psd_name='CosmicExplorerP1600143', f_lower=5.0, f_upper=None, reference='LIGO-P1600143', note='sensitive from ~10 Hz; at 5 Hz it is no better than Advanced LIGO, so the 5-10 Hz decade is carried but contributes almost nothing'), 'CE40': Detector(name='CE40', psd_name='CosmicExplorer40km_T2000017v009', f_lower=5.0, f_upper=None, reference='CE-T2000017 v009, cosmic_explorer_strain.txt: the baseline 40 km detector, https://dcc.cosmicexplorer.org/CE-T2000017/public (zip md5 50221625c73894bb6c15113db2a4d09d, file md5 c65d973c293bf29ab89b0153f578ba64)', note='the current official Cosmic Explorer curve; supersedes CE', psd_file='cosmic_explorer_40km_strain_T2000017v009.txt'), 'CE20': Detector(name='CE20', psd_name='CosmicExplorer20km_T2000017v009', f_lower=5.0, f_upper=None, reference='CE-T2000017 v009, cosmic_explorer_20km_strain.txt: the baseline 20 km detector (file md5 aca1ebf1821f95998cd04d7db59a402c)', note='the proposed second, 20 km Cosmic Explorer', psd_file='cosmic_explorer_20km_strain_T2000017v009.txt'), 'ET': Detector(name='ET', psd_name='EinsteinTelescopeP1600143', f_lower=5.0, f_upper=None, reference='LIGO-P1600143; equal to the official ET-D single-interferometer curve', note='the one curve here with real sensitivity below 10 Hz'), 'CE-pessimistic': Detector(name='CE-pessimistic', psd_name='CosmicExplorerPessimisticP1600143', f_lower=10.0, f_upper=None, reference='LIGO-P1600143', note='effectively blind below 10 Hz, hence the higher cutoff'), 'CE-wideband': Detector(name='CE-wideband', psd_name='CosmicExplorerWidebandP1600143', f_lower=5.0, f_upper=None, reference='LIGO-P1600143', note='sensitivity check on the CE tuning')}` |

---

## *class* `Detector`

One observatory: a curve, a band, and where the curve came from.


## Functions

### `detector`

```python
detector(name: str | Detector) -> Detector
```

Resolve an observatory name, case-insensitively, with a useful error.

---

### `band`

```python
band(name: str | Detector, f_waveform: float | None = None)
```

``(f_lower, f_upper, bound_by)`` for one observatory and one waveform.

``f_waveform`` is the lowest frequency the waveform itself supports, as the
campaigns already compute it from the model's starting frequency.  The band
starts at whichever of the two constraints is higher, and ``bound_by`` says
which -- ``"detector"``, ``"waveform"``, or ``"equal"`` -- so that a result
can report whether it was limited by the instrument or by the length of the
simulation.  Passing ``None`` gives the detector's own band.

---

### `psd_from_name`

```python
psd_from_name(psd_name: str, length: int, delta_f: float, low_freq_cutoff: float)
```

A one-sided PSD from a name: a registry label backed by a data file, or a
LALSimulation name through ``pycbc.psd.from_string``.

This is the one place a ``psd_name`` string becomes a curve, so the matchers
can keep taking names and a registered file-backed curve reaches them
without any caller knowing the difference.

---

### `psd_for`

```python
psd_for(name: str | Detector, length: int, delta_f: float, f_lower=None)
```

The observatory's one-sided PSD on a given frequency grid.

Resolves through :func:`psd_from_name`, so callers never name a curve or a
file directly; both live in :data:`DETECTORS` alone.

---

{% endraw %}
