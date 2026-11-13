---
title: nrcats.waveform.frame
parent: API Reference
nav_order: 17
---

<!-- GENERATED FILE — DO NOT EDIT. Regenerate with `python bin/generate_api_docs.py`. -->
{% raw %}

# `nrcats.waveform.frame`

Put a catalog waveform into a stated asymptotic frame, and say what was done.

The three catalogs publish strain in three different BMS frames, and SXS v3
records what it did to each file in a JSON sidecar: a centre-of-mass
``space_translation`` and ``boost_velocity`` (a BMS transformation), and
``add_memory`` (the electric null memory, integrated from the relaxation
time).  RIT and MAYA do neither.  This module undoes those operations with
the same code that applied them -- ``scri`` for the transformation, ``sxs``
for the memory -- and returns a record of what was applied, so a comparison
can state which frame each side was in.  Nothing here fits anything: every
parameter comes from the sidecar or the caller.

Two facts that shape the implementation, both measured on ``SXS:BBH:0078``
(catalog-comparison-paper findings 5aa and the round-trip experiments behind
it):

* ``scri``'s transform by ``(space_translation, boost_velocity)`` followed
  by the negated pair returns the modes to 1e-5 in L2 over ell <= 8, 3e-7 on
  (2,2).  The negated pair is the inverse to the precision that matters,
  because the corrections are ~1e-2 M and ~1e-4 c and the cross term is
  second order.
* ``sxs.WaveformModes.remove_memory`` returns an object with ``ell_min = 0``
  and a different mode layout.  Index by ``(ell, m)``, never by position: a
  positional comparison reports a 140% change in (2,2) that is not there.


## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## *class* `FrameRecord`

What was done to a waveform, for the manifest.


### `as_dict`

```python
as_dict() -> dict
```

*No docstring.*

---

## Functions

### `read_sidecar`

```python
read_sidecar(sim_id: str, download: bool = False) -> dict
```

The ``Strain_N2.json`` sidecar of ``sim_id``'s highest Lev, as a dict.

Loads the catalog index with ``download=False``: resolving the latest
release through api.github.com once per process is what exhausted the
unauthenticated limit on a job array.  The sidecar itself comes from the
data host, so ``download=True`` is safe for a genuine cache miss.

---

### `undo_com_correction`

```python
undo_com_correction(w, space_translation, boost_velocity, ell_max: int | None = None)
```

Apply the inverse of a recorded ``(space_translation, boost_velocity)``.

The inverse is the negated pair; see the module docstring for the
round-trip measurement.  A BMS transformation resamples retarded time, so
the result is on a new time grid, which is why this returns a whole
object rather than editing in place.

---

### `strip_memory`

```python
strip_memory(w, integration_start_time: float)
```

Remove the electric null memory ``sxs`` added, using the start time it
recorded.  ``sxs`` documents this as returning the waveform "as close as
possible to before the memory correction was applied" when given the
relaxation time, which is what the sidecar records; measured on
SXS:BBH:0078 the (2,0) step goes from 0.182 to 0.0005 of max|h_22|.

---

### `harmonize`

```python
harmonize(w, sim_id: str | None = None, sidecar: dict | None = None, undo_com: bool = True, remove_memory: bool = True, download: bool = False)
```

Undo what the catalog did to ``w``, per its sidecar, and say so.

Returns ``(waveform, FrameRecord)``.  With neither operation requested it
returns ``w`` itself and an empty record, so a pipeline can run the
"as-is" treatment through the same call and record that too.  A waveform
with no sidecar (RIT, MAYA) gets an empty record and a note: nothing is
known to undo, and nothing is guessed.

---

### `to_coprecessing`

```python
to_coprecessing(obj, ell_max: int | None = None, times=None)
```

Rotate a waveform into its own coprecessing frame.

Accepts an nrcats :class:`WaveformModes` (rotated in its native geometric
representation and returned as the same kind of object) or a
``{(ell, m): series}`` dict, whose values may be pycbc ``TimeSeries`` --
returned as the same type on the same grid -- or bare arrays, in which case
``times`` is required and arrays come back.  Returns ``(rotated, axis)``,
where ``axis`` is the dominant radiation axis in the original frame,
sign-resolved.

The coprecessing frame follows the instantaneous radiation axis, so this is
a **time-dependent** rotation and the result is not a waveform any inertial
observer sees: it is a diagnostic that divides the precession out, and the
difference between a match taken here and one taken in the inertial frame
is what the precession dynamics contribute.

Units need no care, unlike a supertranslation.  A rotation acts on the
modes homogeneously -- within each complete ell block and with no additive
term -- so it is exact in whatever units the caller holds, physical or
geometric (contrast :func:`apply_bms_to_mode_dict`, where feeding physical
strain to a dimensionless declaration was wrong by 1e14;
catalog-comparison-paper findings 5ab).

The axis is fixed only up to a sign by ``to_coprecessing_frame``, and the
anti-aligned branch does get picked in practice; the sense of rotation of
the coprecessing (2,2) phase resolves it (see
``reference_frame._coprecessing_axis``, which this delegates to so the two
cannot drift apart).

---

### `apply_bms_to_mode_dict`

```python
apply_bms_to_mode_dict(modes: dict, delta_t: float, supertranslation_M=None, boost_velocity=None, frame_rotation=None, total_mass: float = 1.0, distance: float = 1.0)
```

Transform a ``{(ell, m): series}`` dict by a BMS element and return the
same kind of dict on the transformed retarded-time grid.

``supertranslation_M`` is the coefficient array as ``scri`` expects it
(``(ell_max + 1)**2`` complex entries, ell = 0 first) **in units of M**, so
that an element fitted at one total mass can be applied at another: the
geometry is mass-independent, only its expression in seconds changes.
``boost_velocity`` is dimensionless.  ``frame_rotation`` is a quaternion.
Series may be pycbc TimeSeries or plain arrays; a TimeSeries' epoch is
honoured, otherwise the grid starts at zero.

---

{% endraw %}
