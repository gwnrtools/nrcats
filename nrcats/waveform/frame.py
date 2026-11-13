"""Put a catalog waveform into a stated asymptotic frame, and say what was done.

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
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "FrameRecord",
    "read_sidecar",
    "undo_com_correction",
    "strip_memory",
    "harmonize",
    "to_coprecessing",
]


@dataclass
class FrameRecord:
    """What was done to a waveform, for the manifest."""

    sim_id: str | None = None
    sidecar: str | None = None
    com_undone: bool = False
    space_translation: list[float] | None = None
    boost_velocity: list[float] | None = None
    memory_stripped: bool = False
    memory_integration_start_time: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# the sidecar
# --------------------------------------------------------------------------


def _highest_lev_key(files) -> str | None:
    keys = [k for k in files if k.endswith("Strain_N2.json")]
    if not keys:
        return None
    if "Strain_N2.json" in keys:
        return "Strain_N2.json"
    levs = []
    for k in keys:
        try:
            levs.append((int(k.split(":")[0].replace("Lev", "")), k))
        except ValueError:
            continue
    return sorted(levs)[-1][1] if levs else keys[0]


def read_sidecar(sim_id: str, download: bool = False) -> dict:
    """The ``Strain_N2.json`` sidecar of ``sim_id``'s highest Lev, as a dict.

    Loads the catalog index with ``download=False``: resolving the latest
    release through api.github.com once per process is what exhausted the
    unauthenticated limit on a job array.  The sidecar itself comes from the
    data host, so ``download=True`` is safe for a genuine cache miss.
    """
    import sxs

    sims = sxs.load("simulations", download=False)
    key = _highest_lev_key(sims[sim_id].files)
    if key is None:
        raise KeyError(f"{sim_id}: no Strain_N2.json in its file list")
    try:
        return dict(sxs.load(f"{sim_id}/{key}", download=False))
    except Exception:  # noqa: BLE001 -- cache miss; the file host is not GitHub
        if not download:
            raise
        return dict(sxs.load(f"{sim_id}/{key}", download=True))


# --------------------------------------------------------------------------
# conversions.  scri and sxs each want their own object; the pipeline wants
# nrcats.waveform.WaveformModes back, indexed the way it started.
# --------------------------------------------------------------------------


def _modes_by_lm(w) -> tuple[np.ndarray, dict]:
    t = np.asarray(w.t, dtype=float)
    data = np.asarray(w.data)
    return t, {tuple(int(v) for v in lm): data[:, i] for i, lm in enumerate(w.LM)}


def _to_scri(t, modes: dict, ell_max: int):
    import scri

    ell_min = 2
    n = (ell_max + 1) ** 2 - ell_min**2
    data = np.zeros((len(t), n), dtype=complex)
    for (ell, m), v in modes.items():
        if ell_min <= ell <= ell_max:
            data[:, ell * ell - ell_min * ell_min + (ell + m)] = v
    return scri.WaveformModes(
        t=np.ascontiguousarray(t),
        data=np.ascontiguousarray(data),
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        dataType=scri.h,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )


def _rebuild(template, t, modes: dict):
    """A new nrcats WaveformModes on time grid ``t`` with ``modes`` (a
    (ell, m) -> series dict), carrying ``template``'s metadata and mode order.

    Mirrors how ``SXSCatalog.get`` constructs the object: data, time, the
    simulation metadata, and the sxs-level ``_metadata`` minus the keys the
    constructor supplies itself.
    """
    from nrcats.waveform.modes import WaveformModes

    lm = [tuple(int(v) for v in x) for x in template.LM]
    keep = [k for k in lm if k in modes]
    data = np.column_stack([modes[k] for k in keep])
    meta = dict(getattr(template, "_metadata", {}) or {})
    for k in ("time", "metadata", "waveform_data_location"):
        meta.pop(k, None)
    meta["waveform_data_location"] = ""
    meta["ell_min"] = min(k[0] for k in keep)
    meta["ell_max"] = max(k[0] for k in keep)
    sim_meta = None
    try:
        sim_meta = template.sim_metadata
    except Exception:  # noqa: BLE001 -- RIT/MAYA objects carry it differently
        sim_meta = getattr(template, "metadata", None)
    w = WaveformModes(data, np.asarray(t), metadata=sim_meta, **meta)
    # The constructor derives LM from ell_min/ell_max; the data columns above
    # are ordered to match template.LM restricted to `keep`.  Assert rather
    # than trust it.
    got = [tuple(int(v) for v in x) for x in w.LM]
    if got != keep:
        raise RuntimeError(
            f"mode layout mismatch after rebuild: expected {keep[:6]}..., got {got[:6]}..."
        )
    return w


# --------------------------------------------------------------------------
# the two operations
# --------------------------------------------------------------------------


def undo_com_correction(
    w, space_translation, boost_velocity, ell_max: int | None = None
):
    """Apply the inverse of a recorded ``(space_translation, boost_velocity)``.

    The inverse is the negated pair; see the module docstring for the
    round-trip measurement.  A BMS transformation resamples retarded time, so
    the result is on a new time grid, which is why this returns a whole
    object rather than editing in place.
    """
    t, modes = _modes_by_lm(w)
    lmax = ell_max or max(k[0] for k in modes)
    s = _to_scri(t, modes, lmax)
    out = s.transform(
        space_translation=-np.asarray(space_translation, dtype=float),
        boost_velocity=-np.asarray(boost_velocity, dtype=float),
        ell_max=lmax,
    )
    new = {}
    for ell, m in modes:
        if 2 <= ell <= lmax:
            new[(ell, m)] = np.ascontiguousarray(out.data[:, out.index(ell, m)])
    return _rebuild(w, np.asarray(out.t), new)


def strip_memory(w, integration_start_time: float):
    """Remove the electric null memory ``sxs`` added, using the start time it
    recorded.  ``sxs`` documents this as returning the waveform "as close as
    possible to before the memory correction was applied" when given the
    relaxation time, which is what the sidecar records; measured on
    SXS:BBH:0078 the (2,0) step goes from 0.182 to 0.0005 of max|h_22|."""
    out = w.remove_memory(float(integration_start_time))
    t, modes = _modes_by_lm(out)
    return _rebuild(w, t, {k: v for k, v in modes.items() if k[0] >= 2})


def harmonize(
    w,
    sim_id: str | None = None,
    sidecar: dict | None = None,
    undo_com: bool = True,
    remove_memory: bool = True,
    download: bool = False,
):
    """Undo what the catalog did to ``w``, per its sidecar, and say so.

    Returns ``(waveform, FrameRecord)``.  With neither operation requested it
    returns ``w`` itself and an empty record, so a pipeline can run the
    "as-is" treatment through the same call and record that too.  A waveform
    with no sidecar (RIT, MAYA) gets an empty record and a note: nothing is
    known to undo, and nothing is guessed.
    """
    rec = FrameRecord(sim_id=sim_id)
    if sidecar is None and sim_id is not None and (undo_com or remove_memory):
        try:
            sidecar = read_sidecar(sim_id, download=download)
            rec.sidecar = f"{sim_id}/Strain_N2.json"
        except Exception as exc:  # noqa: BLE001 -- recorded, not fatal
            rec.notes.append(f"no sidecar: {type(exc).__name__}: {exc}")
            return w, rec
    mods = (sidecar or {}).get("modifications", {}) or {}
    tr = mods.get("transform", {}) or {}
    mem = mods.get("add_memory")

    if remove_memory:
        if mem and "integration_start_time" in mem:
            w = strip_memory(w, mem["integration_start_time"])
            rec.memory_stripped = True
            rec.memory_integration_start_time = float(mem["integration_start_time"])
        else:
            rec.notes.append("no add_memory in sidecar; nothing stripped")
    if undo_com:
        st, bv = tr.get("space_translation"), tr.get("boost_velocity")
        if st is not None and bv is not None:
            w = undo_com_correction(w, st, bv)
            rec.com_undone = True
            rec.space_translation = [float(x) for x in st]
            rec.boost_velocity = [float(x) for x in bv]
        else:
            rec.notes.append("no transform in sidecar; nothing undone")
    return w, rec


# --------------------------------------------------------------------------
# applying a *given* BMS element -- for the transfer control: the element
# fitted at one total mass is applied, not re-fitted, at the others.
# --------------------------------------------------------------------------


def to_coprecessing(obj, ell_max: int | None = None, times=None):
    """Rotate a waveform into its own coprecessing frame.

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
    """
    from nrcats.reference_frame import _coprecessing_axis

    if isinstance(obj, dict):
        keys = sorted(k for k in obj if ell_max is None or k[0] <= ell_max)
        if not keys:
            raise ValueError("no modes to rotate")
        lo = min(k[0] for k in keys)
        hi = max(k[0] for k in keys)
        lm = [(ell, m) for ell in range(lo, hi + 1) for m in range(-ell, ell + 1)]
        missing = [k for k in lm if k not in obj]
        if missing:
            raise ValueError(
                f"the rotation mixes m within an ell block, so complete blocks "
                f"are required; missing {missing[:4]}"
            )
        ref = obj[(2, 2)] if (2, 2) in obj else obj[keys[0]]
        n = min(len(obj[k]) for k in lm)
        data = np.zeros((n, len(lm)), dtype=complex)
        for k in lm:
            data[:, lm.index(k)] = np.asarray(obj[k])[:n]
        if hasattr(ref, "sample_times"):
            t = np.asarray(ref.sample_times, dtype=float)[:n]
        elif times is not None:
            t = np.asarray(times, dtype=float)[:n]
        else:
            raise ValueError(
                "a dict of bare arrays needs `times`; only pycbc TimeSeries "
                "carry their own grid"
            )
        _, axis, _, w = _coprecessing_axis(t, data, lo, hi)
        arr = np.asarray(w.data)
        cols = {k: np.ascontiguousarray(arr[:, lm.index(k)]) for k in lm}
        if not hasattr(ref, "sample_times"):
            return cols, axis
        cls = type(ref)
        out = {
            k: cls(v, delta_t=ref.delta_t, epoch=ref.start_time)
            for k, v in cols.items()
        }
        return out, axis

    lm = [tuple(int(v) for v in x) for x in obj.LM]
    _, axis, _, w = _coprecessing_axis(
        np.asarray(obj.time, dtype=float),
        np.asarray(obj),
        int(obj.ell_min),
        int(obj.ell_max),
    )
    arr = np.asarray(w.data)
    rotated = _rebuild(obj, np.asarray(w.time), {k: arr[:, lm.index(k)] for k in lm})
    return rotated, axis


def apply_bms_to_mode_dict(
    modes: dict,
    delta_t: float,
    supertranslation_M=None,
    boost_velocity=None,
    frame_rotation=None,
    total_mass: float = 1.0,
    distance: float = 1.0,
):
    """Transform a ``{(ell, m): series}`` dict by a BMS element and return the
    same kind of dict on the transformed retarded-time grid.

    ``supertranslation_M`` is the coefficient array as ``scri`` expects it
    (``(ell_max + 1)**2`` complex entries, ell = 0 first) **in units of M**, so
    that an element fitted at one total mass can be applied at another: the
    geometry is mass-independent, only its expression in seconds changes.
    ``boost_velocity`` is dimensionless.  ``frame_rotation`` is a quaternion.
    Series may be pycbc TimeSeries or plain arrays; a TimeSeries' epoch is
    honoured, otherwise the grid starts at zero.
    """
    from nrcats import utils

    keys = sorted(modes)
    n = min(len(np.asarray(v)) for v in modes.values())
    first = modes[keys[0]]
    if hasattr(first, "sample_times"):
        t_s = np.asarray(first.sample_times, dtype=float)[:n]
    else:
        t_s = np.arange(n) * float(delta_t)
    ell_max = max(k[0] for k in keys)
    # Geometric units throughout the transform: the scri object declares r and
    # M scaled out, and the inhomogeneous supertranslation term assumes it.
    # Physical strain with alpha in seconds is wrong by ~1e14 for ell >= 2.
    m_secs = utils.time_to_physical(total_mass)
    amp = utils.amp_to_physical(total_mass, distance)
    t_M = t_s / m_secs
    s = _to_scri(t_M, {k: np.asarray(v)[:n] / amp for k, v in modes.items()}, ell_max)
    kwargs = {"ell_max": ell_max}
    if supertranslation_M is not None:
        kwargs["supertranslation"] = np.asarray(supertranslation_M, dtype=complex)
    if boost_velocity is not None and len(boost_velocity):
        kwargs["boost_velocity"] = [float(b) for b in boost_velocity]
    if frame_rotation is not None:
        kwargs["frame_rotation"] = np.asarray(frame_rotation, dtype=float).tolist()
    tr = s.transform(**kwargs)
    # back onto the original uniform grid, over the span the transform kept
    tw = t_M[(t_M >= tr.t[0]) & (t_M <= tr.t[-1])]
    tri = tr.interpolate(tw)
    out = {}
    for k in keys:
        col = np.ascontiguousarray(tri.data[:, tri.index(*k)]) * amp
        if hasattr(first, "sample_times"):
            from pycbc.types import TimeSeries

            out[k] = TimeSeries(
                col.astype(np.complex128),
                delta_t=float(delta_t),
                epoch=float(tw[0] * m_secs),
            )
        else:
            out[k] = col
    return out
