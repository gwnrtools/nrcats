"""nrcats.waveform.frame: undo a recorded BMS correction and strip memory.

Synthetic waveforms throughout -- no network, no cache -- built with the same
helper the match tests use.  The properties pinned here are the ones that, if
violated, would silently produce a wrong frame rather than an error: the
negated pair really inverts a scri transform; memory removal really inverts
sxs's memory addition; the rebuilt object keeps its mode order and metadata;
and a waveform with nothing recorded is returned untouched, not guessed at.
"""

import numpy as np
import pytest

pytest.importorskip("scri")

from nrcats.waveform import frame  # noqa: E402
from test_match_consolidation import _chirp_wfm  # noqa: E402


def _by_lm(w):
    d = np.asarray(w.data)
    return {tuple(int(v) for v in lm): d[:, i] for i, lm in enumerate(w.LM)}


def _rel(a, b, keys):
    num = np.sqrt(sum(np.linalg.norm(a[k] - b[k]) ** 2 for k in keys))
    den = np.sqrt(sum(np.linalg.norm(a[k]) ** 2 for k in keys))
    return num / den


def _on_common_grid(w_ref, w):
    """Interpolate ``w``'s modes onto ``w_ref``'s time grid, away from the ends."""
    t_ref = np.asarray(w_ref.t)
    t = np.asarray(w.t)
    lo, hi = max(t_ref[0], t[0]), min(t_ref[-1], t[-1])
    sel = (t_ref > lo + 0.05 * (hi - lo)) & (t_ref < hi - 0.05 * (hi - lo))
    tt = t_ref[sel]
    A = {k: v[sel] for k, v in _by_lm(w_ref).items()}
    B = {
        k: np.interp(tt, t, v.real) + 1j * np.interp(tt, t, v.imag)
        for k, v in _by_lm(w).items()
    }
    return A, B


def test_rebuild_preserves_mode_order_and_metadata():
    w = _chirp_wfm(n_times=1024, ell_max=3)
    t, modes = frame._modes_by_lm(w)
    w2 = frame._rebuild(w, t, modes)
    assert [tuple(map(int, x)) for x in w2.LM] == [tuple(map(int, x)) for x in w.LM]
    assert np.allclose(np.asarray(w2.data), np.asarray(w.data))
    assert w2.ell_min == w.ell_min and w2.ell_max == w.ell_max


def test_rebuild_refuses_a_scrambled_layout():
    w = _chirp_wfm(n_times=512, ell_max=2)
    t, modes = frame._modes_by_lm(w)
    # drop a mode from the middle of an ell block: the constructor derives a
    # full block from ell_min/ell_max, so the data no longer matches it
    modes.pop((2, 0))
    with pytest.raises(RuntimeError, match="mode layout mismatch"):
        frame._rebuild(w, t, modes)


def test_undo_com_correction_inverts_a_forward_transform():
    """The property the module exists for: transform by (st, bv), undo, get
    the original back -- to well inside the change the transform made."""
    w = _chirp_wfm(n_times=2048, ell_max=4)
    t, modes = frame._modes_by_lm(w)
    st = np.array([0.02, -0.01, 0.015])  # M, the scale of the catalog's corrections
    bv = np.array([3e-5, -5e-5, 8e-5])  # c
    fwd = frame._to_scri(t, modes, 4).transform(
        space_translation=st, boost_velocity=bv, ell_max=4
    )
    w_fwd = frame._rebuild(
        w, np.asarray(fwd.t), {k: fwd.data[:, fwd.index(*k)] for k in modes}
    )
    w_back = frame.undo_com_correction(w_fwd, st, bv)

    keys = list(modes)
    A, F = _on_common_grid(w, w_fwd)
    A2, B = _on_common_grid(w, w_back)
    moved = _rel(A, F, keys)
    residual = _rel(A2, B, keys)
    assert moved > 1e-3, "the forward transform did nothing; the test is vacuous"
    assert residual < 0.05 * moved
    assert residual < 1e-3


def test_strip_memory_inverts_sxs_add_memory():
    from sxs.waveforms import memory as M

    w = _chirp_wfm(n_times=2048, ell_max=4)
    t0 = float(np.asarray(w.t)[len(w.t) // 8])
    with_mem = M.add_memory(w, integration_start_time=t0)
    w_mem = frame._rebuild(
        w,
        np.asarray(with_mem.t),
        {k: v for k, v in _by_lm(with_mem).items() if k[0] >= 2},
    )
    stripped = frame.strip_memory(w_mem, t0)
    keys = [k for k in _by_lm(w)]
    added = _rel(_by_lm(w), _by_lm(w_mem), keys)
    residual = _rel(_by_lm(w), _by_lm(stripped), keys)
    assert added > 1e-4, "add_memory did nothing; the test is vacuous"
    assert residual < 0.05 * added
    assert [tuple(map(int, x)) for x in stripped.LM] == [
        tuple(map(int, x)) for x in w.LM
    ]


def test_harmonize_with_nothing_recorded_returns_the_input():
    w = _chirp_wfm(n_times=512, ell_max=2)
    out, rec = frame.harmonize(w, sidecar={"modifications": {}})
    assert out is w
    assert not rec.com_undone and not rec.memory_stripped
    assert any("nothing" in n for n in rec.notes)


def test_harmonize_as_is_is_a_no_op_with_an_empty_record():
    w = _chirp_wfm(n_times=512, ell_max=2)
    out, rec = frame.harmonize(
        w, sim_id="SXS:BBH:0000", undo_com=False, remove_memory=False
    )
    assert out is w
    assert rec.as_dict()["notes"] == []


def test_harmonize_applies_only_what_the_sidecar_records():
    w = _chirp_wfm(n_times=1024, ell_max=3)
    sidecar = {
        "modifications": {
            "transform": {
                "space_translation": [0.01, 0.0, 0.0],
                "boost_velocity": [0.0, 0.0, 2e-5],
            }
        }
    }
    out, rec = frame.harmonize(w, sidecar=sidecar)
    assert rec.com_undone and not rec.memory_stripped
    assert rec.space_translation == [0.01, 0.0, 0.0]
    assert any("add_memory" in n for n in rec.notes)
    assert out is not w


def _rel_on_grid(ref: dict, x: dict, margin=0.02) -> float:
    """|x - ref| / |ref| with x interpolated onto ref's grid over the overlap.

    Element-wise comparison is wrong here: every transform trims the grid's
    start by a few samples, so a chirp compared against itself shifted by k
    samples reports a phase error that has nothing to do with the transform.
    """
    tr = np.asarray(ref[(2, 2)].sample_times, dtype=float)
    tx = np.asarray(x[(2, 2)].sample_times, dtype=float)
    lo, hi = max(tr[0], tx[0]), min(tr[-1], tx[-1])
    sel = (tr > lo + margin * (hi - lo)) & (tr < hi - margin * (hi - lo))
    tt = tr[sel]
    num = den = 0.0
    for k in ref:
        a = np.asarray(ref[k])[sel]
        xb = np.asarray(x[k])
        b = np.interp(tt, tx, xb.real) + 1j * np.interp(tt, tx, xb.imag)
        num += np.linalg.norm(a - b) ** 2
        den += np.linalg.norm(a) ** 2
    return float(np.sqrt(num / den))


def test_apply_bms_to_mode_dict_round_trips_and_is_in_geometric_units():
    """Apply a CoM-scale boost and an ell = 2 supertranslation to a physical-unit
    mode dict, then the negated pair; the identity must be at the interpolation
    floor and each round trip far inside what the forward step moved.  The
    ell = 2 element is the one that exposes units: in physical strain the
    inhomogeneous term is wrong by ~1e14, and the forward move would read as
    1e15 rather than order one."""
    from test_match_consolidation import _mode_dict, DT, M_TOT

    w = _chirp_wfm(n_times=4096, ell_max=3)
    md = _mode_dict(w)
    ident = frame.apply_bms_to_mode_dict(md, DT, total_mass=M_TOT)
    assert _rel_on_grid(md, ident) < 1e-10

    a20 = np.zeros(16, dtype=complex)
    a20[6] = 0.02  # (2, 0), in M
    v = np.array([0.0, 0.0, 1e-4])
    fwd = frame.apply_bms_to_mode_dict(
        md, DT, supertranslation_M=a20, boost_velocity=v, total_mass=M_TOT
    )
    back = frame.apply_bms_to_mode_dict(
        fwd, DT, supertranslation_M=-a20, boost_velocity=-v, total_mass=M_TOT
    )
    moved, residual = _rel_on_grid(md, fwd), _rel_on_grid(md, back)
    assert 1e-3 < moved < 10.0, moved  # physics-scale, not 1e15
    assert residual < 0.01 * moved, (moved, residual)


def _block_norms(md, ell):
    """sum_m |h_lm|^2 at each time -- invariant under a rotation of the frame."""
    keys = [k for k in md if k[0] == ell]
    n = min(len(md[k]) for k in keys)
    return sum(np.abs(np.asarray(md[k])[:n]) ** 2 for k in keys)


def test_to_coprecessing_preserves_ell_block_norms_and_lands_on_z():
    """A rotation cannot change sum_m |h_lm|^2 within a complete ell block, and
    the rotated waveform's own radiation axis must be zhat -- that is what
    "coprecessing" means.  Both are checked on a WaveformModes and on a mode
    dict, which must agree with each other."""
    w = _chirp_wfm(n_times=1024, ell_max=3)

    rotated, axis = frame.to_coprecessing(w)
    before, after = _by_lm(w), _by_lm(rotated)
    for ell in (2, 3):
        b, a = _block_norms(before, ell), _block_norms(after, ell)
        n = min(len(b), len(a))
        assert np.allclose(b[:n], a[:n], rtol=1e-8, atol=0), ell

    # idempotence: rotating again finds an axis along zhat
    _, axis2 = frame.to_coprecessing(rotated)
    z = np.array([0.0, 0.0, 1.0])
    cos = np.abs(axis2 @ z) / np.linalg.norm(axis2, axis=1)
    assert np.median(cos) > 0.999, np.median(cos)
    assert axis.shape[1] == 3

    # the dict path must give the same rotation as the object path
    md = _by_lm(w)
    rot_md, axis_md = frame.to_coprecessing(md, times=np.asarray(w.time))
    n = min(len(axis), len(axis_md))
    assert np.allclose(axis[:n], axis_md[:n], atol=1e-10)


def test_to_coprecessing_refuses_an_incomplete_ell_block():
    """The rotation mixes m within a block, so a dict missing modes would be
    silently zero-filled and the result would be wrong rather than partial."""
    w = _chirp_wfm(n_times=256, ell_max=2)
    md = _by_lm(w)
    md.pop((2, -1))
    with pytest.raises(ValueError, match="complete blocks"):
        frame.to_coprecessing(md)
