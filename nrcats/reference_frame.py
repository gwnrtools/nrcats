"""Map an NR simulation onto NRSur7dq4's parameter conventions.

The surrogate is told ``(q, chiA, chiB)`` *at a reference frequency*.  Getting
that correspondence right is the whole problem: spins in a precessing system
swing continuously, so "the spins" is meaningless without saying *when*, and a
reference epoch read at the wrong instant describes a different binary.

Two independent choices are exposed so they can be measured against each other:

``frame_method="A"``
    The orbital angular momentum, from the catalog's own dynamics.  This is the
    quantity the surrogate's convention is *defined* against, so it is the
    default wherever a catalog provides it.

    How it is obtained differs by catalog, because the reference epoch does.
    RIT publishes ``relaxed-LNhat`` / ``relaxed-nhat`` at ``relaxed-time``, and
    that is the only epoch where its spins exist, so those are read directly.
    SXS publishes ``reference_orbital_frequency`` at ``reference_time`` only,
    but our epoch is chosen by the backward omega_0 crossing floored at
    ``SXS_RELAXATION_CAP_M`` and so almost never coincides with it (measured:
    98% of runs have relaxation_time > 800 M).  There is no published
    angular-velocity time series to interpolate -- ``HorizonQuantities`` carries
    only masses, spins and coordinate centres -- so Lhat is obtained from
    ``r x rdot`` on smoothed horizon positions, which is the same quantity by
    definition, evaluated where it is needed.  :func:`validate_frame` checks
    that reconstruction against the published vector at ``reference_time``,
    where both exist.

    The cost of the finite difference is that ``coord_center_inertial`` is
    gauge-dependent, so gauge enters the *tilt* of Lhat rather than only the
    azimuth.  Smoothing suppresses it and the validation bounds it.

``frame_method="B"``
    The dominant radiation axis, from the waveform's coprecessing frame.  Needs
    no metadata at all, so it is the only option for MAYA, which publishes no
    orientation.  Measured against A on SXS and RIT it agrees to under a degree
    once the branch ambiguity is resolved (see ``_coprecessing_axis``).

Epoch selection differs by catalog because the *data* differs, not by choice:
SXS ships time-series horizon data so the epoch can be placed at the surrogate's
minimum frequency, while RIT and MAYA publish spins at exactly one instant.
"""

from dataclasses import dataclass, field

import numpy as np

# Junk radiation is an initial-data transient that leaves the domain in about a
# light-crossing time.  SXS quotes relaxation times well past that (1265-1451 M
# on the runs measured here); the physical bound is the domain size in code
# units, so a published value above this is clamped down rather than trusted.
SXS_RELAXATION_CAP_M = 800.0

# Below this, a match is dominated by merger-ringdown and reports little about
# the inspiral the surrogate is being tested on.
MIN_INSPIRAL_M = 300.0


@dataclass
class ReferenceState:
    """Surrogate parameters extracted from an NR simulation, with provenance."""

    chiA: np.ndarray  # heavier body, coprecessing frame at ``epoch``
    chiB: np.ndarray  # lighter body
    q: float
    f_ref: float  # cycles/M
    f_low: float  # cycles/M
    rotation: np.ndarray  # 3x3, inertial -> coprecessing; rows are the new basis
    epoch: float  # in the waveform's own time coordinate
    provenance: dict = field(default_factory=dict)


def _smooth(x, window):
    """Box-average along axis 0, edge-preserving.

    Used before every search and every value read.  Instantaneous frequency and
    spin components both carry orbital-timescale nutation on top of the secular
    trend, and taking a raw sample of either is what makes an epoch search land
    on a dip rather than on the crossing it was looking for.
    """
    x = np.asarray(x, dtype=float)
    if window < 3 or window >= len(x):
        return x
    k = np.ones(int(window)) / float(window)
    pad = int(window) // 2
    if x.ndim == 1:
        xp = np.pad(x, pad, mode="edge")
        return np.convolve(xp, k, mode="same")[pad:-pad]
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        xp = np.pad(x[:, j], pad, mode="edge")
        out[:, j] = np.convolve(xp, k, mode="same")[pad:-pad]
    return out


def _smooth_directions(vecs, window):
    """Average unit-vector-valued data without shrinking its magnitude.

    A box average of a rotating vector contracts toward the chord, losing
    ~sinc(theta/2) of its length over a window spanning angle theta.  Magnitude
    and direction are therefore averaged separately and recombined.
    """
    vecs = np.asarray(vecs, dtype=float)
    mag = _smooth(np.linalg.norm(vecs, axis=1), window)
    direction = _smooth(vecs, window)
    n = np.linalg.norm(direction, axis=1)
    n[n == 0] = 1.0
    return direction / n[:, None] * mag[:, None]


def gw_frequency(t, h22, window=0):
    """f_GW = |d(arg h22)/dt| / 2pi, in cycles/M, optionally smoothed."""
    t = np.asarray(t, dtype=float)
    phase = np.unwrap(np.angle(np.asarray(h22)))
    f = np.abs(np.gradient(phase, t)) / (2.0 * np.pi)
    return _smooth(f, window) if window else f


def backward_crossing(t, f, target, t_min=None, t_max=None):
    """Last time ``f`` crosses ``target`` from below, searching back from merger.

    Searching *forward* for the first sample above ``target`` is what put the
    reference epoch inside the junk burst: over that burst the phase derivative
    is large and erratic, so the first crossing is spurious.  Frequency rises
    monotonically through the inspiral, so the *last* crossing is the secular
    one, and walking back from merger reaches it without ever entering the junk.

    ``t_max`` must exclude merger onward -- ringdown sits at a high, nearly
    constant frequency, and past it the amplitude decays into numerical noise
    where the phase derivative is meaningless.

    Returns ``None`` when ``f`` never drops to ``target`` in the window, which
    is the normal case for short waveforms that start well above it.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    sel = np.ones(len(t), dtype=bool)
    if t_min is not None:
        sel &= t >= t_min
    if t_max is not None:
        sel &= t <= t_max
    if not sel.any():
        return None
    ts, fs = t[sel], f[sel]
    below = np.where(fs <= target)[0]
    if len(below) == 0:
        return None
    return float(ts[below[-1]])


def _coprecessing_axis(times, data, ell_min, ell_max):
    """Dominant radiation axis zhat'(t) from the waveform's coprecessing frame.

    ``to_coprecessing_frame`` fixes the axis only up to a sign, and it does pick
    the anti-aligned branch in practice -- measured at 179.15 deg against the
    orbital Lhat on SXS:BBH:1348 and 179.24 deg on 1350, versus 0.40 deg on 1346.
    Left alone that is a 180 deg error in the frame handed to the surrogate.

    The branch is pinned by the *sense of rotation*, not by mode power.  Power
    cannot work: equatorial symmetry gives h_{2,-2} = (-1)^l conj(h_{2,2}), so
    the two moduli are equal by construction and a power test returns noise
    (measured ratios 0.98-1.17 on runs whose correct branch is known).  The
    orbital phase advances one way about +Lhat and the other about -Lhat, so
    d(arg h22)/dt in the coprecessing frame carries the sign unambiguously.

    Verified against published Lhat: SXS:BBH:1348 gives +1 and does need the
    flip (148.7 deg raw), while SXS:BBH:1346, RIT:BBH:0940 and RIT:BBH:0504 give
    -1 and do not (14.8, 6.6, 0.5 deg raw).  Needs no metadata, so it also works
    for MAYA, where nothing is published to check against.
    """
    import quaternionic
    import sxs as _sxs

    w = _sxs.WaveformModes(
        np.asarray(data, dtype=complex),
        time=np.asarray(times, dtype=float),
        time_axis=0,
        modes_axis=1,
        ell_min=ell_min,
        ell_max=ell_max,
        spin_weight=-2,
        data_type="h",
    ).to_coprecessing_frame()

    arr = np.asarray(w.data)
    lm = [(ell, m) for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)]
    h22 = arr[:, lm.index((2, 2))]
    dphi = np.gradient(np.unwrap(np.angle(h22)), np.asarray(w.time))
    flip = float(np.median(dphi)) > 0.0

    R = quaternionic.array(np.asarray(w.frame))
    z = quaternionic.array([0.0, 0.0, 0.0, 1.0])
    axis = np.asarray(R * z * R.conjugate())[:, 1:]
    if flip:
        axis = -axis
    return np.asarray(w.time), axis, bool(flip), w


def _orbital_phase(w_copr, ell_min, ell_max):
    """Orbital phase from the coprecessing (2,2) mode: h22 ~ exp(-2 i phi_orb)."""
    arr = np.asarray(w_copr.data)
    lm = [(ell, m) for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)]
    return -np.unwrap(np.angle(arr[:, lm.index((2, 2))])) / 2.0


def build_frame(lhat, nhat):
    """Right-handed (nhat, lhat x nhat, lhat); rows are the new basis vectors.

    This is gwsurrogate's convention: chi_x = chi.nhat with nhat running
    lighter->heavier, chi_y = chi.(Lhat x nhat), chi_z = chi.Lhat.
    """
    lhat = np.asarray(lhat, float)
    lhat = lhat / np.linalg.norm(lhat)
    nhat = np.asarray(nhat, float)
    nhat = nhat - np.dot(nhat, lhat) * lhat  # orthogonalize against Lhat
    nn = np.linalg.norm(nhat)
    if nn < 1e-12:  # degenerate seed; any perpendicular direction will do
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, lhat)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        nhat = seed - np.dot(seed, lhat) * lhat
        nn = np.linalg.norm(nhat)
    nhat = nhat / nn
    return np.array([nhat, np.cross(lhat, nhat), lhat])


def omega_0_of(sur, q, chiA, chiB):
    """The surrogate's minimum reference frequency for *these* parameters.

    Reachable directly rather than by provoking an exception and parsing it:
    ``DynamicsSurrogate.get_omega(0, q, y0)`` is what the model evaluates
    internally before deciding to raise.  It depends on both spins and mass
    ratio, and neither dependence is small enough to ignore.  Measured at
    |chi| = 0.8, going from maximal anti-alignment to maximal alignment moves it
    by 17% at q = 1 (0.01485 -> 0.01737) and 28% at q = 6 (0.01756 -> 0.02244);
    over the precessing population with q < 6 and |chi| < 0.8 it spans 0.01497
    to 0.02177.  A hardcoded threshold would reject valid simulations at one end
    of that range and accept invalid epochs at the other.
    """
    y0 = np.append(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.append(np.asarray(chiA, float), np.asarray(chiB, float)),
    )
    return float(sur._sur_dimless.dynamics_sur.get_omega(0, q, y0))


def _lm_list(ell_min, ell_max):
    return [(ell, m) for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)]


def _mode_block(wfm, ell_max=4):
    """Dense (ntimes, nmodes) block for ell in [2, ell_max], in canonical order."""
    have = [tuple(x) for x in wfm.LM]
    lm = [k for k in _lm_list(2, ell_max) if k in have]
    top = max(ell for ell, _ in lm)
    lm = [k for k in _lm_list(2, top) if k in have]
    return np.asarray(wfm.data)[:, [have.index(k) for k in lm]], 2, top


def _axis_and_phase_from_waveform(wfm, epoch, ell_max=4):
    """Option B: (Lhat, nhat) at ``epoch`` derived from the waveform alone."""
    data, lmin, lmax = _mode_block(wfm, ell_max)
    t = np.asarray(wfm.time)
    tw, axis, flipped, w_copr = _coprecessing_axis(t, data, lmin, lmax)
    j = int(np.argmin(np.abs(tw - epoch)))
    lhat = axis[j] / np.linalg.norm(axis[j])
    # nhat: rotate the coprecessing x-axis by the orbital phase about Lhat.
    import quaternionic

    R = quaternionic.array(np.asarray(w_copr.frame)[j])
    x = quaternionic.array([0.0, 1.0, 0.0, 0.0])
    xhat = np.asarray(R * x * R.conjugate())[1:]
    if flipped:
        xhat = -xhat
    phi = float(_orbital_phase(w_copr, lmin, lmax)[j])
    if flipped:
        phi = -phi
    yhat = np.cross(lhat, xhat)
    nhat = np.cos(phi) * xhat + np.sin(phi) * yhat
    return lhat, nhat, flipped


def _sxs_horizon_state(sim_obj, epoch, window_M):
    """Spins, masses and the orbital frame from SXS horizon time series."""
    h = sim_obj.horizons
    th = np.asarray(h.A.time, dtype=float)
    dt = float(np.median(np.diff(th)))
    win = max(int(round(window_M / max(dt, 1e-6))), 3)

    rA = _smooth(h.A.coord_center_inertial.ndarray, win)
    rB = _smooth(h.B.coord_center_inertial.ndarray, win)
    cA = _smooth_directions(h.A.chi_inertial.ndarray, win)
    cB = _smooth_directions(h.B.chi_inertial.ndarray, win)
    mA = _smooth(np.asarray(h.A.christodoulou_mass, dtype=float), win)
    mB = _smooth(np.asarray(h.B.christodoulou_mass, dtype=float), win)

    i = int(np.clip(np.argmin(np.abs(th - epoch)), 1, len(th) - 2))
    heavier_is_A = mA[i] >= mB[i]
    r_prim, r_sec = (rA, rB) if heavier_is_A else (rB, rA)
    chi_prim, chi_sec = (cA[i], cB[i]) if heavier_is_A else (cB[i], cA[i])
    m1, m2 = (mA[i], mB[i]) if heavier_is_A else (mB[i], mA[i])

    r = r_prim[i] - r_sec[i]
    v = (r_prim[i + 1] - r_sec[i + 1]) - (r_prim[i - 1] - r_sec[i - 1])
    v = v / (th[i + 1] - th[i - 1])
    lhat = np.cross(r, v)
    return {
        "chi_primary": chi_prim,
        "chi_secondary": chi_sec,
        "q": float(m1 / m2),
        "lhat": lhat / np.linalg.norm(lhat),
        "nhat": r / np.linalg.norm(r),
        "t_used": float(th[i]),
    }


def _iterate_epoch(sim_obj, wfm, sur, frame_method, window_M, max_iter=12, tol_M=1.0):
    """Fixed point between the epoch and omega_0, which depends on the spins.

    omega_0 is a function of (q, chiA, chiB); the spins come from the epoch; the
    epoch is where the frequency reaches omega_0/pi.  The loop is the honest way
    out of that circle.  It converges quickly because omega_0 is driven mainly by
    the aligned spin components, which barely move over the inspiral.
    """
    st = sim_obj.strain
    t = np.asarray(st.t, dtype=float)
    h22 = st.data[:, st.index(2, 2)]
    dt = float(np.median(np.diff(t)))
    win = max(int(round(window_M / max(dt, 1e-6))), 3)
    f_gw = gw_frequency(t, h22, window=win)
    t_merger = float(st.max_norm_time())

    t_relax = float(sim_obj.metadata["relaxation_time"])
    t_floor = min(t_relax, SXS_RELAXATION_CAP_M)

    epoch = t_floor
    hist, clamped, seen = [], False, []
    for it in range(max_iter):
        hz = _sxs_horizon_state(sim_obj, epoch, window_M)
        if frame_method == "B":
            lhat, nhat, flipped = _axis_and_phase_from_waveform(wfm, epoch)
        else:
            lhat, nhat, flipped = hz["lhat"], hz["nhat"], False
        R = build_frame(lhat, nhat)
        chiA, chiB = R @ hz["chi_primary"], R @ hz["chi_secondary"]
        w0 = omega_0_of(sur, hz["q"], chiA, chiB)
        cross = backward_crossing(t, f_gw, w0 / np.pi, t_max=t_merger)
        new = t_floor if (cross is None or cross < t_floor) else cross
        clamped = cross is None or cross < t_floor
        hist.append((epoch, w0, new))
        if abs(new - epoch) < tol_M:
            epoch = new
            break
        # A 2-cycle is the common non-convergence: the epoch implies spins whose
        # omega_0 sends it back where it came from.  Both endpoints are valid
        # references, so take the later (higher-frequency) one -- it is the
        # conservative choice, safely above omega_0 for either spin set.
        if any(abs(new - p) < tol_M for p in seen[:-1]):
            epoch = max(new, epoch)
            break
        seen.append(new)
        # Damp once the plain iteration has had a fair chance, so a slowly
        # drifting fixed point still lands rather than ringing out the budget.
        epoch = new if it < 3 else epoch + 0.5 * (new - epoch)
    else:
        raise RuntimeError(
            f"reference epoch did not converge in {max_iter} iterations; "
            f"history (epoch, omega_0, next) = {hist}"
        )

    hz = _sxs_horizon_state(sim_obj, epoch, window_M)
    if frame_method == "B":
        lhat, nhat, flipped = _axis_and_phase_from_waveform(wfm, epoch)
    else:
        lhat, nhat, flipped = hz["lhat"], hz["nhat"], False
    R = build_frame(lhat, nhat)
    chiA, chiB = R @ hz["chi_primary"], R @ hz["chi_secondary"]
    w0 = omega_0_of(sur, hz["q"], chiA, chiB)
    f_at = float(np.interp(epoch, t, f_gw))
    # Strictly above: the surrogate rejects omega_ref == omega_0, and landing
    # exactly on the floor is the common case once the epoch is chosen by the
    # omega_0 crossing itself.
    f_min = (w0 / np.pi) * (1.0 + 1e-6)
    return ReferenceState(
        chiA=chiA,
        chiB=chiB,
        q=hz["q"],
        f_ref=max(f_at, f_min),
        f_low=max(f_at, f_min),
        rotation=R,
        epoch=epoch,
        provenance={
            "catalog": "SXS",
            "frame_method": frame_method,
            "t_relaxation_published": t_relax,
            "t_floor": t_floor,
            "clamped": bool(clamped),
            "iterations": len(hist),
            "omega_0": w0,
            "f_gw_at_epoch": f_at,
            "t_merger": t_merger,
            "inspiral_M": t_merger - epoch,
            "coprecessing_branch_flipped": bool(flipped),
            "lhat": lhat.tolist(),
        },
    )


def _short_catalog_state(wfm, sur, frame_method, catalog, window_M):
    """RIT and MAYA: spins exist at exactly one instant, so the epoch is fixed.

    Neither catalog ships time-series horizon data, so there is nothing to
    search -- the reference is wherever the published spins are quoted.  That is
    harmless here because both start well above the surrogate's floor (Omega ~
    0.033-0.057 against omega_0 ~ 0.018), so f_ref is comfortably in range and
    the surrogate evolves the spins itself.
    """
    md = wfm.sim_metadata
    t = np.asarray(wfm.time, dtype=float)
    data, lmin, lmax = _mode_block(wfm)
    lm = _lm_list(lmin, lmax)
    h22 = data[:, lm.index((2, 2))]
    dt = float(np.median(np.diff(t)))
    win = max(int(round(window_M / max(dt, 1e-6))), 3)
    f_gw = gw_frequency(t, h22, window=win)
    t_merger = float(t[int(np.argmax(np.linalg.norm(data, axis=1)))])

    if catalog == "RIT":
        # relaxed-time is measured from the start of the simulation, not from
        # the array's own origin (RIT arrays are merger-centred).
        epoch = float(t[0]) + float(md["relaxed-time"])
        c1 = np.array([float(md[f"relaxed-chi1{c}"]) for c in "xyz"])
        c2 = np.array([float(md[f"relaxed-chi2{c}"]) for c in "xyz"])
        m1, m2 = float(md["relaxed-mass1"]), float(md["relaxed-mass2"])
        lhat_pub = np.array([float(md[f"relaxed-LNhat{c}"]) for c in "xyz"])
        nhat_pub = np.array([float(md[f"relaxed-nhat{c}"]) for c in "xyz"])
    elif catalog == "MAYA":
        # MAYA publishes no epoch and no orientation; the spins are initial-data
        # values, confirmed by omega_orbital agreeing with the measured
        # frequency at t[0] to ~1%.
        epoch = float(t[0])
        c1 = np.array([float(md[f"a1{c}"]) for c in "xyz"])
        c2 = np.array([float(md[f"a2{c}"]) for c in "xyz"])
        m1, m2 = float(md["m1"]), float(md["m2"])
        lhat_pub = nhat_pub = None
    else:
        raise ValueError(f"unsupported catalog {catalog!r}")

    if m1 < m2:  # nhat runs lighter -> heavier; keep the heavier body first
        m1, m2 = m2, m1
        c1, c2 = c2, c1
        if nhat_pub is not None:
            nhat_pub = -nhat_pub
    q = float(m1 / m2)

    flipped = False
    if frame_method == "B" or lhat_pub is None:
        lhat, nhat, flipped = _axis_and_phase_from_waveform(wfm, epoch)
        used = "B"
    else:
        lhat, nhat, used = lhat_pub, nhat_pub, "A"

    R = build_frame(lhat, nhat)
    chiA, chiB = R @ c1, R @ c2
    w0 = omega_0_of(sur, q, chiA, chiB)
    f_at = float(np.interp(epoch, t, f_gw))
    if f_at < w0 / np.pi:
        raise ValueError(
            f"{catalog}: f_GW at the only available epoch ({f_at:.6f} cycles/M) "
            f"is below the surrogate minimum ({w0 / np.pi:.6f}); this simulation "
            "cannot be given a valid reference and must be excluded."
        )
    return ReferenceState(
        chiA=chiA,
        chiB=chiB,
        q=q,
        f_ref=f_at,
        f_low=f_at,
        rotation=R,
        epoch=epoch,
        provenance={
            "catalog": catalog,
            "frame_method": used,
            "clamped": True,
            "iterations": 1,
            "omega_0": w0,
            "f_gw_at_epoch": f_at,
            "t_merger": t_merger,
            "inspiral_M": t_merger - epoch,
            "coprecessing_branch_flipped": bool(flipped),
            "lhat": np.asarray(lhat).tolist(),
            "lhat_published": None if lhat_pub is None else np.asarray(lhat_pub).tolist(),
        },
    )


def extract_reference_state(
    wfm, sur, catalog=None, sim_obj=None, frame_method="A", window_M=None
):
    """Extract surrogate parameters for the binary a given NR waveform describes.

    Parameters
    ----------
    wfm : WaveformModes
        The NR waveform, in the catalog's own time coordinate.
    sur : gwsurrogate model
        Needed for ``omega_0``, which is spin-dependent.
    catalog : {"SXS", "RIT", "MAYA"}, optional
        Defaults to ``wfm.sim_metadata["catalog_type"]``.
    sim_obj : sxs.Simulation, optional
        Required for SXS -- it is the only source of time-series spins.
    frame_method : {"A", "B"}
        See the module docstring.  Falls back to "B" where a catalog publishes
        no orientation, and records which was actually used in ``provenance``.
    window_M : float, optional
        Smoothing window.  Defaults to one orbital period at the epoch.
    """
    if catalog is None:
        catalog = str(wfm.sim_metadata.get("catalog_type", "")).upper()
    if frame_method not in ("A", "B"):
        raise ValueError("frame_method must be 'A' or 'B'")
    if window_M is None:
        window_M = 100.0
    if catalog == "SXS":
        if sim_obj is None:
            raise ValueError("SXS requires sim_obj (time-series spins live there)")
        return _iterate_epoch(sim_obj, wfm, sur, frame_method, window_M)
    return _short_catalog_state(wfm, sur, frame_method, catalog, window_M)


# --- validation gate ---------------------------------------------------------

FRAME_VALIDATION_TOL_DEG = 2.0


def _tilt_and_azimuth(chi, lhat, nhat):
    """(|chi|, tilt from Lhat, azimuth about Lhat) -- azimuth is gauge."""
    chi = np.asarray(chi, float)
    mag = float(np.linalg.norm(chi))
    if mag == 0.0:
        return 0.0, 0.0, 0.0
    lhat = np.asarray(lhat, float) / np.linalg.norm(lhat)
    e1 = np.asarray(nhat, float) - np.dot(nhat, lhat) * lhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(lhat, e1)
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(chi, lhat) / mag, -1, 1))))
    az = float(np.degrees(np.arctan2(np.dot(chi, e2), np.dot(chi, e1))))
    return mag, tilt, az


def validate_frame(wfm, sim_obj=None, catalog=None,
                   tol_deg=FRAME_VALIDATION_TOL_DEG, window_M=100.0):
    """Check the production frame against an independent published one.

    For SXS this is the gate that matters: method A reconstructs Lhat from
    ``r x rdot`` on horizon positions because no angular-velocity time series is
    published, and this asserts -- per simulation, rather than on the strength of
    a few spot checks -- that the reconstruction reproduces the published
    ``reference_orbital_frequency`` direction at ``reference_time``, the one
    epoch where both exist.

    Only *physical* quantities are gated.  The common azimuth of both spins about
    Lhat is a gauge choice (it is an orbital phase shift, exactly degenerate with
    the rotation the matcher already maximizes over), so it is reported but never
    failed on.  Spin magnitudes, tilts from Lhat, and the *relative* azimuth
    between the two spins are physical, and are gated.

    Returns a dict with the measured deviations and ``passed``.  RIT publishes
    Lhat directly, so there is no second dynamics source to check it against and
    the result is marked ``not_applicable``; MAYA publishes no orientation at all.
    """
    if catalog is None:
        catalog = str(wfm.sim_metadata.get("catalog_type", "")).upper()
    out = {"catalog": catalog, "tol_deg": tol_deg, "passed": None,
           "reason": None}

    if catalog != "SXS":
        out["passed"] = None
        out["reason"] = ("not_applicable: this catalog publishes Lhat directly "
                         "(RIT) or not at all (MAYA); no independent source")
        return out
    if sim_obj is None:
        raise ValueError("SXS validation needs sim_obj")

    md = sim_obj.metadata
    t_ref = float(md["reference_time"])
    hz = _sxs_horizon_state(sim_obj, t_ref, window_M)

    w = np.asarray(md["reference_orbital_frequency"], dtype=float)
    lhat_pub = w / np.linalg.norm(w)
    r = (np.asarray(md["reference_position1"], dtype=float)
         - np.asarray(md["reference_position2"], dtype=float))
    s1 = np.asarray(md["reference_dimensionless_spin1"], dtype=float)
    s2 = np.asarray(md["reference_dimensionless_spin2"], dtype=float)
    if float(md["reference_mass1"]) < float(md["reference_mass2"]):
        r = -r
        s1, s2 = s2, s1
    nhat_pub = r / np.linalg.norm(r)

    def ang(a, b):
        a = np.asarray(a, float) / np.linalg.norm(a)
        b = np.asarray(b, float) / np.linalg.norm(b)
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))

    out["t_reference"] = t_ref
    out["lhat_deviation_deg"] = ang(hz["lhat"], lhat_pub)
    out["nhat_deviation_deg"] = ang(hz["nhat"], nhat_pub)
    out["q_deviation"] = abs(hz["q"] - float(md["reference_mass_ratio"]))

    worst_mag = worst_tilt = 0.0
    az_fd, az_pub = [], []
    for chi_fd, chi_pub in ((hz["chi_primary"], s1), (hz["chi_secondary"], s2)):
        m_fd, t_fd, a_fd = _tilt_and_azimuth(chi_fd, hz["lhat"], hz["nhat"])
        m_pb, t_pb, a_pb = _tilt_and_azimuth(chi_pub, lhat_pub, nhat_pub)
        worst_mag = max(worst_mag, abs(m_fd - m_pb))
        worst_tilt = max(worst_tilt, abs(t_fd - t_pb))
        az_fd.append(a_fd)
        az_pub.append(a_pb)
    rel = (az_fd[0] - az_fd[1]) - (az_pub[0] - az_pub[1])
    out["spin_magnitude_deviation"] = worst_mag
    out["spin_tilt_deviation_deg"] = worst_tilt
    out["spin_relative_azimuth_deviation_deg"] = float((rel + 180) % 360 - 180)
    # gauge: reported, never gated
    out["spin_common_azimuth_deg"] = float(
        ((np.mean(az_fd) - np.mean(az_pub)) + 180) % 360 - 180)

    checks = {
        "lhat": out["lhat_deviation_deg"] <= tol_deg,
        "spin_tilt": out["spin_tilt_deviation_deg"] <= tol_deg,
        "spin_relative_azimuth": abs(out["spin_relative_azimuth_deviation_deg"]) <= tol_deg,
        "spin_magnitude": out["spin_magnitude_deviation"] <= 1e-2,
        "mass_ratio": out["q_deviation"] <= 1e-2,
    }
    out["checks"] = checks
    out["passed"] = all(checks.values())
    if not out["passed"]:
        out["reason"] = "failed: " + ", ".join(k for k, v in checks.items() if not v)
    return out
