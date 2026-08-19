"""Heavy-lifting implementations behind :class:`~nrcats.waveform.modes.WaveformModes`.

``WaveformModes`` is a ``numpy.ndarray`` subclass whose public API is documented
on the class itself.  The long numerical routines live here as free functions
taking the waveform instance as an explicit first argument (``wfm``), so the
class stays readable while behaviour is unchanged: each method in ``modes.py``
is a thin delegation to the matching function below.

Nothing here is part of the public API; import ``WaveformModes`` and call its
methods instead.
"""

import logging
import warnings

import numpy as np
import quaternionic
import spherical
from sxs import TimeSeries as sxs_TimeSeries

from nrcats import utils
from nrcats.waveform.units import _modal_dt

logger = logging.getLogger(__name__)


def _restrict_to_complete_blocks(common_modes, allow_partial=False):
    """Drop ell blocks that are not complete from -ell to +ell.

    A Wigner rotation mixes m within an ell block and is unitary on the whole
    block, not on a subset of it.  Given a partial block the callers below still
    rotate the zero-padded block -- scattering power into the m it does not hold
    -- then sum the overlap over the retained m only, while normalising by the
    *un*-rotated norm.  The objective is then not a normalised inner product
    between two fixed vectors, and its maximum is no longer 1 for waveforms that
    genuinely differ by a rotation.

    Measured on SXS:BBH:0161 against an exactly-rotated copy of itself, where
    the true answer is 1 by construction:

        rotation          complete blocks   positive-m only
        identity          1.000000          1.000000
        pure z-rotation   1.000000          1.000000
        general SO(3)     0.999974          0.977307

    The penalty is 2.3e-2, an order of magnitude above the mismatches these
    functions are used to measure, and exactly zero for a z-rotation because
    that is diagonal in m and moves nothing out of the retained set.  That is
    why aligned-spin work never exposed it and why precessing work cannot avoid
    it.

    Parameters
    ----------
    common_modes : set of (ell, m)
    allow_partial : bool
        Return the input unchanged, with a warning, instead of raising when
        nothing complete remains.  Used by the BMS path, where completeness is
        necessary but not sufficient anyway -- a supertranslation mixes across
        ell as well as within it, so a truncated ell_max leaks regardless and
        refusing on this criterion alone would imply a guarantee that does not
        hold.
    """
    ells = {int(ell) for ell, _ in common_modes}
    complete = {
        ell
        for ell in ells
        if all(
            any(int(a) == ell and int(b) == m for a, b in common_modes)
            for m in range(-ell, ell + 1)
        )
    }
    # int(), not the numpy scalars the mode index carries: under numpy 2 a list
    # of np.int64 renders as "[np.int64(3)]" in the log message.
    incomplete = sorted(int(e) for e in ells - complete)
    if incomplete:
        logger.warning(
            "dropping ell=%s -- present in both waveforms but not as complete "
            "-ell..+ell blocks, which a Wigner rotation requires.  Supply every "
            "m in the block (for NRSur7dq4, generate_surrogate_modes(..., "
            "modes='all')).",
            incomplete,
        )
    restricted = {(ell, m) for (ell, m) in common_modes if int(ell) in complete}
    if not restricted:
        if allow_partial:
            logger.warning(
                "no complete ell block is common to both waveforms; proceeding "
                "on partial blocks, which understates the match by up to 2.3e-2"
            )
            return common_modes
        raise ValueError(
            f"No complete ell block is common to both waveforms (ell present: "
            f"{sorted(ells)}, none complete).  A sphere-averaged match "
            "maximised over SO(3) needs every m from -ell to +ell, because the "
            "rotation mixes them; scoring a partial block understates the match "
            "by up to 2.3e-2.  For NRSur7dq4 pass "
            "generate_surrogate_modes(..., modes='all')."
        )
    return restricted


def _analysis_setup(
    arr1,
    arr2,
    common_modes,
    delta_t,
    alignment,
    taper_fraction,
    psd,
    psd_name,
    f_lower,
    f_upper,
    min_cycles,
    keys1=None,
    margin=0,
):
    """Window, taper, padding and PSD shared by the sphere-averaged matches.

    ``arr1`` / ``arr2`` map ``(ell, m)`` to complex mode arrays.  Returns None
    when the pair cannot be matched at all, otherwise a dict describing one
    analysis grid.

    The window is located **once**, on the reference mode, and applied to every
    mode.  Windowing each mode on its own peak would shift modes relative to
    one another, and the relative phase between modes is what the frame
    transformations are being fitted to.
    """
    from nrcats.waveform.matching import (
        MIN_BINS_IN_BAND,
        _get_crosscorr_lag,
        _get_merger_index,
        _start_window,
    )

    keys1 = list(keys1 or common_modes)

    if (2, 2) in common_modes:
        ref = (2, 2)
    else:
        ref = max(common_modes, key=lambda k: float(np.max(np.abs(arr1[k]))))

    if float(np.max(np.abs(arr1[ref]))) < 1e-50 or (
        float(np.max(np.abs(arr2[ref]))) < 1e-50
    ):
        return None

    n1 = min(len(arr1[k]) for k in keys1)
    n2 = min(len(arr2[k]) for k in common_modes)

    if alignment == "peak":
        i1 = _get_merger_index(arr1[ref])
        i2 = _get_merger_index(arr2[ref])
        before = min(i1, i2)
        after = min(n1 - i1, n2 - i2)
        s1, e1 = i1 - before, i1 + after
        s2, e2 = i2 - before, i2 + after
    elif alignment == "crosscorr":
        lag = _get_crosscorr_lag(arr1[ref], arr2[ref])
        s1, s2 = max(0, lag), max(0, -lag)
        e1, e2 = min(n1, lag + n2), min(n2, n1 - lag)
    else:
        raise ValueError(f"Unknown alignment method: {alignment}")

    # Reserve room at both ends for a transformation that consumes data, e.g.
    # a supertranslation: u' = u - alpha(theta, phi) is only defined where the
    # original waveform still has support.  Without this the analysis window
    # spans essentially all the data, every candidate transformation falls off
    # the end, and the search sees a flat penalty instead of a gradient.
    if margin:
        s1, e1 = s1 + margin, e1 - margin
        s2, e2 = s2 + margin, e2 - margin

    n_overlap = e1 - s1
    if n_overlap <= 1:
        return None
    overlap_seconds = n_overlap * delta_t

    # Power of two, chosen from the window rather than inherited from the
    # caller's PSD resolution.
    n_fft = 1
    while n_fft < n_overlap:
        n_fft <<= 1
    df = 1.0 / (n_fft * delta_t)
    length_f = n_fft // 2 + 1
    grid_f = np.arange(length_f) * df

    # Start taper only: these modes carry a full merger and ringdown, and the
    # ringdown decays on its own, so an end taper would attenuate real signal.
    window = _start_window(n_overlap, taper_fraction)

    if psd is None:
        from pycbc.psd import from_string

        one_sided = np.asarray(
            from_string(psd_name, length_f, df, low_freq_cutoff=f_lower)
        )
    else:
        src_f = np.arange(len(psd)) * float(psd.delta_f)
        one_sided = np.interp(grid_f, src_f, np.asarray(psd), left=np.inf, right=np.inf)
    usable = np.isfinite(one_sided) & (one_sided > 0)
    if not usable[grid_f > 0].any():
        return None
    psd_floor = float(grid_f[usable & (grid_f > 0)][0])

    f_floor = (min_cycles / overlap_seconds) if min_cycles > 0 else 0.0
    f_lower_used = max(f_lower or 0.0, f_floor, psd_floor)
    f_hi = f_upper if f_upper is not None else 0.5 / delta_t
    if int((f_hi - f_lower_used) / df) < MIN_BINS_IN_BAND:
        return None

    band = usable & (grid_f >= f_lower_used) & (grid_f <= f_hi)
    psd_full = np.full(n_fft, np.inf)
    psd_full[:length_f] = np.where(band, one_sided, np.inf)
    psd_full[length_f:] = psd_full[1 : (n_fft + 1) // 2][::-1]

    return {
        "s1": s1,
        "e1": e1,
        "s2": s2,
        "e2": e2,
        "n_overlap": n_overlap,
        "overlap_seconds": overlap_seconds,
        "window": window,
        "n_fft": n_fft,
        "df": df,
        "psd_full": psd_full,
        "f_lower_used": f_lower_used,
        "ref": ref,
    }


def get_mode(
    wfm,
    ell,
    em,
    total_mass=1.0,
    distance=1.0,
    delta_t=None,
    to_pycbc=True,
    delta_t_seconds=None,
    delta_t_Msun=None,
    t_relax=None,
):
    """Implementation of :meth:`WaveformModes.get_mode`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method
    for the full parameter and return documentation.
    """
    if delta_t_seconds is not None and delta_t_Msun is not None:
        raise ValueError(
            "Provide only one of `delta_t_seconds` or `delta_t_Msun`, not both."
        )

    m_secs = utils.time_to_physical(total_mass)

    if delta_t_seconds is not None:
        dt_physical = delta_t_seconds
        dt_dimless = delta_t_seconds / m_secs
    elif delta_t_Msun is not None:
        dt_dimless = delta_t_Msun
        dt_physical = delta_t_Msun * m_secs
    else:
        if delta_t is not None:
            warnings.warn(
                "The `delta_t` parameter of get_mode() is deprecated and will be "
                "removed in a future release. Use `delta_t_seconds` for physical "
                "seconds or `delta_t_Msun` for dimensionless M units instead.",
                DeprecationWarning,
                # 3, not 2: the user calls WaveformModes.<method>, which
                # delegates here, so the caller is two frames up.
                stacklevel=3,
            )
        else:
            delta_t = _modal_dt(wfm.time)
        if delta_t > 1.0 / 128:
            dt_dimless = delta_t
            dt_physical = delta_t * m_secs
        else:
            dt_physical = delta_t
            dt_dimless = delta_t / m_secs

    new_time_start = min(wfm.time)
    if t_relax is not None:
        new_time_start = max(new_time_start, float(t_relax))
    new_time = np.arange(new_time_start, max(wfm.time), dt_dimless)

    mode_data = np.array(wfm.data[:, wfm.index(ell, em)], dtype=complex)
    mode_ts = sxs_TimeSeries(mode_data, time=wfm.time)
    interpolated_mode_ts = mode_ts.interpolate(new_time)

    h_mode_complex = np.array(interpolated_mode_ts.data, dtype=complex)
    h_mode_complex *= utils.amp_to_physical(total_mass, distance)

    peak_time_sec = wfm.peak_time_22 * m_secs
    start_time_sec = new_time[0] * m_secs
    epoch = start_time_sec - peak_time_sec

    retval = wfm.to_pycbc(
        input_array=h_mode_complex,
        delta_t=dt_physical,
        epoch=epoch,
    )
    if not to_pycbc:
        retval = sxs_TimeSeries(retval.data, time=retval.sample_times)
    return retval


def get_td_waveform(
    wfm,
    total_mass,
    distance,
    inclination,
    coa_phase,
    delta_t=None,
    f_ref=None,
    t_ref=None,
    k=3,
    kind=None,
    tol=1e-6,
    lal_convention=False,
    delta_t_seconds=None,
    delta_t_Msun=None,
    t_relax=None,
):
    """Implementation of :meth:`WaveformModes.get_td_waveform`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method
    for the full parameter and return documentation.
    """
    from nrcats.waveform.matching import interpolate_in_amp_phase

    if delta_t_seconds is not None and delta_t_Msun is not None:
        raise ValueError(
            "Provide only one of `delta_t_seconds` or `delta_t_Msun`, not both."
        )

    m_secs = utils.time_to_physical(total_mass)

    if delta_t_seconds is not None:
        dt_dimless = delta_t_seconds / m_secs
    elif delta_t_Msun is not None:
        dt_dimless = delta_t_Msun
    else:
        if delta_t is not None:
            warnings.warn(
                "The `delta_t` parameter of get_td_waveform() is deprecated and "
                "will be removed in a future release. Use `delta_t_seconds` for "
                "physical seconds or `delta_t_Msun` for dimensionless M units.",
                DeprecationWarning,
                # 3, not 2: the user calls WaveformModes.<method>, which
                # delegates here, so the caller is two frames up.
                stacklevel=3,
            )
        else:
            delta_t = _modal_dt(wfm.time)
        if delta_t > 1.0 / 128:
            dt_dimless = delta_t
        else:
            dt_dimless = delta_t / m_secs

    new_time_start = min(wfm.time)
    if t_relax is not None:
        new_time_start = max(new_time_start, float(t_relax))
    new_time = np.arange(new_time_start, max(wfm.time), dt_dimless)

    angles = wfm.get_angles(
        inclination=inclination,
        coa_phase=coa_phase,
        f_ref=f_ref,
        t_ref=t_ref,
        tol=tol,
    )
    h = interpolate_in_amp_phase(
        wfm.evaluate([angles["theta"], angles["psi"], angles["alpha"]]),
        new_time,
        k=k,
        kind=kind,
    ) * utils.amp_to_physical(total_mass, distance)

    h.time *= m_secs

    if lal_convention:
        return wfm.to_pycbc(h)
    else:
        return wfm.to_pycbc(np.conjugate(h))


def match_single_mode(
    wfm,
    other,
    ell,
    em,
    psd,
    f_lower,
    delta_t=1.0 / 4096,
    f_upper=None,
    total_mass=1.0,
    distance=1.0,
    psd_name="aLIGOZeroDetHighPower",
    min_cycles=None,
    alignment="peak",
):
    """Implementation of :meth:`WaveformModes.match_single_mode`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method for the full
    parameter and return documentation.

    This delegates to :func:`~nrcats.waveform.matching.compute_mode_match`,
    which owns the merger-aligned common window, the start taper, the
    power-of-two padding and the band-validity checks.  Doing the filtering
    here as well would be a second, weaker implementation of the same thing.
    """
    from nrcats.waveform.matching import (
        MIN_CYCLES_AT_BAND_EDGE,
        compute_mode_match,
        mode_f_lower,
    )

    if min_cycles is None:
        min_cycles = MIN_CYCLES_AT_BAND_EDGE

    # Complex modes throughout: the merger index, the taper and the alignment
    # are all defined on the envelope |h_lm|, which Re(h_lm) cannot give since
    # it passes through zero every half cycle.  compute_mode_match takes the
    # real part itself, at the filter, where it costs nothing.
    h1 = wfm.get_mode(
        ell,
        em,
        total_mass=total_mass,
        distance=distance,
        to_pycbc=True,
        delta_t_seconds=delta_t,
    )

    if isinstance(other, dict):
        if (ell, em) not in other:
            raise KeyError(f"Mode ({ell}, {em}) not found in other waveform dict.")
        val = other[(ell, em)]
        h2 = val[0] if isinstance(val, (tuple, list)) else val
    else:
        h2 = other.get_mode(
            ell,
            em,
            total_mass=total_mass,
            distance=distance,
            to_pycbc=True,
            delta_t_seconds=delta_t,
        )

    return float(
        compute_mode_match(
            h1,
            h2,
            mode_f_lower(f_lower, em),
            psd=psd,
            psd_name=psd_name,
            f_upper=f_upper,
            min_cycles=min_cycles,
            alignment=alignment,
        )
    )


def match_sphere_averaged(
    wfm,
    other,
    psd,
    f_lower,
    f_upper=None,
    delta_t=1.0 / 4096,
    return_rotation=False,
    total_mass=1.0,
    distance=1.0,
    psd_name="aLIGOZeroDetHighPower",
    min_cycles=None,
    alignment="peak",
    taper_fraction=None,
):
    """Implementation of :meth:`WaveformModes.match_sphere_averaged`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method
    for the full parameter and return documentation.

    The setup here mirrors
    :func:`~nrcats.waveform.matching.compute_mode_match_detailed` -- merger
    aligned common window, start taper, power-of-two padding, PSD resampled
    onto the grid actually integrated -- with one deliberate difference: the
    window and the taper are chosen **once**, from the reference mode, and
    applied identically to every mode.  Windowing each mode on its own peak
    would shift modes relative to one another, and the relative phase between
    modes is precisely what the SO(3) rotation is being fitted to.
    """
    from scipy.optimize import differential_evolution
    from scipy.fft import fft, ifft

    from nrcats.waveform.matching import MIN_CYCLES_AT_BAND_EDGE, TAPER_FRACTION

    if min_cycles is None:
        min_cycles = MIN_CYCLES_AT_BAND_EDGE
    if taper_fraction is None:
        taper_fraction = TAPER_FRACTION

    if isinstance(other, dict):
        other_LM = list(other.keys())
    else:
        other_LM = list(map(tuple, other.LM))

    common_modes = set(map(tuple, wfm.LM)) & set(other_LM)
    if not common_modes:
        return (0.0, None) if return_rotation else 0.0

    common_modes = _restrict_to_complete_blocks(common_modes)

    h1_ts_dict = {}
    h2_ts_dict = {}

    # Load modes and align lengths
    for ell, m in common_modes:
        h1_ts_dict[(ell, m)] = wfm.get_mode(
            ell,
            m,
            total_mass=total_mass,
            distance=distance,
            to_pycbc=True,
            delta_t_seconds=delta_t,
        )
        if isinstance(other, dict):
            h2_ts_dict[(ell, m)] = other[(ell, m)]
        else:
            h2_ts_dict[(ell, m)] = other.get_mode(
                ell,
                m,
                total_mass=total_mass,
                distance=distance,
                to_pycbc=True,
                delta_t_seconds=delta_t,
            )

    arr1 = {k: np.asarray(v) for k, v in h1_ts_dict.items()}
    arr2 = {k: np.asarray(v) for k, v in h2_ts_dict.items()}

    setup = _analysis_setup(
        arr1,
        arr2,
        common_modes,
        delta_t,
        alignment,
        taper_fraction,
        psd,
        psd_name,
        f_lower,
        f_upper,
        min_cycles,
    )
    if setup is None:
        return (0.0, None) if return_rotation else 0.0

    n_fft = N_pad = setup["n_fft"]
    df = setup["df"]
    psd_full = setup["psd_full"]
    window = setup["window"]
    n_overlap = setup["n_overlap"]
    s1, e1, s2, e2 = setup["s1"], setup["e1"], setup["s2"], setup["e2"]

    h1_f_dict = {}
    h2_f_dict = {}
    for k in common_modes:
        pad1 = np.zeros(n_fft, dtype=complex)
        pad2 = np.zeros(n_fft, dtype=complex)
        pad1[:n_overlap] = arr1[k][s1:e1] * window
        pad2[:n_overlap] = arr2[k][s2:e2] * window
        h1_f_dict[k] = fft(pad1)
        h2_f_dict[k] = fft(pad2)

    wigner = spherical.Wigner(wfm.ell_max)
    ells_in_common = set(ell for ell, m in common_modes)

    def objective_function(x):
        alpha, beta, gamma = x
        R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
        D_full = wigner.D(R)

        total_norm1_sq = 0.0
        total_norm2_sq = 0.0
        for k in common_modes:
            total_norm1_sq += df * np.sum((np.abs(h1_f_dict[k]) ** 2) / psd_full)
            total_norm2_sq += df * np.sum((np.abs(h2_f_dict[k]) ** 2) / psd_full)

        if total_norm1_sq == 0 or total_norm2_sq == 0:
            return 1.0

        I_f_full = np.zeros(N_pad, dtype=complex)

        for ell in ells_in_common:
            D_ell = np.zeros((2 * ell + 1, 2 * ell + 1), dtype=complex)
            for i, m in enumerate(range(-ell, ell + 1)):
                for j, mp in enumerate(range(-ell, ell + 1)):
                    D_ell[i, j] = D_full[wigner.Dindex(ell, m, mp)]

            h2_matrix = np.zeros((N_pad, 2 * ell + 1), dtype=complex)
            for i, m in enumerate(range(-ell, ell + 1)):
                if (ell, m) in h2_f_dict:
                    h2_matrix[:, i] = h2_f_dict[(ell, m)]

            h2_rot_matrix = h2_matrix @ D_ell

            for j, m in enumerate(range(-ell, ell + 1)):
                if (ell, m) not in common_modes:
                    continue
                # No separate exp(1j * m * phi_c) factor here.  It used to
                # be applied on top of the rotation, but a twist about z is
                # already the third Euler angle: the Wigner matrix carries
                # exp(-1j * m * gamma) on this index, so phi_c and gamma
                # entered the objective only through their sum.  Verified
                # numerically before removal -- objective(phi_c + d, alpha,
                # beta, gamma + d) reproduced objective(phi_c, alpha, beta,
                # gamma) to 3e-16 over random samples.  Carrying both gave
                # the search four parameters and three effective
                # directions, wasting a dimension of the differential
                # evolution budget on an exactly flat direction and leaving
                # part of the transformation outside the returned rotation.
                I_f_full += (
                    h1_f_dict[(ell, m)] * np.conj(h2_rot_matrix[:, j])
                ) / psd_full

        _q = ifft(I_f_full)
        # np.abs, not np.real: taking the modulus maximizes over an overall
        # constant phase on the modes, h_lm -> e^{i alpha} h_lm for every
        # (l, m).  That is the polarization angle (alpha = 2 psi), it is a
        # convention rather than a physical difference, and no rotation in
        # SO(3) can produce it -- Wigner D mixes m within an ell block and
        # never multiplies the block by a scalar phase.  Using the real part
        # therefore penalized a pure convention mismatch.
        #
        # This is not hypothetical.  Every SXS simulation carries alpha ~ pi
        # against the surrogate (40/40 measured; see findings 5n in the
        # catalog-comparison-paper repo), and the SXS reader is where the
        # sign originates.  Measured on SXS:BBH:0304 against a phase-rotated
        # copy of itself, with np.real: identical 1.000000, global sign
        # 0.998682, global phase e^{i pi/2} 0.008579.  The half-turn was
        # partly absorbed by a pure z-rotation, but only to 1.3e-3 -- the
        # same order as the NR-against-surrogate mismatches this function
        # exists to measure -- and a quarter-turn destroyed the match
        # outright.
        max_inner_prod = df * N_pad * np.max(np.abs(_q))

        overlap = max_inner_prod / np.sqrt(total_norm1_sq * total_norm2_sq)
        if np.isnan(overlap):
            return 1.0

        return 1.0 - overlap

    bounds = [(0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi)]

    identity = [0.0, 0.0, 0.0]
    identity_mismatch = objective_function(identity)
    logger.info(
        "      [DEBUG] Sphere-averaged match at Identity Rotation: "
        f"{1.0 - identity_mismatch:.6f}"
    )

    result = differential_evolution(
        objective_function,
        bounds,
        x0=identity,
        popsize=10,
        maxiter=50,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
    )

    # Take the better of the two.  Differential evolution is stochastic and
    # is not guaranteed to return its own starting point: the identity
    # mismatch was computed here before, and logged, but discarded, so this
    # function could report a rotation that fits *worse* than doing nothing
    # -- an unambiguous defect, since the identity is always available.
    # Seeding at the identity via x0 makes that rare; comparing makes it
    # impossible.
    if identity_mismatch <= result.fun:
        best_x, best_fun = identity, identity_mismatch
    else:
        best_x, best_fun = result.x, result.fun
    match = 1.0 - best_fun

    if return_rotation:
        R_opt = quaternionic.array.from_euler_angles(best_x[0], best_x[1], best_x[2])
        return match, R_opt
    return match


def _supertranslation_params(j_max):
    """Real degrees of freedom of a real supertranslation for ell = 1..j_max.

    A supertranslation field is real, so its harmonic coefficients obey
    ``alpha^{l,m} = (-1)^m conj(alpha^{l,-m})``.  That leaves one real number
    for m = 0 and two for each m > 0, i.e. 2l+1 per l -- the dimension of the
    real spherical harmonics at that order, as it must be.  ell = 0 is a rigid
    time translation and is deliberately excluded: it is maximized exactly and
    for free by the FFT over t_c, so handing it to the optimizer would only add
    a redundant direction.

    Returns a list of ``(ell, m, part)`` with part in {'re', 'im'}.
    """
    out = []
    for ell in range(1, j_max + 1):
        out.append((ell, 0, "re"))
        for m in range(1, ell + 1):
            out.append((ell, m, "re"))
            out.append((ell, m, "im"))
    return out


def _build_supertranslation(values, layout, j_max):
    """Pack real parameters into scri's complex, ell=0-based coefficient array."""
    alpha = np.zeros((j_max + 1) ** 2, dtype=complex)

    def idx(ell, m):
        return ell * ell + (ell + m)

    for value, (ell, m, part) in zip(values, layout):
        if m == 0:
            alpha[idx(ell, 0)] += value  # real by the reality condition
        elif part == "re":
            alpha[idx(ell, m)] += value
            alpha[idx(ell, -m)] += ((-1) ** m) * value
        else:
            alpha[idx(ell, m)] += 1j * value
            alpha[idx(ell, -m)] += ((-1) ** m) * (-1j) * value
    return alpha


def match_sphere_averaged_bms_maximized(
    wfm,
    other,
    psd,
    f_lower,
    f_upper=None,
    j_max=1,
    delta_t=1.0 / 4096,
    total_mass=1.0,
    distance=1.0,
    psd_name="aLIGOZeroDetHighPower",
    min_cycles=None,
    alignment="peak",
    taper_fraction=None,
    alpha_max_M=10.0,
    seed_rotation=True,
    n_coarse=128,
    n_starts=3,
    maxfev=800,
    seed=None,
    return_transformation=False,
):
    """Implementation of :meth:`WaveformModes.match_sphere_averaged_bms_maximized`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method for the full
    parameter and return documentation.

    The supertranslation is applied by ``scri``'s exact grid transformation
    (``WaveformGrid.from_modes`` then ``to_modes``), not by a first-order
    expansion in Gaunt coefficients.
    """
    from scipy.optimize import minimize
    from scipy.fft import fft, ifft

    from nrcats.waveform.matching import MIN_CYCLES_AT_BAND_EDGE, TAPER_FRACTION

    try:
        import scri
    except ImportError as e:
        raise ImportError(
            "The 'scri' package is required for BMS supertranslation optimization. "
            "Install it with: pip install scri"
        ) from e

    if min_cycles is None:
        min_cycles = MIN_CYCLES_AT_BAND_EDGE
    if taper_fraction is None:
        taper_fraction = TAPER_FRACTION

    if isinstance(other, dict):
        other_LM = [tuple(k) for k in other.keys()]
    else:
        other_LM = [tuple(x) for x in other.LM]
    common_modes = set(map(tuple, wfm.LM)) & set(other_LM)
    if not common_modes:
        return (0.0, None) if return_transformation else 0.0

    # Same normalisation defect as match_sphere_averaged: the norms below are
    # summed over common_modes while the transformation mixes m.  Completeness
    # is necessary here but not sufficient -- a supertranslation also mixes
    # across ell, so power leaks past a truncated ell_max no matter how complete
    # each block is -- hence allow_partial, which warns rather than refusing.
    common_modes = _restrict_to_complete_blocks(common_modes, allow_partial=True)

    def _get(src, ell, m):
        if isinstance(src, dict):
            val = src[(ell, m)]
            return val[0] if isinstance(val, (tuple, list)) else val
        return src.get_mode(
            ell,
            m,
            total_mass=total_mass,
            distance=distance,
            to_pycbc=True,
            delta_t_seconds=delta_t,
        )

    h1_ts = {k: _get(wfm, *k) for k in common_modes}
    # scri needs complete ell blocks, so carry every mode `other` has, not just
    # the common ones; the transformation mixes m within (and across) blocks.
    h2_ts = {k: _get(other, *k) for k in other_LM}

    arr1 = {k: np.asarray(v) for k, v in h1_ts.items()}
    arr2 = {k: np.asarray(v) for k, v in h2_ts.items()}

    from nrcats import utils as _utils

    m_secs = _utils.time_to_physical(total_mass)
    alpha_bound = abs(alpha_max_M) * m_secs

    # No fixed analysis window here.  A supertranslation moves the merger in
    # retarded time and shortens the usable span, so a window fixed on the
    # untransformed pair no longer describes the transformed one: measured
    # against a known 4 M supertranslation, a fixed window put the objective's
    # minimum at ~2 M with mismatch 3.7e-3 and made the true answer look
    # *worse*, while re-deriving the window reaches 4.7e-5 at 4 M.  The window
    # is cheap next to the transformation, so it is recomputed per evaluation.
    ell_min2 = min(ell for ell, _ in other_LM)
    ell_max2 = max(ell for ell, _ in other_LM)
    n2 = min(len(v) for v in arr2.values())
    ref2 = (
        (2, 2)
        if (2, 2) in common_modes
        else max(common_modes, key=lambda k: float(np.max(np.abs(arr1[k]))))
    )
    t2 = np.asarray(h2_ts[ref2].sample_times, dtype=float)[:n2]

    def _sidx(ell, m):
        return ell * ell - ell_min2 * ell_min2 + (ell + m)

    data2 = np.zeros((n2, (ell_max2 + 1) ** 2 - ell_min2**2), dtype=complex)
    for (ell, m), v in arr2.items():
        data2[:, _sidx(ell, m)] = v[:n2]

    other_scri = scri.WaveformModes(
        t=t2,
        data=np.ascontiguousarray(data2),
        ell_min=ell_min2,
        ell_max=ell_max2,
        frameType=scri.Inertial,
        dataType=scri.h,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )

    layout = _supertranslation_params(j_max)
    euler_scale = np.array([2 * np.pi, np.pi, 2 * np.pi])

    def _unscale(u):
        """Normalized search coordinates -> physical ones.

        The supertranslation coefficients are O(1e-4) seconds and the Euler
        angles are O(1) radians.  A simplex built on the raw vector takes steps
        that are meaningless for one or the other, so the search runs in
        [-1, 1] per supertranslation coefficient and [0, 1] per angle.
        """
        u = np.asarray(u, dtype=float)
        return u[: len(layout)] * alpha_bound, u[len(layout) :] * euler_scale

    def _mismatch(u):
        st_values, euler = _unscale(u)
        alpha = _build_supertranslation(st_values, layout, j_max)
        R = quaternionic.array.from_euler_angles(*euler)

        try:
            tr = other_scri.transform(
                supertranslation=alpha,
                frame_rotation=np.asarray(R, dtype=float).tolist(),
                ell_max=ell_max2,
            )
        except Exception:
            return 1.0

        arr2t = {k: np.ascontiguousarray(tr.data[:, _sidx(*k)]) for k in common_modes}
        setup = _analysis_setup(
            arr1,
            arr2t,
            common_modes,
            delta_t,
            alignment,
            taper_fraction,
            psd,
            psd_name,
            f_lower,
            f_upper,
            min_cycles,
        )
        if setup is None:
            return 1.0

        n_fft = setup["n_fft"]
        df = setup["df"]
        psd_full = setup["psd_full"]
        window = setup["window"]
        n_ov = setup["n_overlap"]
        i1, j1, i2, j2 = setup["s1"], setup["e1"], setup["s2"], setup["e2"]

        inner = np.zeros(n_fft, dtype=complex)
        norm1_sq = 0.0
        norm2_sq = 0.0
        for k in common_modes:
            p1 = np.zeros(n_fft, dtype=complex)
            p2 = np.zeros(n_fft, dtype=complex)
            p1[:n_ov] = arr1[k][i1:j1] * window
            p2[:n_ov] = arr2t[k][i2:j2] * window
            H1 = fft(p1)
            H2 = fft(p2)
            norm1_sq += df * np.sum((np.abs(H1) ** 2) / psd_full)
            norm2_sq += df * np.sum((np.abs(H2) ** 2) / psd_full)
            inner += (H1 * np.conj(H2)) / psd_full

        if not (np.isfinite(norm1_sq) and np.isfinite(norm2_sq)):
            return 1.0
        if norm1_sq == 0 or norm2_sq == 0:
            return 1.0

        # np.abs, not np.real: the modulus maximizes over one constant phase
        # common to every mode -- the polarization angle -- which no BMS
        # transformation reproduces.  Same convention as match_sphere_averaged.
        peak = df * n_fft * np.max(np.abs(ifft(inner)))
        overlap = peak / np.sqrt(norm1_sq * norm2_sq)
        if not np.isfinite(overlap):
            return 1.0
        return 1.0 - overlap

    identity = np.zeros(len(layout) + 3)
    identity_mismatch = _mismatch(identity)

    n_st = len(layout)
    bounds = [(-1.0, 1.0)] * n_st + [(0.0, 1.0)] * 3

    # Coarse pass, then local polish.  The objective is a broad shallow plateau
    # with a narrow deep well at the answer: measured against a known 4 M
    # supertranslation, the mismatch moves only 3.9e-3 -> 2.9e-3 over the first
    # half of the range and then falls to 4.7e-5 inside the well.  Differential
    # evolution spreads its population over the plateau and stalls there --
    # popsize 6/maxiter 15 and popsize 12/maxiter 40 returned identical answers,
    # so the failure is the shape of the landscape, not the budget.  A quasi
    # random sweep locates the well and a simplex descends it.
    starts = [identity]

    # Seed the frame rotation from the rotation-only maximization.  That search
    # is cheap -- the Wigner rotation is applied in the Fourier domain, with no
    # grid transformation -- and it is far better at finding a large frame
    # offset than a local simplex started at the identity.  NR and surrogate
    # mode sets routinely differ by such an offset, so without this the
    # supertranslation search is polishing around the wrong frame entirely.
    # Both R and its inverse are tried: scri's frame_rotation is a passive
    # transformation of the grid frame and need not share the sign convention
    # of the Wigner rotation applied to the modes here.
    if seed_rotation:
        try:
            _, R0 = match_sphere_averaged(
                wfm,
                other,
                psd,
                f_lower,
                f_upper=f_upper,
                delta_t=delta_t,
                return_rotation=True,
                total_mass=total_mass,
                distance=distance,
                psd_name=psd_name,
                min_cycles=min_cycles,
                alignment=alignment,
                taper_fraction=taper_fraction,
            )
        except Exception:
            R0 = None
        if R0 is not None:
            for cand in (R0, R0.inverse):
                try:
                    ang = np.asarray(
                        quaternionic.array(cand).to_euler_angles, dtype=float
                    ).ravel()[:3]
                except Exception:
                    continue
                ang = np.array(
                    [
                        ang[0] % (2 * np.pi),
                        np.clip(ang[1], 0.0, np.pi),
                        ang[2] % (2 * np.pi),
                    ]
                )
                u = np.zeros(len(layout) + 3)
                u[len(layout) :] = ang / euler_scale
                starts.append(u)

    if n_coarse > 0:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=n_st, scramble=True, seed=seed)
        pts = sampler.random(int(n_coarse)) * 2.0 - 1.0
        # Sweep the supertranslation at the best rotation found so far, not at
        # the identity: in the wrong frame every sample looks equally bad and
        # the sweep ranks noise.
        rot_for_sweep = min(starts, key=_mismatch)[len(layout) :]
        scored = sorted(
            ((_mismatch(np.concatenate([pt, rot_for_sweep])), pt) for pt in pts),
            key=lambda t: t[0],
        )
        starts += [
            np.concatenate([pt, rot_for_sweep]) for _, pt in scored[: int(n_starts)]
        ]

    best_x, best_fun = identity, identity_mismatch
    for x0 in starts:
        result = minimize(
            _mismatch,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxfev": int(maxfev), "xatol": 1e-4, "fatol": 1e-10},
        )
        if result.fun < best_fun:
            best_x, best_fun = result.x, result.fun

    # A local method started away from the identity can end up worse than doing
    # nothing; reporting a transformation that fits worse than none is never
    # correct.
    if identity_mismatch <= best_fun:
        best_x, best_fun = identity, identity_mismatch

    match = 1.0 - best_fun
    if return_transformation:
        st_best, euler_best = _unscale(best_x)
        info = {
            "supertranslation": _build_supertranslation(st_best, layout, j_max),
            "supertranslation_M": np.asarray(st_best) / m_secs,
            "layout": layout,
            "frame_rotation": quaternionic.array.from_euler_angles(*euler_best),
            "identity_match": 1.0 - identity_mismatch,
        }
        return match, info
    return match


def diff_l2_norm(wfm, other, time_window=None, phase_align=True):
    """Implementation of :meth:`WaveformModes.diff_l2_norm`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method
    for the full parameter and return documentation.
    """

    t_min = max(wfm.time[0], other.time[0])
    t_max = min(wfm.time[-1], other.time[-1])
    if time_window is not None:
        t_min = max(t_min, time_window[0])
        t_max = min(t_max, time_window[1])

    if t_min >= t_max:
        return float("nan")

    mask1 = (wfm.time >= t_min) & (wfm.time <= t_max)
    t1 = wfm.time[mask1]

    if len(t1) < 2:
        return float("nan")

    data1 = wfm.data[mask1, :]
    other_interp = other.interpolate(t1)
    data2 = other_interp.data

    common_modes = set(map(tuple, wfm.LM)) & set(map(tuple, other.LM))
    if not common_modes:
        return float("nan")

    idx1 = [wfm.index(*lm) for lm in common_modes]
    idx2 = [other_interp.index(*lm) for lm in common_modes]

    d1 = data1[:, idx1]
    d2 = data2[:, idx2]

    if phase_align:
        from scipy.optimize import minimize_scalar

        C_m = {}
        for i, lm in enumerate(common_modes):
            ell, m = lm
            C_m[m] = C_m.get(m, 0) + np.trapz(d1[:, i] * np.conj(d2[:, i]), x=t1)

        def obj(dphi):
            val = 0.0
            for m, C in C_m.items():
                val += np.real(np.exp(-1j * m * dphi) * C)
            return -val

        res = minimize_scalar(obj, bounds=(0, 2 * np.pi), method="bounded")
        dphi = res.x

        for i, lm in enumerate(common_modes):
            ell, m = lm
            d2[:, i] *= np.exp(1j * m * dphi)

    diff = d1 - d2
    error_norm_sq = np.sum(
        [np.trapz(np.abs(diff[:, i]) ** 2, x=t1) for i in range(len(common_modes))]
    )
    norm1_sq = np.sum(
        [np.trapz(np.abs(d1[:, i]) ** 2, x=t1) for i in range(len(common_modes))]
    )

    if norm1_sq == 0:
        return float("nan")

    return float(np.sqrt(max(0, error_norm_sq) / max(0, norm1_sq)))
