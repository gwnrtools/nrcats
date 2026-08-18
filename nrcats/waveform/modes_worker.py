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
):
    """Implementation of :meth:`WaveformModes.match_single_mode`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method
    for the full parameter and return documentation.
    """
    from pycbc.filter import match as pycbc_match

    h1 = wfm.get_mode(ell, em, to_pycbc=True, delta_t_seconds=delta_t).real()

    if isinstance(other, dict):
        if (ell, em) not in other:
            raise KeyError(f"Mode ({ell}, {em}) not found in other waveform dict.")
        val = other[(ell, em)]
        h2 = val[0] if isinstance(val, (tuple, list)) else val.real()
    else:
        h2 = other.get_mode(ell, em, to_pycbc=True, delta_t_seconds=delta_t).real()

    target_len = max(len(h1), len(h2))
    h1.resize(target_len)
    h2.resize(target_len)

    psd_copy = psd.copy()
    psd_copy.resize(len(h1.to_frequencyseries()))

    mode_f_lower = f_lower * abs(em) / 2.0 if em != 0 else f_lower

    mm, _ = pycbc_match(
        h1,
        h2,
        psd=psd_copy,
        low_frequency_cutoff=mode_f_lower,
        high_frequency_cutoff=f_upper,
    )
    return float(mm)


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
):
    """Implementation of :meth:`WaveformModes.match_sphere_averaged`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method
    for the full parameter and return documentation.
    """
    from scipy.optimize import differential_evolution
    from scipy.fft import fft, ifft

    # Compute overlapping frequency range
    df = psd.delta_f
    low_idx = int(f_lower / df) if f_lower else 0
    high_idx = int(np.ceil(f_upper / df)) if f_upper else len(psd)

    if isinstance(other, dict):
        other_LM = list(other.keys())
    else:
        other_LM = list(map(tuple, other.LM))

    common_modes = set(map(tuple, wfm.LM)) & set(other_LM)
    if not common_modes:
        return (0.0, None) if return_rotation else 0.0

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

    # Determine required length to match PSD's delta_f
    N_pad = int(np.round(1.0 / (df * delta_t)))

    # Build two-sided PSD array
    psd_full = np.ones(N_pad) * np.inf
    psd_len = N_pad // 2 + 1
    for i in range(low_idx, min(high_idx, len(psd))):
        if i < psd_len:
            val = psd.data[i]
            if val > 0:
                psd_full[i] = val
                if i > 0 and (N_pad - i) < N_pad:
                    psd_full[N_pad - i] = val

    # Compute full complex FFTs, zero-padded to N_pad
    h1_f_dict = {}
    h2_f_dict = {}
    for k in common_modes:
        ts1 = h1_ts_dict[k].data
        ts2 = h2_ts_dict[k].data

        # Zero-pad arrays to N_pad
        pad1 = np.zeros(N_pad, dtype=complex)
        pad2 = np.zeros(N_pad, dtype=complex)

        pad1[: len(ts1)] = ts1
        pad2[: len(ts2)] = ts2

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


def match_sphere_averaged_bms_maximized(
    wfm,
    other,
    psd,
    f_lower,
    f_upper=None,
    j_max=1,
):
    """Implementation of :meth:`WaveformModes.match_sphere_averaged_bms_maximized`.

    ``wfm`` is the :class:`WaveformModes` instance; see the method
    for the full parameter and return documentation.
    """
    from scipy.optimize import minimize

    try:
        import scri
    except ImportError as e:
        raise ImportError(
            "The 'scri' package is required for BMS supertranslation optimization. "
            "Install it with: pip install scri"
        ) from e

    alpha_jk_indices = [(j, k) for j in range(1, j_max + 1) for k in range(-j, j + 1)]

    max_len = 0
    for ell, m in wfm.LM:
        max_len = max(
            max_len,
            len(wfm.get_mode(ell, m, to_pycbc=True, delta_t_seconds=1 / 4096)),
        )

    ref_mode_ts = wfm.get_mode(2, 2, to_pycbc=True, delta_t_seconds=1 / 4096)
    ref_mode_ts.resize(max_len)
    ref_fs = ref_mode_ts.to_frequencyseries()
    freqs = ref_fs.sample_frequencies
    delta_f = ref_fs.delta_f

    self_modes_tilde = {}
    self_modes_dot_tilde = {}
    for ell, m in wfm.LM:
        h_ts = wfm.get_mode(ell, m, to_pycbc=True, delta_t_seconds=1 / 4096)
        h_ts.resize(max_len)
        h_tilde = h_ts.to_frequencyseries(delta_f=delta_f)
        self_modes_tilde[(ell, m)] = h_tilde
        h_dot_tilde = h_tilde.copy()
        h_dot_tilde.data *= 1j * 2 * np.pi * freqs
        self_modes_dot_tilde[(ell, m)] = h_dot_tilde

    def objective_function(x):
        time_shift, phi_c, alpha, beta, gamma = x[:5]
        alpha_jk_values = x[5:]
        alpha_jk_coeffs = dict(zip(alpha_jk_indices, alpha_jk_values))

        R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
        other_rot = other.rotated(R)

        total_inner_prod = 0.0
        total_norm1_sq = 0.0
        total_norm2_sq = 0.0

        common_modes = set(map(tuple, wfm.LM)) & set(map(tuple, other_rot.LM))

        self_modes_tilde_st = {}
        for ell, m in common_modes:
            h1_tilde = self_modes_tilde[(ell, m)]
            st_correction = np.zeros_like(h1_tilde.data, dtype=complex)

            for (j, k), alpha_jk in alpha_jk_coeffs.items():
                for p, q in wfm.LM:
                    G = scri.coupling_coefficients(
                        s_prime=-2,
                        l_prime=ell,
                        m_prime=m,
                        s1=0,
                        l1=j,
                        m1=k,
                        s2=-2,
                        l2=p,
                        m2=q,
                    )
                    if G == 0:
                        continue
                    h_dot_pq = self_modes_dot_tilde[(p, q)]
                    st_correction += alpha_jk * G * h_dot_pq.data

            h1_tilde_st = h1_tilde.copy()
            h1_tilde_st.data -= st_correction
            self_modes_tilde_st[(ell, m)] = h1_tilde_st

        for ell, m in common_modes:
            h1_tilde = self_modes_tilde_st[(ell, m)]
            h2_mode_ts = other_rot.get_mode(
                ell, m, to_pycbc=True, delta_t_seconds=1 / 4096
            )
            h2_mode_ts.resize(max_len)
            h2_tilde = h2_mode_ts.to_frequencyseries(delta_f=delta_f)

            temp_psd = psd.copy()
            temp_psd.resize(len(h1_tilde))

            h2_tilde *= np.exp(-1j * m * phi_c)
            h2_tilde.data *= np.exp(-2j * np.pi * freqs * time_shift)

            df = delta_f
            low_idx = int(f_lower / df) if f_lower else 0
            high_idx = int(np.ceil(f_upper / df)) if f_upper else len(temp_psd)

            h1 = h1_tilde.data[low_idx:high_idx]
            h2 = h2_tilde.data[low_idx:high_idx]
            psd_vals = temp_psd.data[low_idx:high_idx]
            psd_vals[np.isinf(psd_vals)] = 1.0

            total_norm1_sq += 4 * df * np.sum((np.abs(h1) ** 2) / psd_vals)
            total_norm2_sq += 4 * df * np.sum((np.abs(h2) ** 2) / psd_vals)
            total_inner_prod += 4 * df * np.sum((h1 * np.conj(h2)) / psd_vals)

        if total_norm1_sq == 0 or total_norm2_sq == 0:
            return 1.0

        overlap = np.abs(total_inner_prod) / np.sqrt(total_norm1_sq * total_norm2_sq)
        return 1.0 - overlap

    x0 = [0.0] * (5 + len(alpha_jk_indices))
    result = minimize(objective_function, x0, method="Nelder-Mead")
    return 1.0 - result.fun


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
