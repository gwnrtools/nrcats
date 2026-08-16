"""Standalone waveform matching and rotation helpers.

These are module-level functions (not bound to WaveformModes) so they can
be unit-tested and used independently of the class.
"""

import numpy as np
import spherical
from pycbc.types import TimeSeries as pycbc_TimeSeries
from sxs import TimeSeries as sxs_TimeSeries

try:
    from pycbc.waveform.utils import taper_timeseries as _pycbc_taper

    def _taper(ts):
        return _pycbc_taper(ts, tapermethod="startend", return_lal=False)

except ImportError:
    from scipy.signal.windows import tukey as _tukey

    def _taper(ts):
        win = _tukey(len(ts), alpha=0.2).astype(np.float64)
        data = np.array(ts) * win
        return pycbc_TimeSeries(data, delta_t=ts.delta_t, epoch=ts.start_time)


def apply_wigner_rotation_to_mode_dict(mode_dict, R, ell_max=4):
    """Apply a Wigner rotation to a dictionary of spherical harmonic modes.

    This is useful for rotating the output of ``gwsurrogate`` or
    ``pycbc.waveform.get_td_waveform_modes`` (which return dicts) into the
    NR source frame before computing mode-by-mode matches.

    The rotation is applied mode-by-mode via Wigner D-matrices:

        h'_{ℓm}(t) = Σ_{m'} D^{(ℓ)}_{m'm}(R) h_{ℓm'}(t)

    where R ∈ SO(3) is a unit quaternion and D^{(ℓ)} is the (2ℓ+1)×(2ℓ+1)
    Wigner D-matrix for angular momentum ℓ.

    Parameters
    ----------
    mode_dict : dict
        Keys are ``(l, m)`` integer tuples; values are complex
        ``pycbc.types.TimeSeries`` objects (or 1-D numpy arrays of matching
        length).
    R : quaternionic.array
        Unit quaternion representing the rotation.
    ell_max : int, optional
        Maximum ℓ to include (default 4).

    Returns
    -------
    dict
        Rotated mode dictionary with the same ``(l, m)`` keys.
    """
    wigner = spherical.Wigner(ell_max)

    by_ell = {}
    for (ell, m), val in mode_dict.items():
        if ell > ell_max:
            continue
        by_ell.setdefault(ell, {})[m] = val

    rotated = {}
    for ell, m_dict in by_ell.items():
        m_vals = list(range(-ell, ell + 1))
        first = next(iter(m_dict.values()))
        is_timeseries = hasattr(first, "delta_t")
        n = len(first)

        block = np.zeros((n, 2 * ell + 1), dtype=complex)
        for i, mv in enumerate(m_vals):
            if mv in m_dict:
                block[:, i] = np.asarray(m_dict[mv])

        D = wigner.D(R, ell)
        rotated_block = block @ D.T

        for i, mv in enumerate(m_vals):
            if is_timeseries:
                rotated[(ell, mv)] = type(first)(
                    rotated_block[:, i], delta_t=first.delta_t, epoch=first.start_time
                )
            else:
                rotated[(ell, mv)] = rotated_block[:, i]

    return rotated


def load_psd(
    f_lower: float,
    delta_t: float,
    waveform_length_seconds: float,
    psd_name: str = "aLIGOZeroDetHighPower",
):
    """Load a named analytic PSD sampled to match a waveform's frequency grid.

    Parameters
    ----------
    f_lower : float
        Low-frequency cutoff in Hz.
    delta_t : float
        Time step of the waveforms in seconds (sets the Nyquist limit).
    waveform_length_seconds : float
        Duration of the longest waveform in seconds (sets frequency resolution).
    psd_name : str, optional
        PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``).

    Returns
    -------
    pycbc.types.FrequencySeries
    """
    from pycbc.psd import from_string

    n_samples = int(waveform_length_seconds / delta_t)
    n_fft = 1
    while n_fft < n_samples:
        n_fft <<= 1

    delta_f = 1.0 / (n_fft * delta_t)
    length_f = n_fft // 2 + 1

    return from_string(psd_name, length_f, delta_f, low_freq_cutoff=f_lower)


def compute_mode_match(
    h_nr,
    h_sur,
    f_lower_mode: float,
    psd_name: str = "aLIGOZeroDetHighPower",
    f_upper=None,
) -> float:
    """Compute the noise-weighted match between one NR and one surrogate mode.

    Both inputs should be the *real part* of the complex strain mode
    (h₊ component), sampled at the same ``delta_t``.  The function pads to
    the next power-of-two, builds a PSD at the matching frequency resolution,
    and calls ``pycbc.filter.match()``.

    Parameters
    ----------
    h_nr : pycbc.types.TimeSeries
        Real-valued NR mode time series.
    h_sur : pycbc.types.TimeSeries
        Real-valued surrogate mode time series.
    f_lower_mode : float
        Low-frequency cutoff for this mode in Hz.
        Use ``f_lower * |m| / 2`` (GW frequency scales as |m| × f_orbital).
    psd_name : str, optional
        PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``).
    f_upper : float or None, optional
        Upper frequency cutoff in Hz (default: Nyquist).

    Returns
    -------
    float
        Match in [0, 1], or ``float('nan')`` if either waveform has zero norm.
    """
    from pycbc.filter import match as pycbc_match
    from pycbc.psd import from_string

    # np.asarray, not np.array: pycbc TimeSeries wraps a numpy buffer whose
    # __array__ predates the numpy-2 copy keyword, so np.array() emits a
    # DeprecationWarning and will fail outright once numpy removes the shim.
    # Nothing here mutates the result, so a view is correct as well as cheaper.
    if (
        float(np.max(np.abs(np.asarray(h_nr)))) < 1e-50
        or float(np.max(np.abs(np.asarray(h_sur)))) < 1e-50
    ):
        return float("nan")

    t_start = max(float(h_nr.start_time), float(h_sur.start_time))
    t_end = min(float(h_nr.end_time), float(h_sur.end_time))

    if t_end <= t_start:
        return float("nan")

    h_nr_sliced = h_nr.time_slice(t_start, t_end)
    h_sur_sliced = h_sur.time_slice(t_start, t_end)

    h1_tapered = _taper(h_nr_sliced)
    h2_tapered = _taper(h_sur_sliced)

    raw_len = max(len(h1_tapered), len(h2_tapered))
    n_fft = 1
    while n_fft < raw_len:
        n_fft <<= 1

    h1 = h1_tapered.copy()
    h1.resize(n_fft)
    h2 = h2_tapered.copy()
    h2.resize(n_fft)

    delta_f = 1.0 / (n_fft * h1.delta_t)
    length_f = n_fft // 2 + 1
    psd = from_string(psd_name, length_f, delta_f, low_freq_cutoff=f_lower_mode)

    mm, _ = pycbc_match(
        h1,
        h2,
        psd=psd,
        low_frequency_cutoff=f_lower_mode,
        high_frequency_cutoff=f_upper,
    )
    return float(mm)


def compute_phase_diff_per_cycle(h_nr, h_sur) -> tuple:
    """Compute accumulated phase difference per GW cycle over the common window.

    Both inputs are the *complex* mode time series (h_lm = h+ - i h×).
    The two waveforms are trimmed to their shared time window (both should have
    epoch set so t=0 is at peak amplitude), then the total accumulated phase of
    each is computed from the unwrapped angle.

    The metric returned is::

        phase_diff_per_cycle = |ΔΦ_NR - ΔΦ_sur| / N_cycles_NR   [rad / cycle]

    where ``ΔΦ = |φ(t_end) - φ(t_start)|`` is the total phase evolved and
    ``N_cycles_NR = ΔΦ_NR / (2π)``.

    Parameters
    ----------
    h_nr : pycbc.types.TimeSeries
        Complex NR mode time series.
    h_sur : pycbc.types.TimeSeries
        Complex surrogate mode time series.

    Returns
    -------
    tuple[float, float]
        ``(phase_diff_per_cycle, n_cycles_nr)``.
        Returns ``(nan, nan)`` if either waveform has zero norm or the common
        window contains fewer than 2 samples.
    """
    # np.asarray for the reason given in compute_mode_match: np.array() on a
    # pycbc TimeSeries triggers the numpy-2 __array__ copy-keyword warning.
    arr_nr = np.asarray(h_nr)
    arr_sur = np.asarray(h_sur)

    if float(np.max(np.abs(arr_nr))) < 1e-50 or float(np.max(np.abs(arr_sur))) < 1e-50:
        return float("nan"), float("nan")

    dt = float(h_nr.delta_t)
    t_start = max(float(h_nr.start_time), float(h_sur.start_time))
    t_end = min(float(h_nr.end_time), float(h_sur.end_time))

    if t_end <= t_start:
        return float("nan"), float("nan")

    i_nr_s = max(0, int(round((t_start - float(h_nr.start_time)) / dt)))
    i_nr_e = min(len(arr_nr), int(round((t_end - float(h_nr.start_time)) / dt)) + 1)
    i_sur_s = max(0, int(round((t_start - float(h_sur.start_time)) / dt)))
    i_sur_e = min(len(arr_sur), int(round((t_end - float(h_sur.start_time)) / dt)) + 1)

    n = min(i_nr_e - i_nr_s, i_sur_e - i_sur_s)
    if n < 2:
        return float("nan"), float("nan")

    phi_nr = np.unwrap(np.angle(arr_nr[i_nr_s : i_nr_s + n]))
    phi_sur = np.unwrap(np.angle(arr_sur[i_sur_s : i_sur_s + n]))

    delta_phi_nr = abs(phi_nr[-1] - phi_nr[0])
    delta_phi_sur = abs(phi_sur[-1] - phi_sur[0])

    n_cycles_nr = delta_phi_nr / (2.0 * np.pi)
    if n_cycles_nr < 0.5:
        return float("nan"), float("nan")

    # |ΔΦ_NR − ΔΦ_sur| measures the difference in total accumulated phase
    # (i.e. cycle-count error) over the common window.  Taking differences
    # within each waveform removes any constant initial-phase offset, so the
    # result is independent of coalescence-phase convention.  It is NOT the
    # same as the phase residual after match()-optimal time alignment; it uses
    # the absolute time alignment (both waveforms referenced to t=0 at peak).
    phase_diff_per_cycle = abs(delta_phi_nr - delta_phi_sur) / n_cycles_nr
    return float(phase_diff_per_cycle), float(n_cycles_nr)


def mode_f_lower(f_lower: float, em: int) -> float:
    """Return the GW frequency cutoff for mode (ell, m).

    GW frequency for the (ell, |m|) mode is approximately |m| times the
    orbital frequency: ``f_gw ≈ |m| * f_orbital = |m| * f_lower / 2``
    (since the (2,2) mode has ``f_gw = 2 * f_orbital``).

    Parameters
    ----------
    f_lower : float
        GW frequency of the (2,2) mode in Hz (= 2 × orbital frequency).
        This is what ``CatalogBase.get_parameters()`` returns as ``f_lower``.
    em : int
        Azimuthal mode number m.  For m=0 the mode carries no oscillatory
        GW power at a well-defined frequency; ``f_lower`` is returned as a
        conservative lower bound but the result should not be interpreted as
        a physically meaningful frequency cutoff for that mode.

    Returns
    -------
    float
        Mode-specific GW frequency cutoff in Hz.
    """
    return f_lower * abs(em) / 2.0 if em != 0 else f_lower


def interpolate_in_amp_phase(obj, new_time, k=3, kind=None):
    """Interpolate in amplitude and phase using a variety of methods.

    Parameters
    ----------
    obj : sxs.TimeSeries
        Complex waveform time series.
    new_time : array_like
        New time axis to interpolate onto.
    k : int, optional
        Spline order for ``InterpolatedUnivariateSpline`` (default 3).
    kind : str, optional
        Alternative interpolation: ``'linear'``, ``'quadratic'``, ``'cubic'``,
        or ``'CubicSpline'``.  When specified, ``k`` is ignored.

    Returns
    -------
    sxs.TimeSeries
        Interpolated complex waveform on ``new_time``.
    """
    from waveformtools.waveformtools import interp_resam_wfs

    resam_data = interp_resam_wfs(
        wavf_data=np.array(obj),
        old_taxis=obj.time,
        new_taxis=new_time,
        k=k,
        kind=kind,
    )

    resam_data = sxs_TimeSeries(resam_data, new_time)

    metadata = obj._metadata.copy()
    metadata["time"] = new_time
    metadata["time_axis"] = obj.time_axis

    return type(obj)(resam_data, **metadata)
