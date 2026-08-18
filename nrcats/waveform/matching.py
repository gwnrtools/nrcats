"""Standalone waveform matching and rotation helpers.

These are module-level functions (not bound to WaveformModes) so they can
be unit-tested and used independently of the class.
"""

from dataclasses import dataclass

import numpy as np
import spherical
from pycbc.types import TimeSeries as pycbc_TimeSeries
from sxs import TimeSeries as sxs_TimeSeries

#: Minimum number of cycles at the *lower edge* of the integration band that the
#: common time window must contain for the band to be resolvable.
#:
#: Below one cycle at the band edge ``pycbc.filter.match`` returns a bare NaN
#: (measured: the transition sits at ``T * f_lower ≈ 1``).  The default carries a
#: factor-of-two margin above that boundary.  It is deliberately *not* tuned to
#: make any particular sample pass: matches computed at ``K = 1`` and ``K = 2``
#: agree to <= 3e-3 in tests, because the aLIGO design PSD already suppresses the
#: band this criterion trims.
#:
#: This single constant supplies the mode dependence for free.  The band scales
#: as ``|m|/2``, so at fixed window length the criterion binds first for ``m=1``
#: and requires *twice* the duration that ``(2,2)`` needs -- which is why short
#: waveforms lose ``(2,1)`` while ``(2,2)``, ``(4,4)`` and ``(3,2)`` survive.
MIN_CYCLES_AT_BAND_EDGE = 2.0

#: Minimum number of resolved frequency bins between the (possibly raised) lower
#: cutoff and the upper cutoff.  Guards the case where the band is nominally
#: supported but the frequency resolution set by the overlap duration leaves too
#: few bins for the integral to mean anything.
MIN_BINS_IN_BAND = 8


@dataclass(frozen=True)
class ModeMatchResult:
    """Outcome of a single-mode match, with the reason it succeeded or failed.

    ``compute_mode_match`` returns only ``match`` and therefore cannot say why a
    NaN appeared.  A bare NaN is indistinguishable between "the mode carries no
    signal", "the two waveforms do not overlap in time" and "the requested band
    lies below where this waveform has support" -- three different problems with
    three different remedies.  This type keeps them separable.

    Attributes
    ----------
    match : float
        Match in [0, 1], or NaN when ``reason`` describes a failure.
    reason : str
        One of ``'ok'``, ``'band_raised'``, ``'no_overlap'``, ``'zero_norm'``,
        ``'insufficient_bins'``.  Both ``'ok'`` and ``'band_raised'`` carry a
        usable ``match``; the rest carry NaN.
    f_lower_requested : float
        The cutoff the caller asked for, in Hz.
    f_lower_used : float
        The cutoff actually integrated from, in Hz.  Differs from
        ``f_lower_requested`` only when ``reason == 'band_raised'``.
    overlap_seconds : float
        Duration of the common time window of the two waveforms.
    cycles_at_band_edge : float
        ``overlap_seconds * f_lower_requested`` -- how many cycles of the
        *requested* band edge fit in the window.  Values below
        ``MIN_CYCLES_AT_BAND_EDGE`` are what trigger a raise.
    n_bins_in_band : int
        Resolved frequency bins between ``f_lower_used`` and the upper cutoff.
    """

    match: float
    reason: str
    f_lower_requested: float
    f_lower_used: float
    overlap_seconds: float
    cycles_at_band_edge: float
    n_bins_in_band: int

    @property
    def is_usable(self) -> bool:
        """True when the match is a number rather than a failure code."""
        return self.reason in ("ok", "band_raised")


#: Fraction of the matched segment tapered at the START for complex modes.
#:
#: Complex modes cannot use LAL's ``TAPER_STARTEND``.  LAL tapers from the start
#: to the first *local maximum*: an oscillatory real series reaches one within a
#: half cycle, but a smooth rising amplitude envelope does not reach one until
#: the merger, so applying it to |h| windows a fifth of the inspiral away
#: (measured: 633 of 3551 samples against 157 for the real part, and by
#: different amounts for the NR and surrogate segments -- which drove a (2,2)
#: mismatch of 4.3e-1 against the 1.1e-3 the real path gives).
#:
#: A fixed-fraction window has none of those problems: it is deterministic
#: rather than data-dependent, it is *identical* for the two segments being
#: compared, and multiplying a complex array by a real window leaves the phase
#: untouched.
#:
#: **The taper is applied to the start only.**  Tapering the end is actively
#: harmful here.  These are NR and surrogate waveforms carrying a full merger
#: and ringdown, and a ringdown decays to zero on its own -- there is no
#: discontinuity at the end to suppress.  What an end taper does instead is
#: attenuate genuine merger-ringdown signal, which at M = 40 M_sun is where most
#: of the in-band signal-to-noise sits.  Only the start needs a window, because
#: the segment begins mid-inspiral at whatever amplitude the slice lands on.
TAPER_FRACTION = 0.05


def _start_window(n: int, fraction: float) -> np.ndarray:
    """Half-Tukey: raised-cosine rise over the first ``fraction`` of the segment,
    unity thereafter.  The end is left alone -- see :data:`TAPER_FRACTION`."""
    w = np.ones(n, dtype=np.float64)
    k = int(round(fraction * n))
    if k > 1:
        w[:k] = 0.5 * (1.0 - np.cos(np.pi * np.arange(k) / k))
    return w


try:
    from pycbc.waveform.utils import taper_timeseries as _pycbc_taper

    def _taper(ts, fraction: float = TAPER_FRACTION):
        """Taper the START of a real or complex mode series before matching.

        Neither branch touches the end.  These waveforms carry a full merger and
        ringdown, and a ringdown decays to zero on its own, so there is no
        end discontinuity to suppress -- an end taper only attenuates real
        merger-ringdown signal, which is where most of the in-band SNR sits at
        the masses analysed here.  Measured: with the end tapered, the (2,2)
        mismatch slid by 8-16x as the taper width was varied over 0.02-0.20;
        with the start alone it is constant to four significant figures over the
        same range.

        Real input uses LAL's ``TAPER_START``; complex input cannot (pycbc
        rejects non-float dtypes) and gets the equivalent half-Tukey rise.  Note
        the two are not identical: LAL picks its window width from the series'
        own extrema, so it windows the NR and model segments differently over
        the same interval, whereas the fixed-fraction window is by construction
        the same for both.  A difference introduced by the window is not a
        disagreement between the waveforms, so the complex path is the
        better-defined of the two.
        """
        arr = np.asarray(ts)
        if not np.iscomplexobj(arr):
            return _pycbc_taper(ts, tapermethod="start", return_lal=False)

        return pycbc_TimeSeries(
            arr * _start_window(len(arr), fraction),
            delta_t=ts.delta_t,
            epoch=ts.start_time,
        )

except ImportError:  # pragma: no cover - exercised only without pycbc

    def _taper(ts, fraction: float = TAPER_FRACTION):
        """Fallback taper: identical start-only window, no LAL dependency.

        The fallback deliberately matches the complex branch above rather than
        approximating LAL, so the two paths cannot silently disagree about what
        "tapered" means.
        """
        return pycbc_TimeSeries(
            np.asarray(ts) * _start_window(len(ts), fraction),
            delta_t=ts.delta_t,
            epoch=ts.start_time,
        )


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


def _fail(reason, f_req, overlap=float("nan"), cycles=float("nan")):
    """Build a failed ModeMatchResult carrying NaN and the reason."""
    return ModeMatchResult(
        match=float("nan"),
        reason=reason,
        f_lower_requested=f_req,
        f_lower_used=float("nan"),
        overlap_seconds=overlap,
        cycles_at_band_edge=cycles,
        n_bins_in_band=0,
    )


def _get_merger_index(arr) -> int:
    """Find the merger peak index robustly by searching backwards.

    This avoids initial junk radiation which can exceed the merger amplitude
    in higher modes. It finds the last prominent peak in the amplitude envelope.
    """
    from scipy.signal import find_peaks

    abs_arr = np.abs(arr)

    # Use max of the second half as a baseline to ensure
    # a massive junk peak in the first half doesn't inflate the threshold.
    baseline_max = float(np.max(abs_arr[len(abs_arr) // 2 :]))

    # Find peaks that stand out from their local background
    peaks, _ = find_peaks(abs_arr, prominence=0.1 * baseline_max)

    if len(peaks) > 0:
        # The LAST prominent peak is the merger, naturally ignoring early junk
        return int(peaks[-1])

    return int(np.argmax(abs_arr))


def _get_crosscorr_lag(arr_nr, arr_sur) -> int:
    """Find the optimal integer sample shift between two arrays via matched filter.

    Returns the lag required to shift `arr_sur` so it aligns with `arr_nr`.
    """
    from scipy.signal import correlate
    from scipy.signal.windows import tukey

    # Mildly taper just to prevent edge discontinuities from polluting the cross-correlation
    win_nr = tukey(len(arr_nr), alpha=0.1)
    win_sur = tukey(len(arr_sur), alpha=0.1)

    # Fast FFT cross-correlation
    corr = correlate(arr_nr * win_nr, arr_sur * win_sur, mode="full", method="fft")
    idx_max = int(np.argmax(np.abs(corr)))

    # Calculate the shift of arr_sur relative to arr_nr
    lag = idx_max - (len(arr_sur) - 1)
    return lag


def compute_mode_match_detailed(
    h_nr,
    h_sur,
    f_lower_mode: float,
    psd_name: str = "aLIGOZeroDetHighPower",
    f_upper=None,
    min_cycles: float = MIN_CYCLES_AT_BAND_EDGE,
    alignment: str = "peak",
) -> ModeMatchResult:
    """Match one NR mode against one model mode, reporting *why* on failure.

    Same computation as :func:`compute_mode_match`, but it returns a
    :class:`ModeMatchResult` instead of a bare float and it will **raise the
    lower cutoff** rather than return NaN when the requested band lies below
    what the common time window can resolve.

    Why the band is raised rather than the simulation discarded
    -----------------------------------------------------------
    The per-mode cutoff scales as ``|m|/2`` (see :func:`mode_f_lower`), which
    *lowers* the cutoff for ``m=1`` by a factor of two.  That mapping is
    physically correct but presumes the waveform extends to the lower
    frequency.  For a short waveform it does not: the integral then runs over a
    band where one input has no support, the normalisation degenerates, and
    ``pycbc.filter.match`` returns a bare NaN.  Measured, the transition sits at
    ``overlap * f_lower ≈ 1`` cycle.

    Discarding the simulation is the wrong remedy, because the failure is
    per-mode: at the duration where ``(2,1)`` fails, ``(2,2)``, ``(4,4)`` and
    ``(3,2)`` are still fine, and dropping the simulation throws those away
    too.  Raising the cutoff to the resolvable value instead reports the band
    that was *actually* integrated, and is close to value-neutral in practice
    because the detector PSD already suppresses the trimmed region.

    Parameters
    ----------
    h_nr, h_sur : pycbc.types.TimeSeries
        Complex NR and model mode time series, same ``delta_t``.
    f_lower_mode : float
        Requested low-frequency cutoff in Hz, normally ``mode_f_lower(f, m)``.
    psd_name : str, optional
        PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``).
    f_upper : float or None, optional
        Upper frequency cutoff in Hz (default: Nyquist).
    min_cycles : float, optional
        Cycles at the band edge the window must contain
        (default :data:`MIN_CYCLES_AT_BAND_EDGE`).  Pass ``0`` to disable the
        raise and reproduce the historical behaviour exactly.
    alignment : str, optional
        Method to align waveforms before matching: 'peak' (default) finds
        the merger robustly from the end; 'crosscorr' uses a fast-FFT matched
        filter over the full envelope to find maximum phase coherence.

    Returns
    -------
    ModeMatchResult
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
        return _fail("zero_norm", f_lower_mode)

    arr_nr = np.asarray(h_nr)
    arr_sur = np.asarray(h_sur)

    if alignment == "peak":
        idx_peak_nr = _get_merger_index(arr_nr)
        idx_peak_sur = _get_merger_index(arr_sur)

        samples_before = min(idx_peak_nr, idx_peak_sur)
        samples_after = min(len(arr_nr) - idx_peak_nr, len(arr_sur) - idx_peak_sur)

        start_nr = idx_peak_nr - samples_before
        start_sur = idx_peak_sur - samples_before
        end_nr = idx_peak_nr + samples_after
        end_sur = idx_peak_sur + samples_after

    elif alignment == "crosscorr":
        lag = _get_crosscorr_lag(arr_nr, arr_sur)
        start_nr = max(0, lag)
        start_sur = max(0, -lag)
        end_nr = min(len(arr_nr), lag + len(arr_sur))
        end_sur = min(len(arr_sur), len(arr_nr) - lag)

    else:
        raise ValueError(f"Unknown alignment method: {alignment}")

    n_overlap = end_nr - start_nr

    if n_overlap <= 1:
        return _fail(
            "no_overlap", f_lower_mode, overlap=n_overlap * float(h_nr.delta_t)
        )

    overlap = n_overlap * float(h_nr.delta_t)
    cycles = overlap * f_lower_mode

    # Slice the underlying arrays to the exact common window
    slice_nr = slice(start_nr, end_nr)
    slice_sur = slice(start_sur, end_sur)

    # 4. Wrap the slices back into pycbc TimeSeries
    # We can safely discard the epoch because we've manually aligned the physical window,
    # and pycbc_match will optimize the exact time-shift regardless of the epoch value.
    h_nr_sliced = pycbc_TimeSeries(arr_nr[slice_nr], delta_t=h_nr.delta_t)
    h_sur_sliced = pycbc_TimeSeries(arr_sur[slice_sur], delta_t=h_sur.delta_t)

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

    # Raise the cutoff to what this window can actually resolve.  max() makes
    # this a strict no-op wherever the requested band was already supported, so
    # results for well-sampled waveforms are unchanged bit-for-bit.
    f_floor = (min_cycles / overlap) if min_cycles > 0 else 0.0
    f_lower_used = max(f_lower_mode, f_floor)
    reason = "band_raised" if f_lower_used > f_lower_mode else "ok"

    f_hi = f_upper if f_upper is not None else 0.5 / float(h1.delta_t)
    n_bins = int((f_hi - f_lower_used) / delta_f)
    if n_bins < MIN_BINS_IN_BAND:
        return _fail("insufficient_bins", f_lower_mode, overlap, cycles)

    psd = from_string(psd_name, length_f, delta_f, low_freq_cutoff=f_lower_used)

    # pycbc.filter.match builds a real-to-complex FFT internally and raises
    # "For C2C FFT, len(outvec) must be nbatch*size" if handed a complex series,
    # so the complex mode cannot be passed through to the filter itself.
    #
    # It does not need to be.  The complex series is what everything *upstream*
    # requires -- the merger peak, the cross-correlation lag and the taper are
    # all defined on the amplitude envelope |h_lm|, which the real part alone
    # cannot give (Re h_lm passes through zero every half cycle).  The filter is
    # the one step that is indifferent: for h_lm = A e^{-i phi}, Re(h_lm) is
    # A cos(phi), and match() maximises over a constant phase offset, so the
    # whole family A cos(phi + phi_0) -- every real projection of the mode --
    # returns the same value.  Taking the real part here therefore discards
    # nothing the match could have used.
    h1_re = h1.real() if h1.kind == "complex" else h1
    h2_re = h2.real() if h2.kind == "complex" else h2

    mm, _ = pycbc_match(
        h1_re,
        h2_re,
        psd=psd,
        low_frequency_cutoff=f_lower_used,
        high_frequency_cutoff=f_upper,
    )
    return ModeMatchResult(
        match=float(mm),
        reason=reason,
        f_lower_requested=f_lower_mode,
        f_lower_used=f_lower_used,
        overlap_seconds=overlap,
        cycles_at_band_edge=cycles,
        n_bins_in_band=n_bins,
    )


def compute_mode_match(
    h_nr,
    h_sur,
    f_lower_mode: float,
    psd_name: str = "aLIGOZeroDetHighPower",
    f_upper=None,
    min_cycles: float = MIN_CYCLES_AT_BAND_EDGE,
    alignment: str = "peak",
) -> float:
    """Compute the noise-weighted match between one NR and one model mode.

    Thin wrapper over :func:`compute_mode_match_detailed` that returns only the
    numeric match, for callers that do not need the failure reason.  The
    signature and return type are unchanged from earlier versions.

    Both inputs should be the *complex* strain mode, sampled at the same
    ``delta_t``.  The function pads to the next power-of-two, builds a PSD at
    the matching frequency resolution, and calls ``pycbc.filter.match()``.

    Parameters
    ----------
    h_nr : pycbc.types.TimeSeries
        Complex NR mode time series.
    h_sur : pycbc.types.TimeSeries
        Complex surrogate mode time series.
    f_lower_mode : float
        Low-frequency cutoff for this mode in Hz.
        Use ``f_lower * |m| / 2`` (GW frequency scales as |m| × f_orbital).
    psd_name : str, optional
        PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``).
    f_upper : float or None, optional
        Upper frequency cutoff in Hz (default: Nyquist).
    min_cycles : float, optional
        Cycles at the band edge the common window must contain before the
        cutoff is raised (default :data:`MIN_CYCLES_AT_BAND_EDGE`).  Pass ``0``
        to reproduce the historical behaviour exactly.
    alignment : str, optional
        Method used to align waveforms before matching. 'peak' (default) finds
        the merger peak robustly from the end; 'crosscorr' uses a fast-FFT
        matched filter over the full envelope to find the maximum phase coherence.

    Returns
    -------
    float
        Match in [0, 1], or ``float('nan')`` if the mode carries no signal, the
        two waveforms do not overlap in time, or the band cannot be resolved.
        Use :func:`compute_mode_match_detailed` to tell those cases apart.

    See Also
    --------
    compute_mode_match_detailed : same computation, with the reason attached.
    """
    return compute_mode_match_detailed(
        h_nr,
        h_sur,
        f_lower_mode,
        psd_name=psd_name,
        f_upper=f_upper,
        min_cycles=min_cycles,
        alignment=alignment,
    ).match


def compute_phase_diff_per_cycle(h_nr, h_sur, alignment: str = "peak") -> tuple:
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
    alignment : str, optional
        Method to align waveforms before computing phase diff. 'peak' (default)
        or 'crosscorr'.

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

    if alignment == "peak":
        idx_peak_nr = _get_merger_index(arr_nr)
        idx_peak_sur = _get_merger_index(arr_sur)

        samples_before = min(idx_peak_nr, idx_peak_sur)
        samples_after = min(len(arr_nr) - idx_peak_nr, len(arr_sur) - idx_peak_sur)

        start_nr = idx_peak_nr - samples_before
        start_sur = idx_peak_sur - samples_before
        end_nr = idx_peak_nr + samples_after

    elif alignment == "crosscorr":
        lag = _get_crosscorr_lag(arr_nr, arr_sur)
        start_nr = max(0, lag)
        start_sur = max(0, -lag)
        end_nr = min(len(arr_nr), lag + len(arr_sur))

    else:
        raise ValueError(f"Unknown alignment method: {alignment}")

    n = end_nr - start_nr
    if n < 2:
        return float("nan"), float("nan")

    i_nr_s = start_nr
    i_sur_s = start_sur

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
