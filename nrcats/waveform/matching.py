"""Standalone waveform matching and rotation helpers.

These are module-level functions (not bound to WaveformModes) so they can
be unit-tested and used independently of the class.
"""

from dataclasses import dataclass

import numpy as np
import spherical
from pycbc.types import FrequencySeries as pycbc_FrequencySeries
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
        ``'insufficient_bins'``, ``'no_psd_support'``.  Both ``'ok'`` and ``'band_raised'`` carry a
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
    from .detectors import psd_from_name

    n_samples = int(waveform_length_seconds / delta_t)
    n_fft = 1
    while n_fft < n_samples:
        n_fft <<= 1

    delta_f = 1.0 / (n_fft * delta_t)
    length_f = n_fft // 2 + 1

    return psd_from_name(psd_name, length_f, delta_f, f_lower)


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
    psd=None,
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
    psd : pycbc.types.FrequencySeries or None, optional
        A pre-built one-sided PSD.  It is **resampled** onto the frequency grid
        this function actually integrates on, so its ``delta_f`` need not match
        the padded segment -- the caller cannot generally know that resolution
        in advance, since it depends on the common window found here.  Outside
        the supplied range the PSD is taken as infinite, excluding those bins.
        When ``None`` (default) the PSD is built from ``psd_name``.
    psd_name : str, optional
        PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``).  Ignored
        when ``psd`` is given.
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
    from .detectors import psd_from_name

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

    grid_f = np.arange(length_f) * delta_f
    resampled = None
    psd_floor = 0.0
    if psd is not None:
        # Resample onto this grid rather than assuming it matches, exactly as
        # _inverse_psd does for the strain match.  inf outside the supplied
        # range so those bins drop out of the integral instead of extrapolating.
        src_f = np.arange(len(psd)) * float(psd.delta_f)
        resampled = np.interp(grid_f, src_f, np.asarray(psd), left=np.inf, right=np.inf)
        # A supplied PSD is commonly built with its own low_freq_cutoff, below
        # which PyCBC leaves zeros.  Those are absent information, not zero
        # noise, and dividing by them yields a bare NaN for exactly the modes
        # whose |m|/2 cutoff reaches furthest down.  Treat non-positive and
        # non-finite bins as infinite -- the _inverse_psd convention -- and
        # raise the cutoff to where the PSD actually has support, so the band
        # reported is the band integrated.
        usable = np.isfinite(resampled) & (resampled > 0)
        resampled = np.where(usable, resampled, np.inf)
        in_band = usable & (grid_f > 0)
        if not in_band.any():
            return _fail("no_psd_support", f_lower_mode, overlap, cycles)
        psd_floor = float(grid_f[in_band][0])

    f_lower_used = max(f_lower_mode, f_floor, psd_floor)
    reason = "band_raised" if f_lower_used > f_lower_mode else "ok"

    f_hi = f_upper if f_upper is not None else 0.5 / float(h1.delta_t)
    n_bins = int((f_hi - f_lower_used) / delta_f)
    if n_bins < MIN_BINS_IN_BAND:
        return _fail("insufficient_bins", f_lower_mode, overlap, cycles)

    if resampled is None:
        psd_used = psd_from_name(psd_name, length_f, delta_f, f_lower_used)
    else:
        psd_used = pycbc_FrequencySeries(resampled, delta_f=delta_f)

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

    # subsample_interpolation=True: without it match() reports the largest
    # sample of the SNR time series, quantizing the time shift to one sample.
    # The error that introduces grows as the mismatch shrinks -- a sharper peak
    # is worse approximated by its nearest sample -- so it contaminates exactly
    # the small-mismatch regime the catalog comparison lives in.  It is the
    # mechanism behind the sample-rate sensitivity of findings 5k: the SXS
    # (2,2) median moved by a factor of 2.5 between 4096 and 16384 Hz with
    # nothing near Nyquist, and the approach was not even monotone.  The
    # sphere-averaged matches maximize over time themselves and get the same
    # correction from _interpolated_peak_abs(), so the two remain comparable.
    mm, _ = pycbc_match(
        h1_re,
        h2_re,
        psd=psd_used,
        low_frequency_cutoff=f_lower_used,
        high_frequency_cutoff=f_upper,
        subsample_interpolation=True,
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
    psd=None,
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
    psd : pycbc.types.FrequencySeries or None, optional
        Pre-built one-sided PSD, resampled onto the grid actually integrated.
        When ``None`` (default) it is built from ``psd_name``.
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
        psd=psd,
        psd_name=psd_name,
        f_upper=f_upper,
        min_cycles=min_cycles,
        alignment=alignment,
    ).match


# Cycles the window must carry before the per-cycle phase metric means anything.
# The metric divides by the cycle count, so a short window turns a small phase
# difference into a large rate; below a few cycles the quantity is a ratio of two
# poorly-determined numbers rather than a drift.  See the docstring of
# compute_phase_diff_per_cycle for the measurement that set this.
MIN_CYCLES_FOR_PHASE = 3.0

# How monotone the NR phase must be before a "net accumulated phase" means
# anything: the net advance divided by the total variation, which is 1 for a
# clean chirp.  Set from measurement rather than taste -- see
# compute_phase_diff_per_cycle.
MIN_PHASE_MONOTONICITY = 0.9

# The phase of a sample is only defined where the waveform carries signal.
# Samples below this fraction of each series' own peak amplitude are trimmed
# from both ends of the window before any phase is read.  Without it the window
# admits leading zero-padding and a post-merger tail decaying into numerical
# noise -- RIT:eBBH:1133-n100-ecc reaches 9e-26 of its peak before the signal
# and 1e-13 after it -- whose angle wanders, so a clean 35-cycle chirp read as
# 0.73 monotone and was refused.  1e-3 is the paper's mode-admissibility floor.
# With the window ending at the (2,2) peak only the leading edge is trimmed.
PHASE_AMP_FLOOR = 1e-3


def compute_phase_diff_per_cycle(
    h_nr,
    h_sur,
    alignment: str = "epoch",
    min_cycles: float = MIN_CYCLES_FOR_PHASE,
    t_stop: float | None = 0.0,
) -> tuple:
    """Accumulated inspiral phase difference per GW cycle.

    Both inputs are the *complex* mode time series (h_lm = h+ - i h×), and both
    must be referenced to a common ``t = 0`` -- the (2,2) amplitude peak for
    every mode, which is what ``WaveformModes.get_mode()`` and the surrogate
    generator set.  The metric returned is::

        phase_diff_per_cycle = |ΔΦ_NR - ΔΦ_sur| / N_cycles_NR   [rad / cycle]

    with ``ΔΦ = |φ(t_end) - φ(t_start)|`` the net phase evolved over the window,
    which runs from the start of the common span to ``t_stop`` -- by default
    the (2,2) peak, so the quantity is an inspiral phase drift.  The numerator
    is ``δφ(t_end) - δφ(t_start)``, the change in the phase difference across
    the window, so a constant offset -- a coalescence-phase convention --
    cancels.  It is NOT the phase residual after match()-optimal time
    alignment.

    What this function got wrong until 2026-09-18
    ---------------------------------------------
    Note first that, since ``N_cycles_NR = ΔΦ_NR / 2π``, the metric is
    identically ``2π |1 - ΔΦ_sur/ΔΦ_NR|``: 2π times the *fractional* difference
    in accumulated phase, unbounded above, and divergent as the denominator
    goes to zero.  Three things then went wrong, two in the denominator and
    one in the window:

    1. ``N_cycles`` was taken from the *net* endpoint difference, which counts
       cycles only while the phase is monotone.  On ``RIT:eBBH:1132-n100-ecc``
       -- a single close encounter -- the net difference is 3.14 rad against a
       total variation of 96.9 rad, so ``N_cycles`` came out at 0.50, passing
       the old 0.5-cycle floor by a hair, and the metric reported 288 rad/cycle
       for an accumulated-phase ratio of 46.9.  The cycle count now comes from
       the total variation, which agrees with the net difference exactly for a
       monotone chirp (``SXS:BBH:2348``: 275.13 rad both ways) and differs by a
       factor of 30 where the phase wanders -- which, item 3 found, it did
       only after merger: the run carries 0.71 cycles before its peak.
    2. The window was cut around each mode's *own* amplitude peak, found by
       :func:`_get_merger_index`, which takes the last prominent peak in order
       to step over junk radiation.  On a burst-like waveform the last
       prominent peak is late-time structure: for the same simulation it landed
       at t = +0.2373 s where the epoch puts the peak at 0.  The window is now
       the intersection of the two series *in time*, which is mode-independent
       and needs no peak search.
    3. The window ran to the end of the common span, through merger and
       ringdown, while the denominator counts cycles that are almost all
       inspiral.  The numerator was therefore mostly post-merger: on the (2,2)
       mode, with the window as cut before the amplitude floor, 98% of
       |ΔΦ_NR - ΔΦ_sur| for SXS:BBH:2348 accrued after t = 0, 72% for
       MAYA1022, 94% for GT0357, 69% for RIT:BBH:0555-n120-id1.  A quantity
       described as secular inspiral drift was mostly a ringdown phase
       difference spread over the inspiral cycles -- SXS:BBH:2348 reads 0.019
       rad/cycle over the whole span and 0.0004 up to the peak.  The window
       now stops at ``t_stop = 0``.  Ringdown agreement is what the match
       measures; this metric is kept for the inspiral.

    ``alignment='peak'`` preserves the superseded behaviour, because
    ``compare_alignment_backends.py`` exists to compare these conventions and
    because it is what produced the phase columns of the frozen 2026-08 results.

    Parameters
    ----------
    h_nr, h_sur : pycbc.types.TimeSeries
        Complex NR and model mode time series, same ``delta_t``, both
        referenced to a common ``t = 0``.
    alignment : str, optional
        ``'epoch'`` (default) cuts the window from the epochs; ``'peak'`` from
        each series' own amplitude peak; ``'crosscorr'`` from the cross-
        correlation lag.
    min_cycles : float, optional
        Refuse to report a value below this many cycles in the window
        (default :data:`MIN_CYCLES_FOR_PHASE`).  Pass ``0`` to disable.
    t_stop : float or None, optional
        ``'epoch'`` only: end the window here, in seconds on the common time
        axis (default ``0.0``, the (2,2) peak).  ``None`` runs to the end of
        the common span, which is the superseded behaviour of item 3 and is
        kept so the ringdown share can be measured.

    Returns
    -------
    tuple[float, float]
        ``(phase_diff_per_cycle, n_cycles_nr)``.  The value is ``nan`` when
        either waveform has zero norm, when the window holds fewer than two
        samples, or when it carries fewer than ``min_cycles`` cycles -- and in
        that last case ``n_cycles_nr`` is still returned, so the reason for the
        refusal is visible in the output rather than being another bare NaN.
    """
    # np.asarray for the reason given in compute_mode_match: np.array() on a
    # pycbc TimeSeries triggers the numpy-2 __array__ copy-keyword warning.
    arr_nr = np.asarray(h_nr)
    arr_sur = np.asarray(h_sur)

    if float(np.max(np.abs(arr_nr))) < 1e-50 or float(np.max(np.abs(arr_sur))) < 1e-50:
        return float("nan"), float("nan")

    if alignment == "epoch":
        dt = float(h_nr.delta_t)
        t_start = max(float(h_nr.start_time), float(h_sur.start_time))
        t_end = min(float(h_nr.end_time), float(h_sur.end_time))
        if t_stop is not None:
            t_end = min(t_end, float(t_stop))
        if t_end <= t_start:
            return float("nan"), float("nan")
        start_nr = max(0, int(round((t_start - float(h_nr.start_time)) / dt)))
        start_sur = max(0, int(round((t_start - float(h_sur.start_time)) / dt)))
        n_nr = min(len(arr_nr) - start_nr, int(round((t_end - t_start) / dt)) + 1)
        n_sur = min(len(arr_sur) - start_sur, int(round((t_end - t_start) / dt)) + 1)
        end_nr = start_nr + min(n_nr, n_sur)

    elif alignment == "peak":
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

    seg_nr = arr_nr[start_nr : start_nr + n]
    seg_sur = arr_sur[start_sur : start_sur + n]

    # Read phase only where both series carry signal: an angle taken from a
    # zero-amplitude sample is noise, and it enters both the net advance and
    # the traversed-cycle count.  See PHASE_AMP_FLOOR.
    amp_nr = np.abs(seg_nr)
    amp_sur = np.abs(seg_sur)
    live = np.flatnonzero(
        (amp_nr >= PHASE_AMP_FLOOR * amp_nr.max())
        & (amp_sur >= PHASE_AMP_FLOOR * amp_sur.max())
    )
    if len(live) < 2:
        return float("nan"), float("nan")
    lo, hi = int(live[0]), int(live[-1]) + 1

    phi_nr = np.unwrap(np.angle(seg_nr[lo:hi]))
    phi_sur = np.unwrap(np.angle(seg_sur[lo:hi]))

    delta_phi_nr = abs(phi_nr[-1] - phi_nr[0])
    delta_phi_sur = abs(phi_sur[-1] - phi_sur[0])

    # Cycles actually traversed, not the net phase change: the two agree for a
    # monotone chirp and differ by a factor of 30 where the phase wanders.
    traversed = float(np.abs(np.diff(phi_nr)).sum())
    n_cycles_nr = traversed / (2.0 * np.pi)
    if n_cycles_nr < min_cycles:
        return float("nan"), n_cycles_nr

    # A net phase advance is only "accumulated phase" while the phase advances.
    # The numerator compares net advances, so where the NR phase wanders the
    # comparison is between two quantities that do not mean the same thing, and
    # the honest answer is a refusal.  Over the whole common span this refused
    # GT0420 (0.647) and RIT:eBBH:1132-n100-ecc (0.032), but the wandering was
    # all post-merger: on the window that ends at t = 0 the (2,2) mode reads
    # 1.000 for SXS:BBH:2348, MAYA1022, GT0357, GT0420, RIT:BBH:0555-n120-id1
    # and three RIT eBBH runs, eccentric orbits included, at 16384 Hz.  The
    # guard is kept as a backstop for leading junk radiation and t_stop=None.
    monotonicity = delta_phi_nr / traversed if traversed > 0 else 0.0
    if monotonicity < MIN_PHASE_MONOTONICITY:
        return float("nan"), n_cycles_nr

    phase_diff_per_cycle = abs(delta_phi_nr - delta_phi_sur) / n_cycles_nr
    return float(phase_diff_per_cycle), n_cycles_nr


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


# ---------------------------------------------------------------------------
# Frame-maximised strain match
# ---------------------------------------------------------------------------
#
# Every match above is computed one mode at a time, and ``pycbc.filter.match``
# maximises over a constant phase *independently for each mode*.  Six modes
# therefore carry six free phases, and the relative phase between modes -- which
# is physical -- is discarded before any number is produced.  Two mode sets can
# agree perfectly mode by mode and still describe different waveforms.
#
# Measured: SXS:BBH:0201 has a (2,2) mismatch of 6.6e-5 per mode and 5.7e-2 on
# the coherent strain, because the two mode sets differ by a frame offset
#
#     h_lm^A  =  exp[ i ( alpha + m beta ) ]  h_lm^B                        (*)
#
# with alpha ~ pi.  Fitting (*) over 117 simulations gives coherence 1.0000 in
# every one, so (*) is not an approximation: to the accuracy of the data, two
# mode sets of the same binary differ by exactly these two angles and nothing
# else.  Both are unobservable nuisance parameters and both must be maximised
# over before a strain-level disagreement means anything.


#: Number of ``beta`` samples in the coarse scan.  2pi/720 = 8.7e-3 rad, refined
#: afterwards by a local grid, so this only has to be fine enough not to miss
#: the global peak of a trigonometric polynomial of degree ``max|m|`` (<= 4 for
#: the modes this package carries, hence at most 8 maxima in [0, 2pi)).
STRAIN_MATCH_N_BETA = 720

#: Cap on the number of time samples entered into the full (beta, t_c) grid
#: after pruning.  The pruning below is exact; this is the fallback when it
#: fails to bite, and 2e4 samples at 4096 Hz is a +-2.4 s window.
_MAX_TC_SAMPLES = 20000


@dataclass(frozen=True)
class StrainMatchResult:
    """Outcome of a frame-maximised match between two mode sets.

    Attributes
    ----------
    match : float
        Match in [0, 1] maximised over ``(alpha, beta, t_c)``, or NaN on failure.
    mismatch : float
        ``1 - match``.
    alpha : float
        Overall phase of the complex strain, in radians on (-pi, pi].  Applied
        *after* the mode sum, so it is a rotation of the polarisation basis by
        ``alpha / 2`` -- not an orbital phase.
    beta : float
        Rotation of the source frame about the orbital angular momentum axis, in
        radians on [0, 2pi).  Degenerate with the azimuthal viewing angle.
    phase_offsets : dict
        ``{m: alpha + m * beta}`` wrapped to (-pi, pi], the composite offset
        actually applied to each ``m``.  This is the quantity a single-mode
        match absorbs and therefore cannot report.
    time_shift : float
        Time offset applied to ``modes_a``, in seconds, relative to the
        (2,2)-peak alignment.
    match_at_zero_beta : float
        The same match with ``beta`` held at 0 (``alpha`` and ``t_c`` still
        maximised).  ``match - match_at_zero_beta`` is what the extra degree of
        freedom bought, and a large gap means the two mode sets are reported in
        different frames rather than disagreeing physically.
    inclination, azimuth : float
        Viewing angles the strain was evaluated at, in radians.
    f_lower_used, f_upper_used : float
        Band actually integrated over, in Hz.
    n_bins_in_band : int
        Positive-frequency bins between the two cutoffs.
    ms_used : tuple
        The ``m`` values that contributed, sorted.
    reason : str
        ``'ok'``, or a failure code: ``'no_common_modes'``, ``'zero_norm'``,
        ``'insufficient_bins'``.
    """

    match: float
    mismatch: float
    alpha: float
    beta: float
    phase_offsets: dict
    time_shift: float
    match_at_zero_beta: float
    inclination: float
    azimuth: float
    f_lower_used: float
    f_upper_used: float
    n_bins_in_band: int
    ms_used: tuple
    reason: str

    @property
    def is_usable(self) -> bool:
        return self.reason == "ok"


def sylm(ell: int, em: int, inclination: float, azimuth: float = 0.0) -> complex:
    """Spin-weight -2 spherical harmonic ``{}_{-2}Y_{lm}(iota, phi)``.

    Thin wrapper over LAL so that every caller in this package uses one
    convention.  ``spherical`` is also a dependency here but indexes its
    harmonics differently, and mixing the two is exactly the kind of silent
    convention error this function exists to prevent.
    """
    import lal

    return complex(
        lal.SpinWeightedSphericalHarmonic(
            float(inclination), float(azimuth), -2, int(ell), int(em)
        )
    )


def complete_negative_m(modes: dict) -> dict:
    """Add the ``m < 0`` modes implied by equatorial symmetry.

    For a non-precessing binary ``h_{l,-m} = (-1)^l conj(h_lm)``.  Verified
    against SXS data carrying both signs: the relation holds to 3e-6 (2,2) and
    4e-4 (3,3), i.e. to the numerical error of the simulation.

    **This is wrong for precessing systems**, which have no such symmetry, and
    is why it is a separate function rather than something applied silently
    inside the match.  Modes already present are never overwritten.
    """
    out = dict(modes)
    for (ell, em), h in modes.items():
        if em <= 0 or (ell, -em) in out:
            continue
        out[(ell, -em)] = ((-1) ** ell) * np.conj(np.asarray(h, dtype=complex))
    return out


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _partial_strains(modes, inclination, azimuth, taper_fraction):
    """Group modes by ``m`` into ``c_m(t) = sum_l h_lm  {}_{-2}Y_lm``.

    The frame offset acts on ``m`` alone, so summing over ``l`` first turns a
    scan over ``beta`` into a scan over at most nine coefficients instead of a
    re-evaluation of the whole mode sum.
    """
    by_m: dict = {}
    n = max(len(np.asarray(h)) for h in modes.values())
    for (ell, em), h in modes.items():
        arr = np.asarray(h, dtype=complex)
        if len(arr) < n:  # pragma: no cover - modes of one waveform share a grid
            arr = np.concatenate([arr, np.zeros(n - len(arr), dtype=complex)])
        term = arr * sylm(ell, em, inclination, azimuth)
        by_m[em] = by_m.get(em, 0.0) + term
    if taper_fraction:
        w = _start_window(n, taper_fraction)
        by_m = {em: c * w for em, c in by_m.items()}
    return by_m


def _peak_index(modes) -> int:
    """Index of the (2,2) amplitude peak, falling back to the loudest mode."""
    key = (
        (2, 2)
        if (2, 2) in modes
        else max(modes, key=lambda k: np.abs(np.asarray(modes[k])).max())
    )
    return int(np.argmax(np.abs(np.asarray(modes[key]))))


def compute_strain_match(
    modes_a,
    modes_b,
    delta_t: float,
    inclination: float,
    azimuth: float = 0.0,
    psd=None,
    psd_name: str = "aLIGOZeroDetHighPower",
    f_lower: float = 20.0,
    f_upper: float = None,
    n_beta: int = STRAIN_MATCH_N_BETA,
    taper_fraction: float = TAPER_FRACTION,
    symmetrize: bool = True,
) -> StrainMatchResult:
    r"""Match two mode sets as coherent strain, maximised over the frame offset.

    Builds the complex strain :math:`h = h_+ - i h_\times` from each mode set,

    .. math::

        h^A(t; \alpha, \beta) = e^{i\alpha}
            \sum_{\ell m} e^{i m \beta}\, h^A_{\ell m}(t)\,
            {}_{-2}Y_{\ell m}(\iota, \varphi)

    and returns the noise-weighted match maximised over :math:`\alpha`,
    :math:`\beta` and the time shift :math:`t_c`.

    Why these three and no others
    -----------------------------
    :math:`t_c` and :math:`\alpha` are the parameters a per-mode match already
    maximises over -- :math:`\alpha` multiplies :math:`h_+ - i h_\times` by a
    constant phase, which is a rotation of the polarisation basis by
    :math:`\alpha/2`.  :math:`\beta` is the one that per-mode matching cannot
    see: it rotates the source about the orbital angular momentum axis and is
    exactly degenerate with the azimuthal viewing angle, so it is unobservable
    and must be maximised over, but because it acts as :math:`e^{im\beta}` it is
    *not* absorbed by any single-mode phase maximisation.  Leaving it in place
    reports a mismatch dominated by a convention difference: measured at
    :math:`\iota = 60^\circ`, 5.7e-2 against 2.1e-4 for SXS:BBH:0201.

    Nothing else is maximised over.  Inclination is a physical parameter of the
    comparison, not a nuisance, so it is an argument rather than a search
    dimension; a mode set that only agrees at some fitted inclination is not the
    same waveform.

    Argument order and azimuth
    --------------------------
    ``beta`` is applied to ``modes_a``, which makes the result mildly asymmetric
    under swapping the arguments: measured on SXS:BBH:0201 at
    :math:`\iota = 60^\circ`, 1.87e-3 one way against 1.45e-3 the other.  That
    asymmetry is not an artefact -- an independent reference implementation
    reproduces both numbers -- and it is not really about argument order either.
    Rotating the source by ``beta`` is the same as viewing it from azimuth
    ``beta``, so putting the rotation on ``a`` rather than ``b`` amounts to
    evaluating the pair at a different absolute azimuth, and the mismatch
    genuinely depends on azimuth: the same simulation spans 1.35e-3 to 2.09e-3
    over a uniform azimuth grid, a range that contains both of the numbers
    above.  ``beta`` itself is unaffected, agreeing to 2e-4 rad under a swap.

    Use :func:`compute_strain_mismatch_averaged` when a single
    convention-independent number is wanted; use this function when the
    dependence on viewing geometry is the thing being studied.

    The inner product
    -----------------
    Two-sided, so that it is correct for the *complex* strain and agrees with
    the usual real-signal convention without a case split:

    .. math::

        \langle a | b \rangle = 2 \int_{-\infty}^{\infty}
            \frac{\tilde a(f)\, \tilde b^*(f)}{S_n(|f|)}\, df

    For real :math:`a, b` the negative-frequency half is the conjugate of the
    positive half and this collapses to :math:`4\,\mathrm{Re}\int_0^\infty`.
    Restricting to positive frequencies instead would silently discard the
    counter-rotating content, which is exactly the ``m < 0`` half of the mode
    sum.

    Parameters
    ----------
    modes_a, modes_b : dict
        ``{(ell, em): complex array}``, both sampled at ``delta_t``.  Need not
        be the same length or contain the same modes; only the common ``(l, m)``
        are used.
    delta_t : float
        Sample spacing in seconds.  Must be the same for both.
    inclination, azimuth : float
        Viewing angles in radians.
    psd : pycbc.types.FrequencySeries, optional
        Supply to override ``psd_name``.  Resampled onto the internal grid.
    f_lower, f_upper : float
        Band in Hz.  ``f_upper`` defaults to Nyquist.  Note ``f_lower`` here is
        the *strain* band edge and should be the (2,2) cutoff, not a value
        scaled by ``|m|/2``: the mode-wise scaling in :func:`mode_f_lower`
        exists because each mode is filtered separately, whereas the coherent
        strain carries every mode at once.
    n_beta : int
        Coarse ``beta`` grid size; refined locally afterwards.
    taper_fraction : float
        Start-only taper, as elsewhere in this module.  0 disables.
    symmetrize : bool
        Fill in absent ``m < 0`` modes via :func:`complete_negative_m`.  Leave
        on for non-precessing systems and **off** for precessing ones.

    Returns
    -------
    StrainMatchResult
        Carries ``alpha``, ``beta`` and ``phase_offsets = {m: alpha + m*beta}``
        alongside the match.
    """
    if symmetrize:
        modes_a = complete_negative_m(modes_a)
        modes_b = complete_negative_m(modes_b)

    common = sorted(set(modes_a) & set(modes_b))
    if not common:
        return _strain_fail("no_common_modes", inclination, azimuth, f_lower)
    modes_a = {k: modes_a[k] for k in common}
    modes_b = {k: modes_b[k] for k in common}

    ca = _partial_strains(modes_a, inclination, azimuth, taper_fraction)
    cb = _partial_strains(modes_b, inclination, azimuth, taper_fraction)
    ms = sorted(ca)

    # Align on the (2,2) peaks before padding, so t_c = 0 is the expected
    # optimum and the pruning below starts from a tight bound.
    ia, ib = _peak_index(modes_a), _peak_index(modes_b)
    na = max(len(c) for c in ca.values())
    nb = max(len(c) for c in cb.values())
    lead = max(ia, ib)
    trail = max(na - ia, nb - ib)
    n_fft = _next_pow2(lead + trail)

    def _place(c, i):
        out = np.zeros(n_fft, dtype=complex)
        out[lead - i : lead - i + len(c)] = c
        return out

    A = {em: np.fft.fft(_place(ca[em], ia)) for em in ms}
    B_total = np.fft.fft(sum(_place(cb[em], ib) for em in ms))

    freqs = np.fft.fftfreq(n_fft, d=delta_t)
    f_hi = float(f_upper) if f_upper else 0.5 / delta_t
    inv_psd = _inverse_psd(psd, psd_name, freqs, f_lower, f_hi, delta_t, n_fft)
    n_bins = int(np.count_nonzero(inv_psd[: n_fft // 2 + 1]))
    if n_bins < MIN_BINS_IN_BAND:
        return _strain_fail(
            "insufficient_bins", inclination, azimuth, f_lower, f_hi, n_bins
        )

    # Constant prefactor cancels in the normalised match; kept so that the
    # intermediate norms are ordinary SNR^2 values rather than arbitrary units.
    pref = 2.0 * delta_t / n_fft

    norm_b = np.sqrt(pref * np.sum(np.abs(B_total) ** 2 * inv_psd).real)
    gram = np.array(
        [[pref * np.sum(A[p] * np.conj(A[q]) * inv_psd) for q in ms] for p in ms]
    )
    if norm_b <= 0 or not np.isfinite(gram.trace().real) or gram.trace().real <= 0:
        return _strain_fail("zero_norm", inclination, azimuth, f_lower, f_hi, n_bins)

    # z_m(t_c) = <c_m shifted by t_c | h^B>, all t_c at once.
    z = np.stack([pref * np.fft.fft(A[em] * np.conj(B_total) * inv_psd) for em in ms])

    m_arr = np.asarray(ms, dtype=float)

    def _numerator(betas, cols):
        """|sum_m e^{i m beta} z_m(t)| on a (beta, t) grid."""
        return np.abs(np.exp(1j * np.outer(betas, m_arr)) @ z[:, cols])

    def _norm_a(betas):
        """||h^A(beta)||: the m-modes are not orthogonal under this product."""
        ph = np.exp(1j * np.outer(betas, m_arr))
        val = np.einsum("bp,pq,bq->b", ph, gram, np.conj(ph)).real
        return np.sqrt(np.maximum(val, 0.0))

    betas = np.linspace(0.0, 2.0 * np.pi, int(n_beta), endpoint=False)
    norms = _norm_a(betas)

    # Prune t_c exactly.  sum_m |z_m(t)| bounds the numerator |sum_m e^{i m
    # beta} z_m(t)| from above for every beta, so a column whose bound cannot
    # reach a value already achieved cannot host the maximum.  The bound has to
    # be divided by the *smallest* attainable ||h^A(beta)||, not by ||h^A(0)||:
    # the m-partial strains are not orthogonal under this inner product, so the
    # denominator moves with beta, and pruning on the numerator alone discards
    # columns that a smaller denominator would have promoted.  Measured cost of
    # getting this wrong: the reported match came out 1.6e-5 below the true
    # maximum on SXS:BBH:0304.
    n_min = float(norms.min())
    achieved = float((_numerator(betas[:1], slice(None)) / (norms[0] * norm_b)).max())
    bound = np.abs(z).sum(axis=0)
    cols = np.flatnonzero(bound >= achieved * n_min * norm_b)
    if cols.size > _MAX_TC_SAMPLES:
        cols = cols[np.argsort(bound[cols])[-_MAX_TC_SAMPLES:]]
    if cols.size == 0:  # pragma: no cover - the beta = 0 column always survives
        cols = np.array([int(np.argmax(bound))])

    ratio = _numerator(betas, cols) / (norms[:, None] * norm_b)
    bi = int(np.argmax(ratio) // ratio.shape[1])

    # Local refinement.  The fine grid is evaluated over *every* surviving
    # column, not just the one the coarse grid picked: refining beta at a frozen
    # t_c is only valid if the optimal t_c is independent of beta, and it is not.
    step = 2.0 * np.pi / n_beta
    fine = betas[bi] + np.linspace(-step, step, 65)
    fine_ratio = _numerator(fine, cols) / (_norm_a(fine)[:, None] * norm_b)
    fi, ti = np.unravel_index(int(np.argmax(fine_ratio)), fine_ratio.shape)
    col = cols[ti : ti + 1]
    beta_applied = float(fine[fi])
    match = float(min(fine_ratio[fi, ti], 1.0))

    # alpha is whatever makes the overlap real and positive at the optimum.
    overlap = np.exp(1j * beta_applied * m_arr) @ z[:, col[0]]
    alpha_applied = float(-np.angle(overlap))

    # Report the *offset*, not the correction that removes it, so that
    #     h_lm^A = exp[i(alpha + m beta)] h_lm^B
    # reads the same way here as in the fit it generalises.  The maximisation
    # above finds the inverse rotation, hence the sign flip.
    beta = float(np.mod(-beta_applied, 2.0 * np.pi))
    alpha = float(np.angle(np.exp(-1j * alpha_applied)))

    zero = _numerator(np.zeros(1), slice(None)) / (
        _norm_a(np.zeros(1))[:, None] * norm_b
    )
    n_shift = int(col[0]) - (n_fft if col[0] > n_fft // 2 else 0)

    return StrainMatchResult(
        match=match,
        mismatch=1.0 - match,
        alpha=alpha,
        beta=beta,
        phase_offsets={
            int(em): float(np.angle(np.exp(1j * (alpha + em * beta)))) for em in ms
        },
        time_shift=n_shift * delta_t,
        match_at_zero_beta=float(min(zero.max(), 1.0)),
        inclination=float(inclination),
        azimuth=float(azimuth),
        f_lower_used=float(f_lower),
        f_upper_used=f_hi,
        n_bins_in_band=n_bins,
        ms_used=tuple(int(m) for m in ms),
        reason="ok",
    )


@dataclass(frozen=True)
class PolarizationMatchResult:
    r"""Outcome of a fixed-orientation comparison of the two polarisations.

    Attributes
    ----------
    match_plus, match_cross : float
        Match in [0, 1] of :math:`h_+` and of :math:`h_\times`, maximised over
        the time shift **only**.  NaN on failure.
    match_plus_quadrature, match_cross_quadrature : float
        The same maximised over the time shift *and a quadrature phase of that
        one real series*, which is what ``pycbc.filter.match`` reports for a
        real template.  Always at least the value above.  This is **not** a
        marginalisation over the polarisation angle: rotating the basis mixes
        :math:`h_+` and :math:`h_\times` into each other, and no operation on
        :math:`h_+` alone can undo it.  For the polarisation-angle-maximised
        number use :func:`compute_strain_match`, whose ``alpha`` is exactly
        that rotation.
    inclination, coa_phase : float
        The orientation the comparison was made at, in radians.
    match_plus_best_psi, best_psi : float
        With ``n_psi``, the largest :math:`h_+` match over a rotation of the
        polarisation basis, and the rotation that achieves it in radians on
        [0, pi).  This is the diagnostic that says whether a fixed-basis
        disagreement is a real waveform difference or an unaccounted basis
        convention: if ``best_psi`` takes the same value across a population,
        it is a convention and can be removed; if it scatters, no absolute
        orientation is available and only the quadrature columns mean anything
        (catalog-comparison-paper findings 5aj).
    time_shift_plus, time_shift_cross : float
        Time offset applied, in seconds, relative to the (2,2)-peak alignment.
    n_bins : int
        Frequency bins inside the band.
    failure : str or None
        Reason, when the match could not be computed.
    """

    match_plus: float
    match_cross: float
    match_plus_quadrature: float
    match_cross_quadrature: float
    inclination: float
    coa_phase: float
    match_plus_best_psi: float = float("nan")
    best_psi: float = float("nan")
    time_shift_plus: float = float("nan")
    time_shift_cross: float = float("nan")
    n_bins: int = 0
    failure: str | None = None

    @property
    def mismatch_plus(self) -> float:
        return 1.0 - self.match_plus

    @property
    def mismatch_cross(self) -> float:
        return 1.0 - self.match_cross


def _polarization_fail(reason, inclination, coa_phase, n_bins=0):
    nan = float("nan")
    return PolarizationMatchResult(
        match_plus=nan,
        match_cross=nan,
        match_plus_quadrature=nan,
        match_cross_quadrature=nan,
        inclination=float(inclination),
        coa_phase=float(coa_phase),
        n_bins=int(n_bins),
        failure=reason,
    )


def compute_polarization_match(
    modes_a,
    modes_b,
    delta_t: float,
    inclination: float,
    coa_phase: float = 0.0,
    psd=None,
    psd_name: str = "aLIGOZeroDetHighPower",
    f_lower: float = 20.0,
    f_upper=None,
    taper_fraction: float = TAPER_FRACTION,
    n_psi: int = 0,
) -> PolarizationMatchResult:
    r"""Match :math:`h_+` and :math:`h_\times` separately at one orientation.

    Builds the complex strain from each mode set at the given inclination and
    coalescence phase,

    .. math::

        h = h_+ - i h_\times = \sum_{\ell m} h_{\ell m}\,
            {}_{-2}Y_{\ell m}(\iota, \varphi),

    and matches the two real polarisations one at a time, maximising over the
    time shift alone.

    Why the polarisations separately, and why nothing else is maximised over
    ---------------------------------------------------------------------------
    A rotation of the polarisation basis by :math:`\psi` sends
    :math:`h_+ + i h_\times \to e^{2i\psi}(h_+ + i h_\times)`, so it mixes
    the two polarisations into each other.  :func:`compute_strain_match`
    maximises over exactly that phase, which makes its result independent of
    the basis -- and therefore blind to the *relative* phase of the two
    polarisations.  That relative phase is physical: an observer with a known
    polarisation angle sees a definite mixture, and a parameter-estimation
    analysis that fixes :math:`\psi` inherits the disagreement this function
    reports.  Both mode sets are projected here in the same convention from the
    same source frame, so the basis is shared and the comparison is meaningful.
    ``match_*_quadrature`` is reported alongside as the value a search quoting a
    single real template would see; it is a quadrature phase of that one
    polarisation and not a rotation of the basis, so it does not recover a
    :math:`\psi` offset -- measured on a synthetic chirp rotated by
    :math:`\psi = \pi/4` at :math:`\iota = 60^\circ`, :math:`h_+` falls to
    0.613 and the quadrature value stays at 0.613, while
    :func:`compute_strain_match`, which maximises over the rotation itself,
    returns ~1.

    Inclination and coalescence phase are arguments, not search dimensions, for
    the reason :func:`compute_strain_match` gives about inclination: a mode set
    that agrees only at some fitted orientation is not the same waveform.  In
    particular the source-frame azimuth is **not** maximised over here, unlike
    in :func:`compute_strain_match`, because fixing the orientation is the whole
    point -- which means the caller is responsible for supplying two mode sets
    already in a common source frame.  Two sets that differ by a rigid rotation
    will report a large mismatch, correctly.

    Parameters
    ----------
    modes_a, modes_b : dict
        ``{(ell, m): series}``.  Only the modes present in both are used; no
        completeness requirement, since the projection is a plain sum with no
        mixing of ``m``.
    delta_t : float
        Sample spacing in seconds, shared by both mode sets.
    inclination, coa_phase : float
        Viewing angles in radians.
    n_psi : int, optional
        When positive, also scan this many polarisation-basis rotations over
        [0, pi) and report the best :math:`h_+` match and the angle achieving
        it.  A rotation by :math:`\psi` multiplies the complex strain by
        :math:`e^{2i\psi}`, so the scan costs two transforms per angle and
        nothing else; 24 is plenty to locate a broad maximum.

    Returns
    -------
    PolarizationMatchResult
    """
    from nrcats.waveform.modes_worker import _interpolated_peak

    common = set(modes_a) & set(modes_b)
    if not common:
        return _polarization_fail("no_common_modes", inclination, coa_phase)

    a_in = {k: modes_a[k] for k in common}
    b_in = {k: modes_b[k] for k in common}
    ha = sum(_partial_strains(a_in, inclination, coa_phase, taper_fraction).values())
    hb = sum(_partial_strains(b_in, inclination, coa_phase, taper_fraction).values())
    ha = np.asarray(ha, dtype=complex)
    hb = np.asarray(hb, dtype=complex)

    # Peak-align before padding, as compute_strain_match does, so that t_c = 0
    # is the expected optimum.
    ia, ib = _peak_index(a_in), _peak_index(b_in)
    lead = max(ia, ib)
    trail = max(len(ha) - ia, len(hb) - ib)
    n_fft = _next_pow2(lead + trail)

    def _place(c, i):
        out = np.zeros(n_fft, dtype=float)
        out[lead - i : lead - i + len(c)] = c
        return out

    freqs = np.fft.fftfreq(n_fft, d=delta_t)
    f_hi = float(f_upper) if f_upper else 0.5 / delta_t
    inv_psd = _inverse_psd(psd, psd_name, freqs, f_lower, f_hi, delta_t, n_fft)
    n_bins = int(np.count_nonzero(inv_psd[: n_fft // 2 + 1]))
    if n_bins < MIN_BINS_IN_BAND:
        return _polarization_fail("insufficient_bins", inclination, coa_phase, n_bins)

    # Same normalisation as compute_strain_match, so the numbers are comparable;
    # it cancels in the ratio either way.
    pref = 2.0 * delta_t / n_fft
    out = {}
    for name, ra, rb in (
        ("plus", ha.real, hb.real),
        ("cross", -ha.imag, -hb.imag),
    ):
        A = np.fft.fft(_place(ra, ia))
        B = np.fft.fft(_place(rb, ib))
        norm_a = np.sqrt(pref * float(np.sum(np.abs(A) ** 2 * inv_psd)))
        norm_b = np.sqrt(pref * float(np.sum(np.abs(B) ** 2 * inv_psd)))
        if not (norm_a > 0 and norm_b > 0):
            # A polarisation can be identically zero -- h_cross vanishes at
            # iota = 0 for an aligned-spin system -- and a normalised match on
            # it would be an arbitrary number rather than a failure.
            out[name] = (float("nan"), float("nan"), float("nan"))
            continue
        z = pref * np.fft.fft(A * np.conj(B) * inv_psd)
        denom = norm_a * norm_b
        m_fixed = _interpolated_peak(z.real) / denom
        m_quad = _interpolated_peak(np.abs(z)) / denom
        lag = int(np.argmax(z.real))
        shift = (lag if lag <= n_fft // 2 else lag - n_fft) * delta_t
        out[name] = (float(m_fixed), float(m_quad), float(shift))

    if all(not np.isfinite(v[0]) for v in out.values()):
        return _polarization_fail("zero_norm", inclination, coa_phase, n_bins)

    best_psi = best_psi_match = float("nan")
    if n_psi and int(n_psi) > 0:
        # h_+(psi) = Re[e^{2 i psi} h], applied to ``modes_b`` only: the scan
        # asks what basis rotation of the *model* reproduces the NR h_+, so it
        # must undo a relative offset.  Rotating both sides by the same psi
        # changes what h_+ means for each while preserving their relative
        # offset, which is a common rotation and recovers nothing -- measured,
        # it left a known 90 degree offset at a match of 0.66.
        A = np.fft.fft(_place(ha.real, ia))
        na = np.sqrt(pref * float(np.sum(np.abs(A) ** 2 * inv_psd)))
        for psi in np.linspace(0.0, np.pi, int(n_psi), endpoint=False):
            B = np.fft.fft(_place((hb * np.exp(2j * psi)).real, ib))
            nb = np.sqrt(pref * float(np.sum(np.abs(B) ** 2 * inv_psd)))
            if not (na > 0 and nb > 0):
                continue
            z = pref * np.fft.fft(A * np.conj(B) * inv_psd)
            m = _interpolated_peak(z.real) / (na * nb)
            if not np.isfinite(best_psi_match) or m > best_psi_match:
                best_psi_match, best_psi = float(m), float(psi)

    return PolarizationMatchResult(
        match_plus=out["plus"][0],
        match_cross=out["cross"][0],
        match_plus_quadrature=out["plus"][1],
        match_cross_quadrature=out["cross"][1],
        inclination=float(inclination),
        coa_phase=float(coa_phase),
        time_shift_plus=out["plus"][2],
        time_shift_cross=out["cross"][2],
        match_plus_best_psi=best_psi_match,
        best_psi=best_psi,
        n_bins=n_bins,
    )


def compute_strain_mismatch_averaged(
    modes_a, modes_b, delta_t: float, inclination: float, n_azimuth: int = 12, **kwargs
) -> dict:
    """Frame-maximised strain mismatch averaged over azimuth.

    :func:`compute_strain_match` is evaluated at a single azimuth, and its value
    depends on that choice -- 1.55x between the best and worst azimuth for
    SXS:BBH:0201.  Averaging removes both that dependence and the residual
    asymmetry under swapping ``modes_a`` and ``modes_b``, since the two argument
    orders differ only by which absolute azimuth the pair is evaluated at.  This
    is the number to quote when the viewing geometry is a nuisance rather than
    the subject.

    Returns
    -------
    dict
        ``mean``, ``median``, ``min``, ``max`` of the mismatch over the grid,
        plus ``per_azimuth`` (the individual :class:`StrainMatchResult` objects)
        so the spread can be inspected rather than assumed small.
    """
    results = [
        compute_strain_match(
            modes_a, modes_b, delta_t, inclination, azimuth=float(az), **kwargs
        )
        for az in np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    ]
    vals = np.array([r.mismatch for r in results if r.is_usable])
    if vals.size == 0:
        nan = float("nan")
        return {
            "mean": nan,
            "median": nan,
            "min": nan,
            "max": nan,
            "per_azimuth": results,
        }
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "per_azimuth": results,
    }


def _inverse_psd(psd, psd_name, freqs, f_lower, f_upper, delta_t, n_fft):
    """``1 / S_n(|f|)`` on the two-sided FFT grid, zeroed outside the band."""
    from .detectors import psd_from_name

    delta_f = 1.0 / (n_fft * delta_t)
    if psd is None:
        s = np.asarray(psd_from_name(psd_name, n_fft // 2 + 1, delta_f, f_lower))
    else:
        # Resample a supplied PSD onto this grid rather than assuming it matches.
        src_f = np.arange(len(psd)) * float(psd.delta_f)
        s = np.interp(
            np.arange(n_fft // 2 + 1) * delta_f,
            src_f,
            np.asarray(psd),
            left=np.inf,
            right=np.inf,
        )
    inv = np.zeros(n_fft // 2 + 1)
    band = (np.arange(n_fft // 2 + 1) * delta_f >= f_lower) & (
        np.arange(n_fft // 2 + 1) * delta_f <= f_upper
    )
    good = band & np.isfinite(s) & (s > 0)
    inv[good] = 1.0 / s[good]
    two_sided = np.zeros(n_fft)
    two_sided[: n_fft // 2 + 1] = inv
    two_sided[n_fft // 2 + 1 :] = inv[1 : (n_fft + 1) // 2][::-1]
    return two_sided


def _strain_fail(reason, inclination, azimuth, f_lower, f_upper=float("nan"), n_bins=0):
    nan = float("nan")
    return StrainMatchResult(
        match=nan,
        mismatch=nan,
        alpha=nan,
        beta=nan,
        phase_offsets={},
        time_shift=nan,
        match_at_zero_beta=nan,
        inclination=float(inclination),
        azimuth=float(azimuth),
        f_lower_used=float(f_lower),
        f_upper_used=float(f_upper),
        n_bins_in_band=int(n_bins),
        ms_used=(),
        reason=reason,
    )
