import numpy as np
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform_modes
from .waveform.modes import WaveformModes


def generate_lalsim_modes(
    params: dict,
    total_mass: float,
    distance: float = 1.0,
    delta_t_seconds: float = 1.0 / 4096,
    approximant: str = "SEOBNRv4PHM",
    sim_name: str | None = None,
    catalog=None,
    nr_wfm=None,
    f_lower_override: float | None = None,
    time_bounds: tuple[float, float] | None = None,
) -> tuple[dict, float]:
    """Generate physical-unit modes using LALSimulation via PyCBC.

    Parameters
    ----------
    params : dict
        PyCBC-compatible parameter dict (e.g., from CatalogBase.get_parameters()).
    total_mass : float
        Total binary mass in solar masses.
    distance : float, optional
        Luminosity distance in Mpc.
    delta_t_seconds : float, optional
        Sample spacing in seconds.
    approximant : str, optional
        LAL approximant string (e.g., 'SEOBNRv4PHM').
    f_lower_override : float, optional
        Override the starting GW frequency in Hz.
    time_bounds : tuple[float, float], optional
        (t_start, t_end) in seconds relative to the peak to precisely truncate/pad the LAL waveform
        to match another waveform's length.

    Returns
    -------
    tuple[dict, float]
        ( {(ell, em): pycbc.types.TimeSeries}, f_lower_effective )
    """
    f_lower_hz = f_lower_override if f_lower_override is not None else params["f_lower"]

    if f_lower_hz <= 0:
        raise ValueError("f_lower must be positive for LALSimulation models.")

    # Call pycbc's mode generator (returns dict of TimeSeries in dimensionless units if not specified,
    # but PyCBC usually returns strain in physical units. However, get_td_waveform_modes has specific
    # scaling. Let's check: PyCBC's get_td_waveform_modes takes physical masses and distances).
    # Wait, PyCBC's get_td_waveform_modes takes mass1, mass2, etc. and returns physical strain.

    # We construct the arguments
    kwargs = {
        "approximant": approximant,
        "mass1": params["mass1"],
        "mass2": params["mass2"],
        "spin1x": params["spin1x"],
        "spin1y": params["spin1y"],
        "spin1z": params["spin1z"],
        "spin2x": params["spin2x"],
        "spin2y": params["spin2y"],
        "spin2z": params["spin2z"],
        "f_lower": f_lower_hz,
        "delta_t": delta_t_seconds,
        "distance": distance,
    }

    try:
        modes = get_td_waveform_modes(**kwargs)
    except ValueError as e:
        if "I don't support approximant" in str(e):
            raise ValueError(
                f"Approximant {approximant} is not supported for TD mode generation in PyCBC. "
                f"Consider using a time-domain model like SEOBNRv4PHM."
            )
        raise

    # PyCBC get_td_waveform_modes returns modes as pycbc TimeSeries objects.
    # The epoch of these modes is usually set such that t=0 is near merger.

    # Let's align the peak of the (2,2) mode to t=0 exactly.
    if (2, 2) not in modes:
        raise ValueError(
            f"LAL approximant {approximant} did not return the dominant (2,2) mode."
        )

    hp22, hc22 = modes[(2, 2)]
    amp22 = np.sqrt(hp22.data**2 + hc22.data**2)
    peak_idx = int(np.argmax(amp22))
    peak_time_phys = float(hp22.sample_times[peak_idx])

    # Shift all epochs so that the peak is at exactly 0.0 and combine into complex TimeSeries
    complex_modes = {}
    for key, (hp, hc) in modes.items():
        t_shifted = hp.start_time - peak_time_phys
        h_complex = hp.data - 1j * hc.data
        complex_modes[key] = TimeSeries(
            h_complex, delta_t=delta_t_seconds, epoch=t_shifted
        )

    modes = complex_modes

    # If time_bounds are provided, we must truncate and zero-pad the waveform to match exactly
    if time_bounds is not None:
        t_start, t_end = time_bounds
        # We need to resample/truncate the PyCBC TimeSeries to precisely match the time array
        # spanning [t_start, t_end] with spacing delta_t_seconds.
        num_samples = int(np.round((t_end - t_start) / delta_t_seconds)) + 1
        target_times = t_start + np.arange(num_samples) * delta_t_seconds

        aligned_modes = {}
        for key, ts in modes.items():
            # Linear interpolation onto the target time grid
            # For regions outside the LAL simulation bounds, fill with 0
            interp_real = np.interp(
                target_times,
                np.array(ts.sample_times),
                ts.data.real,
                left=0.0,
                right=0.0,
            )
            interp_imag = np.interp(
                target_times,
                np.array(ts.sample_times),
                ts.data.imag,
                left=0.0,
                right=0.0,
            )

            new_ts = TimeSeries(
                interp_real + 1j * interp_imag, delta_t=delta_t_seconds, epoch=t_start
            )
            aligned_modes[key] = new_ts
        modes = aligned_modes

    return modes, f_lower_hz


def lalsim_dict_to_waveform_modes(h_dict: dict, ell_max: int = 4):
    """Wrap a dictionary of pycbc mode TimeSeries into a WaveformModes object."""
    if not h_dict:
        raise ValueError("h_dict is empty")

    class LALSimWaveformModes(WaveformModes):
        def __init__(self, h_dict_in, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._h_dict = h_dict_in

        def get_mode(self, ell, em, **kwargs):
            return self._h_dict[(ell, em)]

    # Extract time array from any mode
    any_mode = next(iter(h_dict.values()))
    time_array = np.array(any_mode.sample_times)

    num_modes = (ell_max + 1) ** 2 - 4
    data = np.zeros((len(time_array), num_modes), dtype=complex)

    wfm = LALSimWaveformModes(h_dict, data, time=time_array, ell_min=2, ell_max=ell_max)

    for (ell, m), ts in h_dict.items():
        if ell > ell_max or ell < 2:
            continue
        try:
            idx = wfm.index(ell, m)
            wfm.data[:, idx] = ts.data
        except ValueError:
            pass

    return wfm
