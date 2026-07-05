"""Standalone loader functions for WaveformModes.

Each function accepts ``cls`` as its first argument (classmethod pattern)
so the caller in ``modes.py`` can do::

    @classmethod
    def load_from_h5(cls, ...):
        from nrcats.waveform.loaders import load_from_h5 as _impl
        return _impl(cls, ...)

No import of ``WaveformModes`` is needed here, avoiding circular imports.
"""

import logging
logger = logging.getLogger(__name__)

import os
import warnings

import h5py
import numpy as np
import quaternionic
from scipy.interpolate import InterpolatedUnivariateSpline
from sxs.waveforms.format_handlers.nrar import (
    h,
    translate_data_type_to_spin_weight,
    translate_data_type_to_sxs_string,
)

from nrcats.waveform.units import ELL_MIN, ELL_MAX, _modal_dt


def load_from_h5(cls, file_path_or_open_file, metadata={}, verbosity=0):
    """Load SWSH waveform modes from an HDF5 file (RIT/MAYA catalog format).

    Parameters
    ----------
    cls : type
        The ``WaveformModes`` class (or subclass) to instantiate.
    file_path_or_open_file : str or h5py.File
        Path to the HDF5 file, or an already-open file object.
    metadata : dict, optional
        Simulation metadata dict.
    verbosity : int, optional
        Verbosity level (0 = quiet).

    Returns
    -------
    WaveformModes
    """
    if type(file_path_or_open_file) is h5py._hl.files.File:
        h5_file = file_path_or_open_file
        close_input_file = False
    elif os.path.exists(file_path_or_open_file):
        h5_file = h5py.File(file_path_or_open_file, "r")
        close_input_file = True
    else:
        raise RuntimeError(f"Could not use or open {file_path_or_open_file}")

    h5_filepath = h5_file.filename

    ell_min, ell_max = 99, -1
    LM = []
    t_min, t_max, dt = -1e99, 1e99, 1
    mode_data = {}
    for ell in range(ELL_MIN, ELL_MAX + 1):
        for em in range(-ell, ell + 1):
            afmt = f"amp_l{ell}_m{em}"
            pfmt = f"phase_l{ell}_m{em}"
            if afmt not in h5_file or pfmt not in h5_file:
                continue

            try:
                amp_time = h5_file[afmt]["X"][:]
                amp = h5_file[afmt]["Y"][:]
                phase_time = h5_file[pfmt]["X"][:]
                phase = h5_file[pfmt]["Y"][:]
            except KeyError:
                if verbosity > 0:
                    logger.info(
                        f"Skipping mode l={ell}, m={em} for {file_path_or_open_file} "
                        "since columns 'X' and 'Y' not found"
                    )
                continue
            mode_data[(ell, em)] = [amp_time, amp, phase_time, phase]
            t_min = max(t_min, amp_time[0], phase_time[0])
            t_max = min(t_max, amp_time[-1], phase_time[-1])
            dt = min(dt, _modal_dt(amp_time), _modal_dt(phase_time))
            ell_min = min(ell_min, ell)
            ell_max = max(ell_max, ell)
            LM.append([ell, em])

    if close_input_file:
        h5_file.close()

    if len(LM) == 0:
        raise RuntimeError(
            "We did not find even one mode in the file. Perhaps the "
            "format `amp_l?_m?` and `phase_l?_m?` is not the "
            "nomenclature of datagroups in the input file?"
        )

    times = np.arange(t_min, t_max + 0.5 * dt, dt)
    data = np.empty((len(times), len(LM)), dtype=complex)
    for idx, (ell, em) in enumerate(LM):
        amp_time, amp, phase_time, phase = mode_data[(ell, em)]
        amp_interp = InterpolatedUnivariateSpline(amp_time, amp)
        phase_interp = InterpolatedUnivariateSpline(phase_time, phase)
        data[:, idx] = amp_interp(times) * np.exp(1j * phase_interp(times))

    w_attributes = {}
    w_attributes["_filepath"] = h5_filepath
    w_attributes["metadata"] = metadata
    w_attributes["history"] = ""
    w_attributes["frame"] = quaternionic.array([[1.0, 0.0, 0.0, 0.0]])
    w_attributes["frame_type"] = "inertial"
    w_attributes["data_type"] = h
    w_attributes["spin_weight"] = translate_data_type_to_spin_weight(
        w_attributes["data_type"]
    )
    w_attributes["data_type"] = translate_data_type_to_sxs_string(
        w_attributes["data_type"]
    )
    w_attributes["r_is_scaled_out"] = True
    w_attributes["m_is_scaled_out"] = True

    return cls(
        data,
        time=times,
        time_axis=0,
        modes_axis=1,
        ell_min=ell_min,
        ell_max=ell_max,
        verbosity=verbosity,
        **w_attributes,
    )


def load_from_targz(cls, file_path, metadata={}, verbosity=0):
    """Load SWSH waveform modes from a ``.tar.gz`` archive (RIT psi4 format).

    Parameters
    ----------
    cls : type
        The ``WaveformModes`` class (or subclass) to instantiate.
    file_path : str
        Path to the ``.tar.gz`` archive.
    metadata : dict, optional
        Simulation metadata dict.
    verbosity : int, optional
        Verbosity level (0 = quiet).

    Returns
    -------
    WaveformModes
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise RuntimeError(f"Could not use or open {file_path}")

    import re
    import tarfile

    def get_tag(name):
        return os.path.splitext(os.path.splitext(os.path.basename(name))[0])[0]

    def get_el_em_from_filename(filename: str):
        substr = re.search(pattern=r"l\d+_m\d+", string=filename)
        if substr is None:
            substr = re.search(pattern=r"l\d+_m-\d+", string=filename)
        elem = substr[0].split("_")
        return (int(elem[0].strip("l")), int(elem[1].strip("m")))

    ell_min, ell_max = 99, -1
    t_min, t_max, dt = -1e99, 1e99, 1

    file_tag = get_tag(file_path)
    mode_data = {}
    reference_mode_num_for_length = ()
    possible_ascii_extensions = ["asc", "dat", "txt"]

    with tarfile.open(file_path, "r:gz") as tar:
        if verbosity > 4:
            logger.info(f"Opening tarfile: {file_path}")
        for dat_file in tar.getmembers():
            dat_file_name = dat_file.name
            if verbosity > 4:
                logger.info(f"dat_file_name is: {dat_file_name}")
            if file_tag not in dat_file_name or np.all(
                [f".{ext}" not in dat_file_name for ext in possible_ascii_extensions]
            ):
                if verbosity > 5:
                    logger.info(
                        f"{file_tag} not in {dat_file_name} is"
                        f" {file_tag not in dat_file_name}"
                    )
                    logger.info(
                        "the other flag is: ",
                        np.all(
                            [
                                f".{ext}" not in dat_file_name
                                for ext in possible_ascii_extensions
                            ]
                        ),
                    )
                continue
            ell, em = get_el_em_from_filename(dat_file_name)
            with tar.extractfile(dat_file_name) as f:
                reference_mode_num_for_length = (ell, em)
                mode_data[(ell, em)] = np.loadtxt(f)
                nrows, ncols = np.shape(mode_data[(ell, em)])
                if nrows < ncols:
                    mode_data[(ell, em)] = mode_data[(ell, em)].T
            t_min = max(t_min, mode_data[(ell, em)][0, 0])
            t_max = min(t_max, mode_data[(ell, em)][-1, 0])
            dt = min(dt, _modal_dt(mode_data[(ell, em)][:, 0]))
            ell_min = min(ell_min, ell)
            ell_max = max(ell_max, ell)

    # Capture the modes actually present in the file before zero-padding.
    present_modes = set(mode_data.keys())

    # We populate LM here because it has to be ordered, as the WaveformModes
    # class expects an ordered data set.
    LM = []
    zero_padded_modes = []
    for ell in range(ELL_MIN, ELL_MAX + 1):
        for em in range(-ell, ell + 1):
            if (ell, em) in mode_data:
                LM.append([ell, em])
            else:
                reference_mode = mode_data[reference_mode_num_for_length]
                mode_data[(ell, em)] = np.zeros(np.shape(reference_mode))
                mode_data[(ell, em)][:, 0] = reference_mode[:, 0]
                LM.append([ell, em])
                zero_padded_modes.append((ell, em))

    if zero_padded_modes:
        warnings.warn(
            f"{file_path}: {len(zero_padded_modes)} mode(s) not present in file"
            f" and were zero-padded: {zero_padded_modes}",
            UserWarning,
            stacklevel=3,  # load_from_targz → _impl → caller
        )

    if len(LM) == 0:
        raise RuntimeError(
            "We did not find even one mode in the file. Perhaps the "
            "format `amp_l?_m?` and `phase_l?_m?` is not the "
            "nomenclature of datagroups in the input file?"
        )

    times = np.arange(t_min, t_max + 0.5 * dt, dt)
    data = np.empty((len(times), len(LM)), dtype=complex)
    for idx, (ell, em) in enumerate(LM):
        mode_time = mode_data[(ell, em)][:, 0]
        sort_idx = np.argsort(mode_time)
        mode_time = mode_time[sort_idx]
        
        mode_real = mode_data[(ell, em)][sort_idx, 1]
        mode_imag = mode_data[(ell, em)][sort_idx, 2]
        if verbosity > 5:
            logger.info(f"Interpolating mode {ell}, {em}. Data length: {len(mode_time)}")
        mode_real_interp = InterpolatedUnivariateSpline(mode_time, mode_real)
        mode_imag_interp = InterpolatedUnivariateSpline(mode_time, mode_imag)
        data[:, idx] = mode_real_interp(times) + 1j * mode_imag_interp(times)

    w_attributes = {}
    w_attributes["_filepath"] = file_path
    w_attributes["metadata"] = metadata
    w_attributes["history"] = ""
    w_attributes["frame"] = quaternionic.array([[1.0, 0.0, 0.0, 0.0]])
    w_attributes["frame_type"] = "inertial"
    w_attributes["data_type"] = h
    w_attributes["spin_weight"] = translate_data_type_to_spin_weight(
        w_attributes["data_type"]
    )
    w_attributes["data_type"] = translate_data_type_to_sxs_string(
        w_attributes["data_type"]
    )
    w_attributes["r_is_scaled_out"] = True
    w_attributes["m_is_scaled_out"] = True

    wfm = cls(
        data,
        time=times,
        time_axis=0,
        modes_axis=1,
        ell_min=ell_min,
        ell_max=ell_max,
        verbosity=verbosity,
        **w_attributes,
    )
    wfm._present_modes = present_modes
    return wfm
