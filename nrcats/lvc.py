"""Frame-rotation helpers and LVCNR format utilities.

This module provides the low-level functions needed to:

1. **Read LVCNR HDF5 files** – parse mode data and strain from the LVK
   NR format (``get_modes_from_lvcnr_file``, ``get_strain_from_lvcnr_file``).

2. **Compute the LAL source frame** – derive the NR orbital normal
   (``LNhat``) and binary-separation unit vector (``nhat``) from HDF5
   file attributes, time-series data, or SXS-style simulation metadata
   (``compute_lal_source_frame_from_sxs_metadata``,
   ``compute_lal_source_frame_by_interp``).

3. **Rotate NR waveforms into the LAL frame** – compute the SO(3) rotation
   angles (``theta``, ``psi``, ``alpha``) needed to project the NR
   waveform onto the detector line of sight at a given inclination and
   reference phase (``get_nr_to_lal_rotation_angles``).

4. **Build LAL mode dictionaries** – create LAL parameter dictionaries
   with a specified set of (ell, m) modes for use with LALSimulation
   waveform routines (``get_lal_mode_dictionary``,
   ``get_lal_mode_dictionary_from_lmax``).

Public functions
----------------
get_lal_mode_dictionary(mode_array)
get_lal_mode_dictionary_from_lmax(lmax)
get_modes_from_lvcnr_file(path_to_file, Mtot, distance, srate, lmax, f_low)
get_strain_from_lvcnr_file(path_to_file, Mtot, distance, inclination, phi_ref, srate, mode_array)
check_interp_req(h5_file, metadata, ref_time, avail_ref_time)
get_ref_freq_from_ref_time(h5_file, ref_time)
get_ref_time_from_ref_freq(h5_file, ref_freq)
check_nr_attrs(sim_metadata_object, req_attrs)
get_interp_ref_values_from_h5_file(h5_file, req_ts_attrs, ref_time)
get_ref_vals(sim_metadata_object, req_attrs)
compute_lal_source_frame_from_sxs_metadata(sim_metadata)
compute_lal_source_frame_by_interp(h5_file, req_ts_attrs, t_ref)
normalize_metadata(sim_metadata)
get_ref_time_from_metadata(sim_metadata)
transform_spins_nr_to_lal(nrSpin1, nrSpin2, n_hat, ln_hat)
get_nr_to_lal_rotation_angles(h5_file, sim_metadata, inclination, phi_ref, f_ref, t_ref, tol)
"""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import h5py
import lal
import lalsimulation as lalsim
import numpy as np
from pycbc.pnutils import mtotal_eta_to_mass1_mass2
from pycbc.types import TimeSeries
from scipy.interpolate import interp1d


def get_lal_mode_dictionary(mode_array: list) -> object:
    """
    Get LALDict with all specified modes.


    Parameters
    ----------
    mode_array: list of modes, eg [[2,2], [3,3]]

    Returns
    -------
    waveform_dictionary: LALDict with all modes included

    """
    waveform_dictionary = lal.CreateDict()
    mode_array_lal = lalsim.SimInspiralCreateModeArray()
    for mode in mode_array:
        lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
    lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)

    return waveform_dictionary


def get_lal_mode_dictionary_from_lmax(lmax: int) -> object:
    r"""
    Get LALDict with modes derived from `lmax`.


    Parameters
    ----------
    lmax: max value of :math:`\ell` to use

    Returns
    -------
    waveform_dictionary: LALDict with all modes upto `lmax` included

    """

    mode_array = [
        [ell, emm] for ell in range(2, lmax + 1) for emm in range(-ell, ell + 1)
    ]
    return get_lal_mode_dictionary(mode_array)


def get_modes_from_lvcnr_file(
    path_to_file: str,
    Mtot: float,
    distance: float,
    srate: int,
    lmax: int = 4,
    f_low: float | None = None,
) -> tuple:
    r"""
    Get individual modes from LVCNR format file.


    Parameters
    ==========
    path_to_file: string
        Path to LVCNR file
    Mtot: float
        Total mass (in units of MSUN) to scale the waveform to
    distance: float
        Luminosity Distance (in units of MPc) to scale the waveform to
    srate: int
        Sampling rate for the waveform
    lmax: int
        Max value of :math:`\ell` to use
        (Default: 4)
    f_low: float
        Value of the low frequency to start waveform generation
        Uses value given from the LVCNR file if `None` is provided
        (Default: None)

    Returns
    =======
    mass_ratio: float
        Mass ratio derived from the LVCNR file
    spins_args: list
        List of spins derived from the LVCNR file
    eccentricity: float
        Eccentricty derived from the LVCNR file.
        Returns `None` is eccentricity is not a number.
    f_low: float
        Low Frequency derived either from the file, or provided
        in the call
    f_ref: float
        Reference Frequency derived from the file
    modes: dict of pycbc TimeSeries objects
        dict containing all the read in modes
    """

    with h5py.File(path_to_file) as h5file:
        waveform_dict = get_lal_mode_dictionary_from_lmax(lmax)

        f_low_in_file = h5file.attrs["f_lower_at_1MSUN"] / Mtot
        f_ref = f_low_in_file

        if f_low is None:
            f_low = f_low_in_file

        if h5file.attrs["Format"] < 3:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                -1, Mtot, path_to_file
            )
        else:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                f_low, Mtot, path_to_file
            )

        mass_args = list(
            mtotal_eta_to_mass1_mass2(Mtot * lal.MSUN_SI, h5file.attrs["eta"])
        )

        try:
            eccentricity = float(h5file.attrs["eccentricity"])
        except ValueError:
            eccentricity = None

    values_mode_array = lalsim.SimInspiralWaveformParamsLookupModeArray(waveform_dict)
    _, modes = lalsim.SimInspiralNRWaveformGetHlms(
        1 / srate,
        *mass_args,
        distance * 1e6 * lal.PC_SI,
        f_low,
        f_ref,
        *spin_args,
        path_to_file,
        values_mode_array,
    )
    mode = modes
    return_dict = dict()
    while 1 > 0:
        try:
            l, m = mode.l, mode.m
            read_mode = mode.mode.data.data
            return_dict[(l, m)] = TimeSeries(read_mode, 1 / srate)
            mode = mode.next
        except AttributeError:
            break

    return (
        mass_args[1] / mass_args[0],
        spin_args,
        eccentricity,
        f_low,
        f_ref,
        return_dict,
    )


def get_strain_from_lvcnr_file(
    path_to_file: str,
    Mtot: float,
    distance: float,
    inclination: float,
    phi_ref: float,
    srate: int,
    mode_array: list | None = None,
) -> tuple:
    """
    Get full strain from LVCNR format file.


    Parameters
    ==========
    path_to_file: string
        Path to LVCNR file
    Mtot: float
        Total mass (in units of MSUN) to scale the waveform to
    distance: float
        Luminosity Distance (in units of MPc) to scale the waveform to
    srate: int
        Sampling rate for the waveform
    mode_array: list
        list of modes to be included. `None` means all modes are included.
        (Default:None)

    Returns
    =======
    UNDER CONSTRUCTION
    """

    longAscNodes = 0
    eccentricity = 0
    meanPerAno = 0

    fixed_args = [
        distance * lal.PC_SI * 1e6,
        inclination,
        phi_ref,
        longAscNodes,
        eccentricity,
        meanPerAno,
        1 / srate,
    ]

    with h5py.File(path_to_file) as h5file:
        if mode_array is not None:
            waveform_dict = get_lal_mode_dictionary(mode_array)
        else:
            waveform_dict = lal.CreateDict()

        lalsim.SimInspiralWaveformParamsInsertNumRelData(waveform_dict, path_to_file)
        f_low = h5file.attrs["f_lower_at_1MSUN"] / Mtot

        if h5file.attrs["Format"] < 3:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                -1, Mtot, path_to_file
            )
        else:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                f_low, Mtot, path_to_file
            )

        mass_args = list(
            mtotal_eta_to_mass1_mass2(Mtot * lal.MSUN_SI, h5file.attrs["eta"])
        )

    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        *mass_args,
        *spin_args,
        *fixed_args,
        f_low,
        f_low,
        waveform_dict,
        lalsim.GetApproximantFromString("NR_hdf5"),
    )

    hp_timeseries = TimeSeries(hp.data.data, fixed_args[-1])
    hc_timeseries = TimeSeries(hc.data.data, fixed_args[-1])

    return dict(
        hp=hp_timeseries,
        hc=hc_timeseries,
        f_low=f_low,
        mass_args=mass_args,
        spin_args=spin_args,
        fixed_args=fixed_args,
    )


def check_interp_req(
    h5_file: object = None,
    metadata: dict | None = None,
    ref_time: float | None = None,
    avail_ref_time: float | None = None,
) -> tuple:
    """Check if the required reference time is different from
    the available reference time in the NR HDF5 file or the
    simulation metadata.

    Parameters
    ----------
    h5_file : file object
                The waveform h5 file handle.
    metadata :
    ref_time, avail_ref_time : float
                                The use and  available nr reference time.

    Returns
    -------
    interp : bool
             Whether interpolation across time is required.
    avail_ref_time: float
                    The ref_time available in the NR HDF5 file.
    """

    if avail_ref_time is None:
        # CCheck for ref time in h5 file
        if h5_file is not None:
            keys = list(h5_file.attrs.keys())

            if "reference_time" in keys:
                avail_ref_time = h5_file.attrs["reference_time"]
            elif "ref_time" in keys:
                avail_ref_time = h5_file.attrs["ref_time"]
            elif "relaxed_time" in keys:
                avail_ref_time = h5_file.attrs["relaxed_time"]
            else:
                logger.info("Reference time not found in waveform h5 file.")

        if not avail_ref_time:
            # If not found, continue search in metadata.
            if metadata is not None:
                keys = list(metadata.keys())

                if "reference_time" in keys:
                    avail_ref_time = metadata["reference_time"]
                elif "relaxed_time" in keys:
                    avail_ref_time = metadata["relaxed_time"]
                else:
                    logger.info("Reference time not found in simulation metadata file.")

        if not avail_ref_time:
            # Then this is GT simulation!
            logger.info(
                "Reference time should be computed from"
                "the reference orbital frequency!"
            )

    interp = True
    if isinstance(ref_time, float):
        if abs(avail_ref_time - ref_time) < 1e-5:
            interp = False
    else:
        # ref_time was not supplied (None or non-float); no interpolation needed.
        interp = False

    return interp, avail_ref_time


def get_ref_freq_from_ref_time(h5_file: object, ref_time: float) -> float:
    """Get the reference frequency from reference time

    Parameters
    ----------
    h5_file : file object
             The waveform h5 file handle.
    ref_time : float
               Reference time.
    Returns
    -------
    f_ref : float
           Reference frequency.
    """

    time, freq = h5_file.attrs["Omega-vs-time"]

    ref_freq = interp1d(time, freq, kind="cubic")[ref_time]

    return ref_freq


def get_ref_time_from_ref_freq(h5_file: object, ref_freq: float) -> float:
    """Get the reference time from reference frequency

    Parameters
    ----------
    h5_file : file object
             The waveform h5 file handle.
    ref_freq : float
               The reference frequency.
    Returns
    -------
    fTime : float
           Reference time.
    """

    time, freq = h5_file.attrs["Omega-vs-time"]

    ref_time = interp1d(freq, time, kind="cubic")[ref_freq]

    return ref_time


def check_nr_attrs(
    sim_metadata_object: object,
    req_attrs: list = ["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"],
) -> bool:
    """Check if the NR h5 file or a simulation metadata dictionary
        contains all the attributes required.

    Parameters
    ----------
    sim_metadata_object : h5 file object, dict
                     The NR h5py file handle or simulation metadata.
    req_attrs : list
               A list of attribute keys.
    Returns
    -------
    present : bool
              Whether or not all specified attributes are present.
    absent_attrs : list
                 The attributes that are absent.
    """
    if isinstance(sim_metadata_object, h5py.File):
        all_attrs = list(sim_metadata_object.attrs.keys())

    elif isinstance(sim_metadata_object, dict):
        all_attrs = list(sim_metadata_object.keys())
    else:
        raise TypeError("Please supply an open h5py file handle or a dictionary")

    absent_attrs = []
    present = True

    for item in req_attrs:
        if item not in all_attrs:
            present = False
            absent_attrs.append(item)

    return present, absent_attrs


def get_interp_ref_values_from_h5_file(
    h5_file: object, req_ts_attrs: list, ref_time: float
) -> dict:
    """Get the interpolated reference values at a given reference time
    from the NR HDF5 File

    Parameters
    ----------
    h5_file : file object
             The waveform h5 file handle.
    req_ts_attrs : list
               A list of attribute keys.
    ref_time : float
            Reference time.
    Returns
    -------
    params : dict
             The parameter values at the reference time.
    """

    params = {}

    for key in req_ts_attrs:
        time, var = h5_file.attrs[key]
        RefVal = interp1d(time, var, kind="cubic")[ref_time]
        params.update({key: RefVal})
    return params


def get_ref_vals(
    sim_metadata_object: object,
    req_attrs: list = ["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"],
) -> dict:
    """Get the reference values from a NR HDF5 file
    or a simulation metadata dictionary.

    Parameters
    ----------
    sim_metadata_object : h5 file object, dict
                     The NR h5py file handle or
                     the simulation metadata.
    req_attrs : list
               A list of attribute keys.
    Returns
    -------
    params : dict
             The parameter values at the reference time.
    """
    if isinstance(sim_metadata_object, h5py.File):
        source = sim_metadata_object.attrs

    elif isinstance(sim_metadata_object, dict):
        source = sim_metadata_object
    else:
        raise TypeError("Please supply an open h5py file handle or a dictionary")

    params = {}

    for key in req_attrs:
        RefVal = source[key]
        params.update({key: RefVal})
    return params


def compute_lal_source_frame_from_sxs_metadata(sim_metadata: dict) -> tuple:
    """Compute the LAL source frame vectors at the
    available reference time from the SXS simulation
    metadata.

    Parameters
    ----------
    sim_metadata : dict
               The NR sim_metadata.
    Returns
    -------
    params : dict
             A dictionary containing the LAL source frame
             vectors.
    """
    # Masses
    M1 = sim_metadata["reference_mass1"]
    M2 = sim_metadata["reference_mass2"]
    M = M1 + M2

    # Orbital anglular frequency
    Omega = np.array(sim_metadata["reference_orbital_frequency"])

    # Positions
    Pos1 = np.array(sim_metadata["reference_position1"])
    Pos2 = np.array(sim_metadata["reference_position2"])

    # CoM position
    CoMPos = (M1 * Pos1 + M2 * Pos2) / (M)

    Pos1 = Pos1 - CoMPos
    Pos2 = Pos2 - CoMPos

    # This should point from
    # smaller to larger BH
    # Assumed here is 1 for smaller
    # BH
    DPos = Pos2 - Pos1

    # Oribital angular momentum, Eucledian
    P1 = M1 * np.cross(Omega, Pos1)
    P2 = M2 * np.cross(Omega, Pos2)

    Lbar = np.cross(Pos1, P1) + np.cross(Pos2, P2)

    # Orbital normal LNhat
    LNhat = Lbar / np.linalg.norm(Lbar)
    LNhatx, LNhaty, LNhatz = LNhat

    # Position vector nhat
    nhat = (DPos) / np.linalg.norm(DPos)
    nhatx, nhaty, nhatz = nhat

    params = {
        "LNhatx": LNhatx,
        "LNhaty": LNhaty,
        "LNhatz": LNhatz,
        "nhatx": nhatx,
        "nhaty": nhaty,
        "nhatz": nhatz,
    }

    return params


def compute_lal_source_frame_by_interp(
    h5_file: object, req_ts_attrs: list, t_ref: float
) -> tuple:
    """
    Compute the LAL source frame vectors at a given reference time
    by interpolation of time series data.

    Parameters
    ----------
    h5_file : h5 file object
             The NR h5py file handle that contains the simulation metadata.
    t_ref : float
           The reference time.
    Returns
    -------
    params : dict
             The LAL source frame vectors at the reference time.

    """
    # For reference: attributes required for interpolation.
    # req_ts_attrs = ['LNhatx-vs-time', 'LNhaty-vs-time', 'LNhatz-vs-time', \
    #            'position1x-vs-time', 'position1y-vs-time', 'position1z-vs-time', \
    #            'position2x-vs-time', 'position2y-vs-time', 'position2z-vs-time']

    ref_params = get_interp_ref_values_from_h5_file(h5_file, req_ts_attrs, t_ref)

    # r21 vec
    r21_x = ref_params["position1x-vs-time"] - ref_params["position2x-vs-time"]
    r21_y = ref_params["position1y-vs-time"] - ref_params["position2y-vs-time"]
    r21_z = ref_params["position1z-vs-time"] - ref_params["position2z-vs-time"]

    # Position unit vector nhat
    dPos = np.array([r21_x, r21_y, r21_z])
    nhat = dPos / np.linalg.norm(dPos)
    nhatx, nhaty, nhatz = nhat

    # Orbital unit normal LNhat
    LNhatx = ref_params["LNhatx-vs-time"]
    LNhaty = ref_params["LNhaty-vs-time"]
    LNhatz = ref_params["LNhatz-vs-time"]

    LNhat = np.array([LNhatx, LNhaty, LNhatz])
    LNhat = LNhat / np.linalg.norm(LNhat)
    # LNhatx, LNhaty, LNhatz = LNhat

    # params = {"LNhat": LNhat, "nhat": nhat}
    params = {
        "LNhatx": LNhatx,
        "LNhaty": LNhaty,
        "LNhatz": LNhatz,
        "nhatx": nhatx,
        "nhaty": nhaty,
        "nhatz": nhatz,
    }

    return params


def normalize_metadata(sim_metadata: dict) -> dict:
    """Ensure that the keys of the metadata are
    as required.

    Parameters
    ----------
    sim_metadata : dict
               The NR sim_metadata.

    Returns
    -------
    norm_sim_metadata : dict
               The normalized simulation metadata
    """

    norm_sim_metadata = {}

    for key, val in sim_metadata.items():
        norm_sim_metadata.update({key.replace("relaxed-", ""): val})

    return norm_sim_metadata


def get_ref_time_from_metadata(sim_metadata: dict) -> float:
    """Get the reference time of definition of the LAL
    frame from the simulation metadata, if available.

    Parameters
    ----------
    sim_metadata : dict
                   The simulation metadata.

    Returns
    -------
    t_ref : float
           The reference time
    """

    MdataKeys = list(sim_metadata.keys())

    if "relaxed-time" in MdataKeys:
        # RIT style
        t_ref = sim_metadata["relaxed-time"]
    elif "reference_time" in MdataKeys:
        # SXS Style
        t_ref = sim_metadata["reference_time"]
    else:
        t_ref = -1

    return t_ref


def transform_spins_nr_to_lal(
    nrSpin1: object, nrSpin2: object, n_hat: object, ln_hat: object
) -> tuple:
    """Trnasform the spins of the NR simulation from the
    NR frame to the  frame.
    Parameters
    ---------
    nrSpin1, nrSpin2 : list
             A list of the components of the spins of the objects.
    n_hat, ln_hat : list
             A list of the components of the unit vectors of the objects,
             against which the components of the spins are specified.
    Returns
    -------
    S1, S2 : list
             The transformed spins in LAL frame.
    """
    nrSpin1x, nrSpin1y, nrSpin1z = nrSpin1
    nrSpin2x, nrSpin2y, nrSpin2z = nrSpin2

    n_hat_x, n_hat_y, n_hat_z = n_hat
    ln_hat_x, ln_hat_y, ln_hat_z = ln_hat

    S1x = nrSpin1x * n_hat_x + nrSpin1y * n_hat_y + nrSpin1z * n_hat_z

    S1y = (
        nrSpin1x * (-ln_hat_z * n_hat_y + ln_hat_y * n_hat_z)
        + nrSpin1y * (ln_hat_z * n_hat_x - ln_hat_x * n_hat_z)
        + nrSpin1z * (-ln_hat_y * n_hat_x + ln_hat_x * n_hat_y)
    )

    S1z = nrSpin1x * ln_hat_x + nrSpin1y * ln_hat_y + nrSpin1z * ln_hat_z

    S2x = nrSpin2x * n_hat_x + nrSpin2y * n_hat_y + nrSpin2z * n_hat_z
    S2y = (
        nrSpin2x * (-ln_hat_z * n_hat_y + ln_hat_y * n_hat_z)
        + nrSpin2y * (ln_hat_z * n_hat_x - ln_hat_x * n_hat_z)
        + nrSpin2z * (-ln_hat_y * n_hat_x + ln_hat_x * n_hat_y)
    )

    S2z = nrSpin2x * ln_hat_x + nrSpin2y * ln_hat_y + nrSpin2z * ln_hat_z

    S1 = [S1x, S1y, S1z]
    S2 = [S2x, S2y, S2z]

    return S1, S2


def get_nr_to_lal_rotation_angles(
    h5_file: object,
    sim_metadata: dict,
    inclination: float,
    phi_ref: float = 0,
    f_ref: float | None = None,
    t_ref: float | None = None,
    tol: float = 1e-6,
) -> tuple:
    r"""Get the angular coordinates :math:`\theta, \phi`
    and the rotation angle :math:`\alpha` from the H5 file

    Parameters
    ----------
    h5_file : file object
            The waveform h5 file handle.

    inclination : float
                  The inclination angle.
    phi_ref : float
             The orbital phase at reference time.
    f_ref, t_ref : float, optional
                 The reference orbital frequency or time

    sim_metadata : dict
               The sim_metadata of the waveform file.
    tol : float
          The tolerance to use to allow floating point
            representation errors.

    Returns
    -------
    angles : dict
             The angular corrdinates Theta, Psi,  and the rotation angle Alpha.
             If available, this also contains the reference time and frequency.

    Notes
    -----

    Variable definitions.

    theta : Returned inclination angle of source in NR coordinates.
    psi :   Returned azimuth angle of source in NR coordinates.
    alpha: Returned polarisation angle.
    h5_file: h5py object of the NR HDF5 file.
    inclination: inclination of source in LAL source frame.
    phi_ref: Orbital reference phase.
    t_ref : Reference time. -1 or None indicates it was not found in the sim_metadata.
    f_ref: Reference frequency.

    The reference epoch is defined close to the beginning of the simulation.
    """

    # Compute the angles necessary to rotate from the intrinsic NR source frame
    # into the LAL frame. See DCC-T1600045 for details.

    # Following section IV of DCC-T1600045
    # Step 1: Define Phi = phiref
    orb_phase = phi_ref

    ##########################################
    # Step 2: Compute Zref
    # 2.1 : Check if interpolation is required in IntReq
    # 2.2 : Get/ Compute the basis vectors of the LAL
    #       frame.
    # 2.2.1 : If IntReq=yes, given the reference time, interpolate and get
    #         the required basis vectors in the LAL source frame.
    # 2.2.2 : If no, then check for the default values of the
    #         LAL frame basis vectors in the h5_file
    # 2.2.3 : If the h5_file does not contain the required default
    #         vectors, then raise an error.
    # 2.3 : Carryout vector math to get Zref.
    # 2.1: Compute LN_hat from file. LN_hat = direction of orbital ang. mom.
    # 2.2: Compute n_hat from file. n_hat = direction from object 2 to object 1

    ###########################################
    # Cases
    ###########################################
    # req_def_attrs = ["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"]

    # SXS default attributes required
    # for computing the LAL source frame.
    req_def_attrs_sxs = [
        "reference_time",
        "reference_mass1",
        "reference_mass2",
        "reference_orbital_frequency",
        "reference_position1",
        "reference_position2",
    ]

    # Check if interpolation is necessary
    # if t_ref is supplied

    interp = False

    if not t_ref:
        if not f_ref:
            # No interpolation required. Use available reference values.
            interp = False

        else:
            # Try to get the reference time from orbital frequency
            try:
                t_ref = get_ref_time_from_ref_freq(h5_file, f_ref)

                # Check if interpolation is required
                interp, avail_ref_time = check_interp_req(h5_file, ref_time=t_ref)
            except Exception as excep:
                logger.info(
                    f"Could not obtain reference time from given reference frequency {f_ref}.",
                    excep,
                )
                logger.info("Choosing available reference time")
                interp = False
    else:
        interp, avail_ref_time = check_interp_req(h5_file, ref_time=t_ref)

    if interp is False:
        # Then load default values from the NR data
        # at hard coded reference time.

        # Get the available reference time for book keeping
        t_ref = get_ref_time_from_metadata(sim_metadata)

        # Check for LAL frame in simulation metadata.
        # RIT and GT qualify this.
        # Default attributes in case of no interpolation
        ref_check_def_h5, absent_attrs_h5 = check_nr_attrs(h5_file)

        if ref_check_def_h5 is False:
            # Then the LAL source frame information is not present in the H5 file.
            # Then this could be SXS or GT data. The LAL source frame need to be computed from
            # the H5 File or simulation metadata.

            # Check if LAL source frame info is present in the simulation metadata.
            ref_check_def_meta, absent_attrs_meta = check_nr_attrs(sim_metadata)

            if ref_check_def_meta is False:
                # Then this is SXS data.

                # Check for raw information in metadata to compute the LAL source frame.
                ref_check_def_meta_sxs, absent_attrs_meta_sxs = check_nr_attrs(
                    sim_metadata, req_def_attrs_sxs
                )

                if ref_check_def_meta_sxs is True:
                    # Compute the LAL source frame from simulation metadata
                    ref_params = compute_lal_source_frame_from_sxs_metadata(
                        sim_metadata
                    )
                else:
                    raise Exception(
                        "Insufficient information to compute the LAL source frame."
                        f"\n Missing information is {absent_attrs_meta_sxs}."
                    )
            else:
                # LAL source frame is present in the simulation metadata
                ref_params = get_ref_vals(sim_metadata)
        else:
            ref_params = get_ref_vals(h5_file)

    elif interp is True:
        # Experimental; This assumes all the required atributes  needed
        # to compute the LAL source frame at the given reference time
        # are present in the H5file only.

        # Attributes required for interpolation.
        req_ts_attrs = [
            "LNhatx-vs-time",
            "LNhaty-vs-time",
            "LNhatz-vs-time",
            "position1x-vs-time",
            "position1y-vs-time",
            "position1z-vs-time",
            "position2x-vs-time",
            "position2y-vs-time",
            "position2z-vs-time",
        ]

        # Check if time series data of required reference data is present
        ref_check_interp_req, absent_interp_attrs = check_nr_attrs(
            h5_file, req_ts_attrs
        )

        if ref_check_interp_req is False:
            raise KeyError(
                "Insufficient information to compute the LAL source frame at given reference time."
                f"Missing information is {absent_interp_attrs}."
            )
        else:
            ref_params = compute_lal_source_frame_by_interp(
                h5_file, req_ts_attrs, t_ref
            )

        # Warning 1
        # Implement this Warning
        # XLAL_CHECK( ref_time!=XLAL_FAILURE, XLAL_FAILURE, "Error computing reference time.
        # Try setting fRef equal to the f_low given by the NR simulation or to a value <=0 to deactivate
        # fRef for a non-precessing simulation.\n")

    # Get the LAL source frame vectors
    ln_hat_x = ref_params["LNhatx"]
    ln_hat_y = ref_params["LNhaty"]
    ln_hat_z = ref_params["LNhatz"]

    n_hat_x = ref_params["nhatx"]
    n_hat_y = ref_params["nhaty"]
    n_hat_z = ref_params["nhatz"]

    ln_hat = np.array([ln_hat_x, ln_hat_y, ln_hat_z])
    n_hat = np.array([n_hat_x, n_hat_y, n_hat_z])

    # 2.3: Carryout vector math to get Zref in the lal wave frame
    corb_phase = np.cos(orb_phase)
    sorb_phase = np.sin(orb_phase)
    sinclination = np.sin(inclination)
    cinclination = np.cos(inclination)

    ln_cross_n = np.cross(ln_hat, n_hat)
    ln_cross_n_x, ln_cross_n_y, ln_cross_n_z = ln_cross_n

    z_wave_x = sinclination * (sorb_phase * n_hat_x + corb_phase * ln_cross_n_x)
    z_wave_y = sinclination * (sorb_phase * n_hat_y + corb_phase * ln_cross_n_y)
    z_wave_z = sinclination * (sorb_phase * n_hat_z + corb_phase * ln_cross_n_z)

    z_wave_x += cinclination * ln_hat_x
    z_wave_y += cinclination * ln_hat_y
    z_wave_z += cinclination * ln_hat_z

    z_wave = np.array([z_wave_x, z_wave_y, z_wave_z])
    z_wave = z_wave / np.linalg.norm(z_wave)

    #################################################################
    # Step 3.1: Extract theta and psi from Z in the lal wave frame
    # NOTE: Theta can only run between 0 and pi, so no problem with arccos here
    theta = np.arccos(z_wave_z)

    # Degenerate if Z_wave[2] == 1. In this case just choose psi randomly,
    # the choice will be cancelled out by alpha correction (I hope!)

    # If theta is very close to the poles
    # return a random value
    if abs(z_wave_z - 1.0) < tol:
        psi = 0.5

    else:
        # psi can run between 0 and 2pi, but only one solution works for x and y */
        # Possible numerical issues if z_wave_x = sin(theta) */
        if abs(z_wave_x / np.sin(theta)) > 1.0:
            if abs(z_wave_x / np.sin(theta)) < (1 + 10 * tol):
                # LAL tol retained.
                if (z_wave_x * np.sin(theta)) < 0.0:
                    psi = np.pi

                else:
                    psi = 0.0

            else:
                logger.info(f"z_wave_x = {z_wave_x}")
                logger.info(f"sin(theta) = {np.sin(theta)}")
                raise ValueError(
                    "Z_x cannot be bigger than sin(theta). Please contact the developers."
                )

        else:
            psi = np.arccos(z_wave_x / np.sin(theta))

        y_val = np.sin(psi) * np.sin(theta)

        # If z_wave[1] is negative, flip psi so that sin(psi) goes negative
        # while preserving cos(psi) */
        if z_wave_y < 0.0:
            psi = 2 * np.pi - psi
            y_val = np.sin(psi) * np.sin(theta)

        if abs(y_val - z_wave_y) > (5e3 * tol):
            # LAL tol retained.
            logger.info(f"orb_phase = {orb_phase}")
            logger.info(
                f"y_val = {y_val}, z_wave_y = {z_wave_y}, abs(y_val - z_wave_y) = {abs(y_val - z_wave_y)}"
            )
            raise ValueError("Math consistency failure! Please contact the developers.")

    # 3.2: Compute the vectors theta_hat and psi_hat
    # stheta = np.sin(theta)
    # ctheta = np.cos(theta)

    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    # theta_hat_x = cpsi * ctheta
    # theta_hat_y = spsi * ctheta
    # theta_hat_z = -stheta
    # theta_hat = np.array([theta_hat_x, theta_hat_y, theta_hat_z])

    psi_hat_x = -spsi
    psi_hat_y = cpsi
    psi_hat_z = 0.0
    psi_hat = np.array([psi_hat_x, psi_hat_y, psi_hat_z])

    # Step 4: Compute sin(alpha) and cos(alpha)
    # Rotation angles on the tangent plane
    # due to spin weight.

    # n_dot_theta = np.dot(n_hat, theta_hat)
    # ln_cross_n_dot_theta = np.dot(ln_cross_n, theta_hat)

    n_dot_psi = np.dot(n_hat, psi_hat)
    ln_cross_n_dot_psi = np.dot(ln_cross_n, psi_hat)

    # salpha = corb_phase * n_dot_theta - sorb_phase * ln_cross_n_dot_theta
    calpha = corb_phase * n_dot_psi - sorb_phase * ln_cross_n_dot_psi

    if abs(calpha) > 1:
        calpha_err = abs(calpha) - 1
        if calpha_err < tol:
            # This tol could have been much smaller.
            # Just resuing the default for now.
            logger.info(
                f"Correcting the polarization angle for finite precision error {calpha_err}"
            )
            calpha = calpha / abs(calpha)
        else:
            raise ValueError(
                "Seems like something is wrong with the polarization angle. Please contact the developers!"
            )

    alpha = np.arccos(calpha)

    angles = {
        "theta": theta,
        "psi": psi,
        "alpha": alpha,
        "t_ref": t_ref,
        "f_ref": f_ref,
    }

    return angles
