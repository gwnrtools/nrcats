"""Test the generation of observer polarizations
of nrcats against lal and waveformtools
using GT waveforms.
"""

import os
import sys

import h5py
import numpy as np

cwd = os.getcwd()
from pathlib import Path

home = Path.home()

libpath = f"{cwd}/../"

if libpath not in sys.path:
    sys.path.append(libpath)

import unittest
from pathlib import Path

from nrcats.maya import MayaCatalog
from pycbc import pnutils
from nrcats.lvc import transform_spins_nr_to_lal

from pycbc.filter.matchedfilter import match

# pycbc
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform

try:
    from waveformtools.waveforms import ModesArray as modes_array
    from waveformtools.waveformtools import message, roll, xtract_camp_phase

    _WAVEFORMTOOLS_AVAILABLE = True
except ImportError:
    _WAVEFORMTOOLS_AVAILABLE = False
    # Stubs so the rest of the module parses without error
    modes_array = None

    def message(*args, **kwargs):
        pass

    def roll(a, shift):
        return a

    def xtract_camp_phase(real, imag):
        return None, None


# unittest helper funcs — may be unavailable when test/ is not on sys.path
try:
    from helper import rms_errs

    _HELPER_AVAILABLE = True
except ImportError:
    _HELPER_AVAILABLE = False

    def rms_errs(*args, **kwargs):
        return 0, 0, 0


_VALIDATION_DIR = Path(__file__).parent / "validation_data"

sim_name = "GT0001"

# Parameters
total_mass = 40
distance = 1000
delta_t = 1.0 / (4 * 2048)

message(f"Simulation {sim_name}", message_verbosity=2)

wftools_use_lal_conven = True
lal_use_coa_phase_as_phi_ref = False
test_wrt_wftools = False


def GetPolsToCompare(sim_name, total_mass, distance, inclination, coa_phase, delta_t):
    """Get polarization time-series to compare from
    nrcats, waveformtools and lal methods.

    Parameters
    ----------
    sim_name : str
               The simulation name
    total_mass : float
                 The total mass in solar mass
                 units.
    distance : float
               The distance in Mpc
    inclination : float
                  The inclination angle in radians
    coa_phase : float
                The coalescence phase in radians
    delta_t : float
              The sampling time step in
              seconds.
    Returns
    -------
    waveforms : dict
                The waveform polarizations from nrcats,
                waveformtools and lal.
    """
    mc = MayaCatalog.load(verbosity=1, download=False)

    mwf1 = mc.get(sim_name)

    hpc = mwf1.get_td_waveform(
        total_mass=total_mass,
        distance=distance,
        inclination=inclination,
        coa_phase=coa_phase,
        delta_t=delta_t,
    )
    hpc_pycbc = hpc
    hp_n, hx_n = hpc_pycbc.real(), hpc_pycbc.imag()

    time_n = hp_n.sample_times

    # Recenter
    mtime = time_n[np.argmax(np.asarray(hp_n) ** 2 + np.asarray(hx_n) ** 2)]
    time_n -= mtime

    phi_ref_obs = mwf1.get_obs_phi_ref_from_obs_coa_phase(coa_phase)

    if lal_use_coa_phase_as_phi_ref is False:
        lal_coa_phase = phi_ref_obs

    fdir = str(_VALIDATION_DIR)
    fname = f"{sim_name}.h5"
    file = str(_VALIDATION_DIR / fname)

    f = h5py.File(file, "a")

    # Extrinsic parameters:
    f_lower = 1
    f_lower_at_1MSUN = f_lower / total_mass
    if "f_lower_at_1MSUN" not in list(f.attrs.keys()):
        f.attrs["f_lower_at_1MSUN"] = f_lower_at_1MSUN
        f.close()
    else:
        f.close()

    f = h5py.File(file, "r")

    message(
        "All attributes in source h5 file", list(f.attrs.keys()), message_verbosity=3
    )
    message("All keys in source h5 file", f.keys(), message_verbosity=3)

    params = {}

    params["f_lower"] = f_lower
    params["mtotal"] = total_mass  # 150.0
    params["inclination"] = inclination  # 0.0
    params["distance"] = distance  # 100.0

    # Metadata parameters:
    params["eta"] = f.attrs["eta"]

    params["mass1"] = pnutils.mtotal_eta_to_mass1_mass2(
        params["mtotal"], params["eta"]
    )[0]
    params["mass2"] = pnutils.mtotal_eta_to_mass1_mass2(
        params["mtotal"], params["eta"]
    )[1]

    # BH1 spins
    params["spin1x"] = f.attrs["spin1x"]
    params["spin1y"] = f.attrs["spin1y"]
    params["spin1z"] = f.attrs["spin1z"]

    # BH2 spins
    params["spin2x"] = f.attrs["spin2x"]
    params["spin2y"] = f.attrs["spin2y"]
    params["spin2z"] = f.attrs["spin2z"]

    # Spin unit vectors
    params["nhat"] = [f.attrs["nhatx"], f.attrs["nhaty"], f.attrs["nhatz"]]
    params["lnhat"] = [f.attrs["LNhatx"], f.attrs["LNhaty"], f.attrs["LNhatz"]]

    # Check for coa_phase, else use the phase from nr cat load.
    try:
        params["coa_phase"] = f.attrs["coa_phase"]
        raise Exception["NR coa phase is present!"]
    except Exception as excep:
        message(
            f"Cannot find the attribute `coa_phase` in the file. Setting to {coa_phase}",
            excep,
            message_verbosity=2,
        )
        params["coa_phase"] = lal_coa_phase

    # Transform spins

    # NR frame
    s1 = [params["spin1x"], params["spin1y"], params["spin1z"]]
    s2 = [params["spin2x"], params["spin2y"], params["spin2z"]]

    # LAL frame
    S1, S2 = transform_spins_nr_to_lal(s1, s2, params["nhat"], params["lnhat"])

    message("Loading waveform through PyCBC", message_verbosity=3)

    hp_l, hx_l = get_td_waveform(
        approximant="NR_hdf5",
        numrel_data=file,
        mass1=params["mass1"],
        mass2=params["mass2"],
        spin1x=S1[0],
        spin1y=S1[1],
        spin1z=S1[2],
        spin2x=S2[0],
        spin2y=S2[1],
        spin2z=S2[2],
        delta_t=delta_t,
        f_lower=f_lower,
        inclination=params["inclination"],
        coa_phase=params["coa_phase"],
        distance=params["distance"],
    )

    time_l = np.array(range(len(hp_l))) * delta_t

    # Recenter
    mtime = time_l[np.argmax(hp_l**2 + hx_l**2)]
    time_l -= mtime
    f.close()

    resam_type = "auto"
    angles = mwf1.get_angles(inclination=inclination, coa_phase=coa_phase)

    # Default 3rd order interp1d
    wfm3 = modes_array(label="GT1 3", data_dir=fdir, file_name=fname)

    wfm3.load_modes(
        ftype="GT", var_type="Strain", resam_type=resam_type, interp_kind="cubic"
    )

    time_w3, hp_w3, hx_w3 = wfm3.to_td_waveform(
        Mtotal=total_mass,
        distance=distance,
        theta=angles["theta"],
        phi=angles["psi"],
        alpha=angles["alpha"],
        delta_t=delta_t,
        method="fast",
        k=None,
    )

    if wftools_use_lal_conven is True:
        hp_w3, hx_w3 = hp_w3, -hx_w3

    # Recenter
    mtime = time_w3[np.argmax(hp_w3**2 + hx_w3**2)]
    time_w3 -= mtime

    # Prepare waveforms
    wf_n = np.asarray(hp_n) + 1j * np.asarray(hx_n)
    wf_l = np.asarray(hp_l) + 1j * np.asarray(hx_l)
    wf_w3 = hp_w3 + 1j * hx_w3

    a_n, p_n = xtract_camp_phase(wf_n.real, wf_n.imag)
    a_l, p_l = xtract_camp_phase(wf_l.real, wf_l.imag)
    a_w3, p_w3 = xtract_camp_phase(wf_w3.real, wf_w3.imag)

    imax_n, imax_l, imax_w3 = np.argmax(a_n), np.argmax(a_l), np.argmax(a_w3)

    message("Max locations before centering", message_verbosity=3)
    message(imax_n, imax_l, imax_w3, message_verbosity=3)

    shift = imax_w3 - imax_n

    c = 0
    if c < 1:
        a_w3 = roll(a_w3, shift)
        p_w3 = roll(p_w3, shift)
        c = 1

    imax_n, imax_l, imax_w3 = np.argmax(a_n), np.argmax(a_l), np.argmax(a_w3)

    message("Max locations after centering", message_verbosity=3)
    message(imax_n, imax_l, imax_w3, message_verbosity=3)

    if (imax_w3 - imax_l) != 0 or (imax_n - imax_l) != 0:
        raise Exception("The waveforms need to be centered!")

    # Recenter time axis
    t_l = np.array(time_l - time_l[imax_l])
    t_n = np.array(time_n - time_n[imax_n])
    t_w3 = np.array(time_w3 - time_w3[imax_w3])

    wf_n = a_n * np.exp(1j * p_n)
    wf_l = a_l * np.exp(1j * p_l)
    wf_w3 = a_w3 * np.exp(1j * p_w3)

    return {
        "nrcat": [t_n, wf_n, a_n, p_n],
        "wftools": [t_w3, wf_w3, a_w3, p_w3],
        "lal": [t_l, wf_l, a_l, p_l],
    }


@unittest.skipUnless(
    _WAVEFORMTOOLS_AVAILABLE and _HELPER_AVAILABLE,
    "waveformtools.waveforms.ModesArray or test helper not available",
)
class TestGTPol(unittest.TestCase):
    """Test the computation of polarizattions"""

    def test_pol(self):
        """Test the computations of GT waveforms from nrcats
        against  waveformtools and lal. Tested are RMS errors, maximum deviation and mismatches.
        """
        validation_file = _VALIDATION_DIR / f"{sim_name}.h5"
        if not validation_file.exists():
            self.skipTest(
                f"Validation data not found: {validation_file}. "
                "Provide the file to run this test."
            )

        inc_angles = [
            0.0015,
            0.5,
            1,
            1.5,
            np.pi / 2 - 0.001,
            np.pi / 2,
            np.pi / 2 + 0.0001,
            2.5,
            3,
            np.pi - 0.0015,
        ]

        coa_phases = np.linspace(0, 2 * np.pi, 10)

        all_mm = []

        for inclination in inc_angles:
            for coa_phase in coa_phases:
                message("\n--------------------------", message_verbosity=1)
                message(
                    f"inclination {inclination} Coa phase {coa_phase}",
                    message_verbosity=1,
                )
                message("--------------------------", message_verbosity=1)

                try:
                    waveforms = GetPolsToCompare(
                        sim_name=sim_name,
                        total_mass=total_mass,
                        inclination=inclination,
                        coa_phase=coa_phase,
                        delta_t=delta_t,
                        distance=distance,
                    )
                except (RuntimeError, ValueError) as exc:
                    self.skipTest(
                        f"Required data not available (missing catalog cache or data): {exc}"
                    )
                # Prepare TimeSeries
                t_n, wf_n, a_n, p_n = waveforms["nrcat"]

                hp_n = TimeSeries(wf_n.real, delta_t)
                hx_n = TimeSeries(wf_n.imag, delta_t)

                t_l, wf_l, a_l, p_l = waveforms["lal"]

                hp_l = TimeSeries(wf_l.real, delta_t)
                hx_l = TimeSeries(wf_l.imag, delta_t)

                t_w3, wf_w3, a_w3, p_w3 = waveforms["wftools"]

                hp_w3 = TimeSeries(wf_w3.real, delta_t)
                hx_w3 = TimeSeries(wf_w3.imag, delta_t)

                # L2 errors
                Res_l, Amin_l, Amax_l = rms_errs(np.array(wf_n), np.array(wf_l))
                Res_w, Amin_w, Amax_w = rms_errs(np.array(wf_n), np.array(wf_l))

                # Match
                mp_nl = match(hp_n, hp_l)[0]
                mx_nl = match(hx_n, hx_l)[0]
                m_nl = min(mp_nl, mx_nl)
                mm_nl = 100 * (1 - m_nl)

                all_mm.append(mm_nl)

                mp_w3n = match(hp_w3, hp_n)[0]
                mx_w3n = match(hx_w3, hp_n)[0]
                m_w3n = min(mp_w3n, mx_w3n)
                mm_w3n = 100 * (1 - m_w3n)

                message(
                    "-------------------------------------------------------",
                    message_verbosity=1,
                )
                message(f"Mismatches are nrcats - lal {mm_nl}%", message_verbosity=1)
                if test_wrt_wftools:
                    message("nrcats-wftools {mm_w3n}%", message_verbosity=1)
                message(
                    "-------------------------------------------------------",
                    message_verbosity=1,
                )

                prec = 1
                # RMS error should be less than 0.01 x Amax(wf1)
                self.assertAlmostEqual(
                    Res_l,
                    0,
                    prec,
                    "The RMS error wrt LAL waveform  must be"
                    f"atmost 1e-{prec} times Max amplitude of the normalized waveform",
                )

                prec = 1
                # Max relative point-wise deviation w.r.t Amax(wf1) should be less than 1 (100)%
                self.assertAlmostEqual(
                    np.absolute(Amin_l),
                    0,
                    prec,
                    f"The maximum lower deviation wrt LAL waveform must be almost 1e-{prec}%",
                )
                self.assertAlmostEqual(
                    np.absolute(Amax_l),
                    0,
                    prec,
                    f"The maximum upper deviation wrt LAL waveforms must be almost 1e-{prec}%",
                )

                prec = 0
                # Mismatch should be less than 1e-3 or 1e-1 %
                self.assertAlmostEqual(
                    mm_nl,
                    0,
                    prec,
                    f"The mismatch against the LAL waveforms must be atmost 1e-{prec}%",
                )

                prec = 0
                # Full arrays must agree pointwiswe
                np.testing.assert_almost_equal(
                    np.array(wf_n),
                    np.array(wf_l),
                    prec,
                    f"The arrays (nrcats, lal) must equal atleast upto {prec} decimals",
                )

                if test_wrt_wftools is True:
                    prec = 2
                    self.assertAlmostEqual(
                        Res_w,
                        0,
                        prec,
                        "The RMS error wrt waveformtools must be"
                        f"almost 1e-{prec} times Max amplitude of the normalized waveform",
                    )

                    prec = 0
                    self.assertAlmostEqual(
                        np.absolute(Amax_w),
                        0,
                        prec,
                        f"The maximum upper deviation wrt the waveformtools waveforms must be almost 1e-{prec}",
                    )

                    self.assertAlmostEqual(
                        np.absolute(Amax_w),
                        0,
                        prec,
                        f"The maximum upper deviation wrt to the waveformtools waveform must be almost 1e-{prec}",
                    )

                    prec = 1
                    self.assertAlmostEqual(
                        mm_w3n,
                        0,
                        prec,
                        f"The mismatch against the waveformtools waveforms must be atmost 1e-{prec}%",
                    )

                    prec = 2
                    # Full arrays must agree pointwiswe
                    np.testing.assert_almost_equal(
                        np.array(wf_n),
                        np.array(wf_w3),
                        prec,
                        f"The arrays (nrcats, waveformtools) must equal atleast upto {prec} decimals",
                    )

        max_mm = max(all_mm)

        message(f"Max mismatch from this set is {max_mm}", message_verbosity=1)

    if __name__ == "__main__":
        unittest.main(argv=["first-arg-is-ignored"], exit=False, verbosity=3)
