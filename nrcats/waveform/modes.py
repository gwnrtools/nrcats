"""WaveformModes class and related helpers."""

import logging

logger = logging.getLogger(__name__)

import warnings

import h5py
import lal
import numpy as np
import quaternionic
import spherical
from pycbc.types import TimeSeries
from pycbc.waveform import frequency_from_polarizations
from scipy.interpolate import InterpolatedUnivariateSpline
from sxs import TimeSeries as sxs_TimeSeries
from sxs import WaveformModes as sxs_WaveformModes

from nrcats import metadata as md
from nrcats import utils
from nrcats.lvc import (
    check_interp_req,
    get_nr_to_lal_rotation_angles,
    get_ref_vals,
)
from nrcats.waveform.units import _modal_dt


class WaveformModes(sxs_WaveformModes):
    """Catalog-agnostic container for spin-weighted spherical-harmonic waveform modes.

    Inherits from ``sxs.WaveformModes`` (itself an ``numpy.ndarray`` subclass)
    so that instances *are* NumPy arrays.  This is an **intentional design
    choice**, not technical debt, motivated by three requirements:

    1. **Zero-copy performance.**  Mismatch calculations (``match_single_mode``,
       ``match_sphere_averaged``, BMS supertranslation optimization) pass mode
       data directly to PyCBC and SciPy routines that expect array-protocol
       objects.  Inheritance lets NumPy hand them the underlying buffer without
       an intermediate copy.

    2. **Wigner-rotation reuse.**  The parent class exposes ``evaluate()``,
       ``index()``, ``LM``, and Wigner-D rotation infrastructure from the
       ``sxs`` / ``spherical`` stack.  Inheriting avoids re-implementing or
       wrapping that non-trivial mathematics.

    3. **Downstream compatibility.**  Research workflows in PyCBC, ``scri``,
       and user scripts rely on ``isinstance(wfm, sxs.WaveformModes)``
       checks and on standard NumPy slicing semantics.  Breaking that
       contract would impose migration costs across the gravitational-wave
       community.

    **Attribute propagation.**  Because ``numpy.ndarray`` subclasses lose
    plain instance attributes during slicing and view-casting, all custom
    state (``_filepath``, ``_present_modes``, ``_peak_time_22``,
    ``_t_ref_nr``, ``verbosity``) is stored inside the ``_metadata`` dict
    that ``sxs.TimeSeries`` already propagates.  Property descriptors
    provide transparent read/write access.  See ``_custom_meta_keys``,
    ``__array_finalize__``, ``__copy__``, and ``__deepcopy__`` for details.
    """

    # Custom keys stored inside ``_metadata`` so they survive the
    # ``sxs.TimeSeries._slice`` → ``type(self)(new_data, **metadata)``
    # reconstruction path.  Each maps to a factory producing a safe default.
    _custom_meta_keys = {
        "_filepath": lambda: None,
        "_present_modes": set,
        "_peak_time_22": lambda: None,
        "_t_ref_nr": lambda: None,
        "verbosity": lambda: 0,
    }

    def __new__(
        cls,
        data,
        time=None,
        time_axis=0,
        modes_axis=1,
        ell_min=2,
        ell_max=4,
        verbosity=0,
        **w_attributes,
    ) -> None:
        # Pull custom keys out of w_attributes so they don't confuse the
        # parent constructor, then re-inject them into _metadata afterwards.
        custom_vals = {}
        for key in list(cls._custom_meta_keys):
            if key in w_attributes:
                custom_vals[key] = w_attributes.pop(key)

        self = super().__new__(
            cls,
            data,
            time=time,
            time_axis=time_axis,
            modes_axis=modes_axis,
            ell_min=ell_min,
            ell_max=ell_max,
            **w_attributes,
        )

        # Store custom attrs inside _metadata.
        self._metadata.setdefault("_filepath", custom_vals.get("_filepath", None))
        self._metadata.setdefault(
            "_present_modes", custom_vals.get("_present_modes", set())
        )
        self._metadata.setdefault(
            "_peak_time_22", custom_vals.get("_peak_time_22", None)
        )
        self._metadata.setdefault("_t_ref_nr", custom_vals.get("_t_ref_nr", None))
        self._metadata.setdefault("verbosity", custom_vals.get("verbosity", verbosity))
        return self

    def __getitem__(self, key):
        """Extract a slice of this object.

        Overrides parent `__getitem__` to fix an issue where slicing `WaveformModes`
        loses `time_axis` or custom metadata due to how `ndarray.view()` interacts
        with `__array_finalize__` in some versions of `sxs`.
        """
        if isinstance(key, str):
            return super().__getitem__(key)

        obj, time_key = self._slice(key)
        if time_key is None:
            raise ValueError(f"Fancy indexing (with {key}) is not supported")

        result = obj.view(type(self))
        # Explicitly preserve the metadata that `_slice` carefully constructed.
        # This prevents `__array_finalize__` from reverting or stripping keys.
        result._metadata = getattr(obj, "_metadata", {}).copy()

        return result

    # -- Attribute-propagation machinery (REQ-3.2) -------------------------
    #
    # All custom state lives in ``_metadata`` so it naturally travels through
    # the ``sxs.TimeSeries._slice`` reconstruction path.  We expose
    # convenient instance-level accessors that read/write ``_metadata``.

    @property
    def _filepath(self):
        return self._metadata.get("_filepath")

    @_filepath.setter
    def _filepath(self, value):
        self._metadata["_filepath"] = value

    @property
    def _present_modes(self):
        return self._metadata.get("_present_modes", set())

    @_present_modes.setter
    def _present_modes(self, value):
        self._metadata["_present_modes"] = value

    @property
    def _peak_time_22(self):
        return self._metadata.get("_peak_time_22")

    @_peak_time_22.setter
    def _peak_time_22(self, value):
        self._metadata["_peak_time_22"] = value

    @property
    def _t_ref_nr(self):
        return self._metadata.get("_t_ref_nr")

    @_t_ref_nr.setter
    def _t_ref_nr(self, value):
        self._metadata["_t_ref_nr"] = value

    @property
    def verbosity(self):
        return self._metadata.get("verbosity", 0)

    @verbosity.setter
    def verbosity(self, value):
        self._metadata["verbosity"] = value

    def __array_finalize__(self, obj):
        """Propagate ``_metadata`` (including custom keys) from *obj*.

        Delegates to the parent ``sxs.TimeSeries.__array_finalize__`` which
        handles the core ``_metadata`` dict copy.  Then ensures our custom
        keys have safe defaults if they were absent on the source object
        (e.g. view-casting from a plain ndarray).
        """
        super().__array_finalize__(obj)
        if obj is None:
            return
        for key, default_factory in self._custom_meta_keys.items():
            self._metadata.setdefault(key, default_factory())

    def __copy__(self):
        """Shallow copy that duplicates mutable custom containers."""

        result = super().__copy__()
        # Shallow-copy mutable containers to break aliasing.
        pm = result._metadata.get("_present_modes")
        if isinstance(pm, (set, dict, list)):
            result._metadata["_present_modes"] = pm.copy()
        return result

    def __deepcopy__(self, memo):
        """Deep copy that deeply duplicates custom metadata entries."""
        import copy as _copy_mod

        result = super().__deepcopy__(memo)
        for key in self._custom_meta_keys:
            val = result._metadata.get(key)
            if val is not None:
                result._metadata[key] = _copy_mod.deepcopy(val, memo)
        return result

    # -- End attribute-propagation machinery --------------------------------

    @classmethod
    def _load(
        cls,
        data,
        time=None,
        time_axis=0,
        modes_axis=1,
        ell_min=2,
        ell_max=4,
        verbosity=0,
        **w_attributes,
    ):
        if time is None:
            time = np.arange(0, len(data[:, 0]))
        return cls(
            data,
            time=time,
            time_axis=time_axis,
            modes_axis=modes_axis,
            ell_min=ell_min,
            ell_max=ell_max,
            verbosity=verbosity,
            **w_attributes,
        )

    @classmethod
    def load_from_h5(cls, file_path_or_open_file, metadata={}, verbosity=0):
        """Load SWSH waveform modes from an HDF5 file (RIT/MAYA catalog format).

        See ``nrcats.waveform.loaders.load_from_h5`` for full docs.
        """
        from nrcats.waveform.loaders import load_from_h5 as _impl

        return _impl(cls, file_path_or_open_file, metadata, verbosity)

    @classmethod
    def load_from_targz(cls, file_path, metadata={}, verbosity=0):
        """Load SWSH waveform modes from a ``.tar.gz`` archive (RIT psi4 format).

        See ``nrcats.waveform.loaders.load_from_targz`` for full docs.
        """
        from nrcats.waveform.loaders import load_from_targz as _impl

        return _impl(cls, file_path, metadata, verbosity)

    @property
    def filepath(self):
        """Return the data file path"""
        if not self._filepath:
            self._filepath = self.sim_metadata["waveform_data_location"]
        return self._filepath

    @property
    def sim_metadata(self):
        """Return the simulation metadata dictionary"""
        return self._metadata["metadata"]

    @property
    def metadata(self):
        """Return the simulation metadata dictionary"""
        return self.sim_metadata

    def _get_label_params(self):
        """Extract mass ratio and spin components from metadata in a
        catalog-agnostic way. Returns (q, s1x, s1y, s1z, s2x, s2y, s2z)
        using whichever metadata keys are present (RIT, MAYA, or SXS)."""
        meta = self.metadata
        try:
            if "relaxed_mass_ratio_1_over_2" in meta:
                q = meta["relaxed_mass_ratio_1_over_2"]
                s1x, s1y, s1z = (
                    meta.get("relaxed_chi1x", float("nan")),
                    meta.get("relaxed_chi1y", float("nan")),
                    meta.get("relaxed_chi1z", float("nan")),
                )
                s2x, s2y, s2z = (
                    meta.get("relaxed_chi2x", float("nan")),
                    meta.get("relaxed_chi2y", float("nan")),
                    meta.get("relaxed_chi2z", float("nan")),
                )
            elif "q" in meta:
                q = meta["q"]
                s1x, s1y, s1z = (
                    meta.get("a1x", float("nan")),
                    meta.get("a1y", float("nan")),
                    meta.get("a1z", float("nan")),
                )
                s2x, s2y, s2z = (
                    meta.get("a2x", float("nan")),
                    meta.get("a2y", float("nan")),
                    meta.get("a2z", float("nan")),
                )
            elif "reference_mass_ratio" in meta:
                q = meta["reference_mass_ratio"]
                sp1 = meta.get("reference_dimensionless_spin1", [float("nan")] * 3)
                sp2 = meta.get("reference_dimensionless_spin2", [float("nan")] * 3)
                s1x, s1y, s1z = sp1[0], sp1[1], sp1[2]
                s2x, s2y, s2z = sp2[0], sp2[1], sp2[2]
            else:
                q = float("nan")
                s1x = s1y = s1z = s2x = s2y = s2z = float("nan")
        except Exception:
            q = float("nan")
            s1x = s1y = s1z = s2x = s2y = s2z = float("nan")
        return q, s1x, s1y, s1z, s2x, s2y, s2z

    @property
    def label(self):
        """Return a LaTeX label summarizing key simulation parameters."""
        q, s1x, s1y, s1z, s2x, s2y, s2z = self._get_label_params()
        return (
            f"$q{q:0.3f}\\_"
            f"\\chi_A{s1x:0.3f}\\_{s1y:0.3f}\\_{s1z:0.3f}\\_\\_"
            f"\\chi_B{s2x:0.3f}\\_{s2y:0.3f}\\_{s2z:0.3f}$"
        )

    @property
    def label_nolatex(self):
        """Return a plain-text label summarizing key simulation parameters."""
        q, s1x, s1y, s1z, s2x, s2y, s2z = self._get_label_params()
        return (
            f"q{q:0.3f}-sA{s1x:0.3f}-{s1y:0.3f}-{s1z:0.3f}"
            f"--sB{s2x:0.3f}-{s2y:0.3f}-{s2z:0.3f}"
        )

    def get_parameters(self, total_mass: float = 1.0) -> dict:
        """Return the initial physical parameters for the simulation.

        Args:
            total_mass (float, optional): Total Mass of Binary (solar masses).

        Returns:
            dict: Initial binary parameters compatible with PyCBC.
        """
        metadata = self.metadata
        parameters = md.get_source_parameters_from_metadata(
            metadata, total_mass=total_mass
        )
        catalog_type = metadata.get("catalog_type")
        if catalog_type in ("RIT", "MAYA"):
            if parameters["f_lower"] == -1:
                h = self.get_mode(
                    2, 2, total_mass, distance=1, delta_t_seconds=1.0 / 8192
                )
                fr = frequency_from_polarizations(h.real(), -h.imag())
                parameters.update(f_lower=fr[0])
        elif catalog_type == "SXS":
            if parameters["f_lower"] == -1:
                delta_t_secs = 1.0 / 8192
                h = self.get_mode(
                    2, 2, total_mass, distance=1, delta_t_seconds=delta_t_secs
                )
                fr = frequency_from_polarizations(h.real(), -h.imag())
                reference_time_idx = int(
                    np.round(
                        (metadata["reference_time"] / (total_mass * lal.MTSUN_SI))
                        / delta_t_secs
                    )
                )
                parameters.update(f_lower=fr[reference_time_idx])
        return parameters

    def get_mode_data(self, ell, em):
        return self[f"Y_l{ell}_m{em}.dat"]

    def get_mode(
        self,
        ell,
        em,
        total_mass=1.0,
        distance=1.0,
        delta_t=None,
        to_pycbc=True,
        delta_t_seconds=None,
        delta_t_Msun=None,
    ):
        """Return a single (ℓ, m) waveform mode, rescaled to physical units.

        Parameters
        ----------
        ell, em : int
            Spherical-harmonic indices.
        total_mass : float, optional
            Total mass in solar masses (default 1).
        distance : float, optional
            Luminosity distance in Mpc (default 1).
        delta_t_seconds : float, optional
            Sample spacing in physical seconds.  Mutually exclusive with
            ``delta_t_Msun``.
        delta_t_Msun : float, optional
            Sample spacing in dimensionless M units.  Mutually exclusive with
            ``delta_t_seconds``.
        delta_t : float, optional
            *Deprecated.* Use ``delta_t_seconds`` or ``delta_t_Msun`` instead.
        to_pycbc : bool, optional
            Return a ``pycbc.types.TimeSeries`` (default True).

        Returns
        -------
        pycbc.types.TimeSeries or sxs.TimeSeries
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
                    stacklevel=2,
                )
            else:
                delta_t = _modal_dt(self.time)
            if delta_t > 1.0 / 128:
                dt_dimless = delta_t
                dt_physical = delta_t * m_secs
            else:
                dt_physical = delta_t
                dt_dimless = delta_t / m_secs

        new_time = np.arange(min(self.time), max(self.time), dt_dimless)

        mode_data = np.array(self.data[:, self.index(ell, em)], dtype=complex)
        mode_ts = sxs_TimeSeries(mode_data, time=self.time)
        interpolated_mode_ts = mode_ts.interpolate(new_time)

        h_mode_complex = np.array(interpolated_mode_ts.data, dtype=complex)
        h_mode_complex *= utils.amp_to_physical(total_mass, distance)

        peak_time_sec = self.peak_time_22 * m_secs
        start_time_sec = new_time[0] * m_secs
        epoch = start_time_sec - peak_time_sec

        retval = self.to_pycbc(
            input_array=h_mode_complex,
            delta_t=dt_physical,
            epoch=epoch,
        )
        if not to_pycbc:
            retval = sxs_TimeSeries(retval.data, time=retval.sample_times)
        return retval

    def f_lower_at_1Msun(self, t=None):
        """Return the instantaneous GW frequency of the (2,2) mode at 1 M☉.

        Parameters
        ----------
        t : float or None, optional
            Evaluation time in dimensionless M units.  If None, returns the
            frequency at the first sample.

        Returns
        -------
        float
            GW frequency in Hz at 1 M☉.  Divide by ``total_mass`` [M☉] to
            get physical Hz.
        """
        mode22 = self.get_mode_data(2, 2)
        fr22 = frequency_from_polarizations(
            TimeSeries(mode22[:, 1], delta_t=np.diff(self.time)[0]),
            TimeSeries(-1 * mode22[:, 2], delta_t=np.diff(self.time)[0]),
        )
        fr22 = np.abs(fr22)
        if t is None:
            return float(fr22[0] / lal.MTSUN_SI)
        sample_times = self.time[: len(fr22)]
        interp_fr22 = InterpolatedUnivariateSpline(sample_times, fr22, k=3)
        return float(interp_fr22(t) / lal.MTSUN_SI)

    def _get_relaxation_time_dimless(self):
        """Return the relaxation time in dimensionless M units from metadata."""
        meta = self.sim_metadata
        for key in ("relaxed-time", "relaxation_time", "reference_time"):
            if key in meta and meta[key] is not None:
                return float(meta[key])
        return 0.0

    def trim_to_relaxation_time(self, total_mass, delta_t=1.0 / 4096):
        """Return the (2,2) mode trimmed to start at the relaxation epoch.

        Parameters
        ----------
        total_mass : float
            Total mass of the binary (solar masses).
        delta_t : float, optional
            Sample spacing in seconds (default 1/4096).

        Returns
        -------
        pycbc.types.TimeSeries
        """
        t_relax = self._get_relaxation_time_dimless()
        mode = self.get_mode(
            2, 2, total_mass=total_mass, distance=1.0, delta_t_seconds=delta_t
        )
        t_start = mode.start_time
        t_relax_phys = t_relax * utils.time_to_physical(total_mass)
        idx = 0
        for i, t in enumerate(mode.sample_times):
            if t >= t_start + t_relax_phys:
                idx = i
                break
        return mode[idx:]

    def f_lower_at_relaxation(self, total_mass):
        """Return the GW frequency at the relaxation epoch, in Hz.

        Parameters
        ----------
        total_mass : float
            Total mass of the binary (solar masses).

        Returns
        -------
        float
        """
        t_relax = self._get_relaxation_time_dimless()
        t_eval = self.time[0] + t_relax
        return self.f_lower_at_1Msun(t=t_eval) / total_mass

    def get_polarizations(
        self, inclination, coa_phase, f_ref=None, t_ref=None, tol=1e-6
    ):
        """Sum over modes and return plus/cross GW polarizations.

        Parameters
        ----------
        inclination : float
            Inclination angle (radians).
        coa_phase : float
            Coalescence orbital phase (radians).
        tol : float, optional
            Floating-point tolerance for rotation angle computation (1e-6).
        """
        angles = self.get_angles(inclination, coa_phase, f_ref, t_ref, tol)
        return self.evaluate([angles["theta"], angles["psi"], angles["alpha"]])

    def get_td_waveform(
        self,
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
    ):
        """Sum over modes and return GW polarizations rescaled to physical units.

        Parameters
        ----------
        total_mass : float
            Total mass (solar masses).
        distance : float
            Luminosity distance (megaparsecs).
        inclination : float
            Inclination angle (radians).
        coa_phase : float
            Coalescence orbital phase (radians).
        delta_t_seconds : float, optional
            Sample spacing in physical seconds.
        delta_t_Msun : float, optional
            Sample spacing in dimensionless M units.
        delta_t : float, optional
            *Deprecated.* Use ``delta_t_seconds`` or ``delta_t_Msun`` instead.
        lal_convention : bool, optional
            If True, return h₊ − i h× (LAL convention).  Default returns
            h₊ + i h× (imaginary part = +h×).

        Returns
        -------
        pycbc.types.TimeSeries (complex128)
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
                    stacklevel=2,
                )
            else:
                delta_t = _modal_dt(self.time)
            if delta_t > 1.0 / 128:
                dt_dimless = delta_t
            else:
                dt_dimless = delta_t / m_secs
        new_time = np.arange(min(self.time), max(self.time), dt_dimless)

        angles = self.get_angles(
            inclination=inclination,
            coa_phase=coa_phase,
            f_ref=f_ref,
            t_ref=t_ref,
            tol=tol,
        )
        h = interpolate_in_amp_phase(
            self.evaluate([angles["theta"], angles["psi"], angles["alpha"]]),
            new_time,
            k=k,
            kind=kind,
        ) * utils.amp_to_physical(total_mass, distance)

        h.time *= m_secs

        if lal_convention:
            return self.to_pycbc(h)
        else:
            return self.to_pycbc(np.conjugate(h))

    def get_angles(self, inclination, coa_phase, f_ref=None, t_ref=None, tol=1e-6):
        """Get the inclination, azimuthal and polarization angles
        of the observer in the NR source frame.

        Parameters
        ----------
        inclination : float
            Inclination angle in the LAL source frame.
        coa_phase : float
            Coalescence phase.
        f_ref, t_ref : float, optional
            Reference frequency and time.
        tol : float, optional
            Tolerance for rotation angle computation (1e-6).

        Returns
        -------
        dict
            Angles dict with keys ``theta``, ``psi``, ``alpha``, and
            optionally ``t_ref``, ``f_ref``.
        """
        obs_phi_ref = self.get_obs_phi_ref_from_obs_coa_phase(
            coa_phase=coa_phase, t_ref=t_ref, f_ref=f_ref
        )
        with h5py.File(self.filepath) as h5_file:
            angles = get_nr_to_lal_rotation_angles(
                h5_file=h5_file,
                sim_metadata=self.sim_metadata,
                inclination=inclination,
                phi_ref=obs_phi_ref,
                f_ref=f_ref,
                t_ref=t_ref,
                tol=tol,
            )
        return angles

    def to_pycbc(self, input_array=None, delta_t=None, epoch=None):
        if input_array is None:
            input_array = self
        if epoch is None:
            epoch = input_array.time[0]
        if delta_t is None:
            delta_t = _modal_dt(input_array.time)
        return TimeSeries(
            np.array(input_array),
            delta_t=delta_t,
            dtype=self.ndarray.dtype,
            epoch=epoch,
            copy=True,
        )

    def get_nr_coa_phase(self):
        """Get the NR coalescence orbital phase from the (2,2) mode."""
        phase_22 = self._get_phase(2, 2)
        waveform_22 = (
            self.get_mode_data(2, 2)[:, 1] + 1j * self.get_mode_data(2, 2)[:, 2]
        )
        maxloc = np.argmax(np.absolute(waveform_22))
        return phase_22[maxloc] / 2

    def get_obs_phi_ref_from_obs_coa_phase(self, coa_phase, t_ref=None, f_ref=None):
        """Get the observer reference phase given the observer coalescence phase."""
        nr_coa_phase = self.get_nr_coa_phase()
        nr_orb_phase_ts = self._get_phase(2, 2) / 2
        avail_t_ref = self.t_ref_nr
        from scipy.interpolate import interp1d

        nr_phi_ref = interp1d(self.time, nr_orb_phase_ts, kind="cubic")(avail_t_ref)
        delta_phi_ref = coa_phase - nr_coa_phase
        return nr_phi_ref + delta_phi_ref

    def to_lal(self):
        raise NotImplementedError()

    def to_astropy(self):
        return self.to_pycbc().to_astropy()

    def _get_phase(self, ell=2, emm=2):
        """Get the phasing of a particular waveform mode."""
        wfm_array = self.get_mode_data(ell, emm)
        waveform_lm = wfm_array[:, 1] + 1j * wfm_array[:, 2]
        return np.unwrap(np.angle(waveform_lm))

    def _compute_reference_time(self):
        """Obtain the reference time from the simulation data."""
        with h5py.File(self.filepath) as h5_file:
            interp, avail_t_ref = check_interp_req(
                h5_file, self.sim_metadata, ref_time=None
            )

        if avail_t_ref is None:
            ref_omega = None
            try:
                ref_omega = get_ref_vals(self.sim_metadata, req_attrs=["Omega"])[
                    "Omega"
                ]
            except Exception as excep:
                logger.info(
                    "Reference orbital phase not found in simulation metadata."
                    "Proceeding to retrieve from the h5 file..",
                    excep,
                )
                with h5py.File(self.filepath) as h5_file:
                    ref_omega = get_ref_vals(h5_file, req_attrs=["Omega"])["Omega"]
            if ref_omega is None:
                raise KeyError("Could not compute reference omega!")

            nr_orb_phase_ts = self._get_phase(2, 2) / 2

            from waveformtools.differentiate import derivative

            nr_omega_ts = derivative(self.time, nr_orb_phase_ts, method="FD", degree=2)
            ref_loc = np.argmin(np.absolute(nr_omega_ts - ref_omega))
            avail_t_ref = self.time[ref_loc]

        self._t_ref_nr = avail_t_ref
        return avail_t_ref

    @property
    def t_ref_nr(self):
        """Fetch the reference time of the simulation."""
        if not isinstance(self._t_ref_nr, float):
            logger.info("Computing reference time..")
            self._compute_reference_time()
        return self._t_ref_nr

    @property
    def peak_time_22(self):
        """Dimensionless time of the peak amplitude of the (2,2) mode."""
        if self._peak_time_22 is not None:
            return self._peak_time_22

        try:
            mode22_idx = self.index(2, 2)
        except ValueError:
            self._peak_time_22 = 0.0
            return self._peak_time_22

        mode22_data = np.array(self.data[:, mode22_idx], dtype=complex)
        amp22 = np.abs(mode22_data)
        self._peak_time_22 = float(np.array(self.time)[np.argmax(amp22)])
        return self._peak_time_22

    def rotated(self, R):
        """Rotate the waveform modes.

        Parameters
        ----------
        R : quaternionic.array
            Unit quaternion representing the rotation.

        Returns
        -------
        WaveformModes
        """
        rotated_self = self.copy()
        wigner = spherical.Wigner(self.ell_max)
        rotated_data = np.zeros_like(self.data)

        for ell in range(self.ell_min, self.ell_max + 1):
            if ell not in self.ells:
                continue
            l_modes_indices = np.where(self.LM[:, 0] == ell)[0]
            if len(l_modes_indices) == 0:
                continue
            l_modes = self.data[:, l_modes_indices]
            D = wigner.D(R, ell)
            rotated_data[:, l_modes_indices] = l_modes @ D

        rotated_self.data = rotated_data
        rotated_self.frame = R * self.frame
        return rotated_self

    def match_single_mode(
        self,
        other,
        ell,
        em,
        psd,
        f_lower,
        delta_t=1.0 / 4096,
        f_upper=None,
    ):
        """Compute the noise-weighted match for a single spherical harmonic mode.

        Parameters
        ----------
        other : WaveformModes or dict
            The second waveform.
        ell, em : int
            Spherical harmonic indices.
        psd : pycbc.types.FrequencySeries
            One-sided noise PSD.
        f_lower : float
            Orbital reference frequency in Hz.
        delta_t : float, optional
            Sample spacing in physical seconds (default 1/4096).
        f_upper : float, optional
            Upper frequency cutoff in Hz.

        Returns
        -------
        float
            Match value in [0, 1].
        """
        from pycbc.filter import match as pycbc_match

        h1 = self.get_mode(ell, em, to_pycbc=True, delta_t_seconds=delta_t).real()

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
        self,
        other,
        psd,
        f_lower,
        f_upper=None,
        delta_t=1.0 / 4096,
        return_rotation=False,
        total_mass=1.0,
        distance=1.0,
    ):
        r"""Calculate the match (noise-weighted overlap) between this waveform
        and another, integrated over all observer directions on the sphere
        (sky-averaged) and maximized over time shift, phase shift, and
        active/passive SO(3) coordinate rotation of the source frame.

        Mathematical Formulation
        ------------------------
        The full multi-mode gravitational-wave strain field $H(t, \theta, \phi) = h_+ - i h_\times$
        as observed at polar angles $(\theta, \phi)$ in the source frame is:
        $$
        H(t, \theta, \phi) = \sum_{\ell=2}^{\infty} \sum_{m=-\ell}^{\ell}
        h_{\ell m}(t) \, {}^{-2}Y_{\ell m}(\theta, \phi)
        $$
        where ${}^{-2}Y_{\ell m}$ are the spin-weight $-2$ spherical harmonics.

        The global overlap between two waveforms $h_1$ and $h_2$, integrated over the entire
        sphere of possible observer directions (sky locations), is defined as:
        $$
        \mathcal{O}_{\text{sphere}}(h_1, h_2) =
        \frac{\int_{S^2} \langle h_1(t, \Omega) \mid h_2(t, \Omega) \rangle_t \, d\Omega}{
        \sqrt{\left[ \int_{S^2} \langle h_1(t, \Omega) \mid h_1(t, \Omega) \rangle_t \,
        d\Omega \right] \left[ \int_{S^2} \langle h_2(t, \Omega) \mid h_2(t, \Omega) \rangle_t \,
        d\Omega \right]}}
        $$
        where $\langle \cdot \mid \cdot \rangle_t$ is the standard frequency-domain noise-weighted
        inner product:
        $$
        \langle u \mid v \rangle_t = 4 \, \mathrm{Re}
        \int_{f_{\mathrm{min}}}^{f_{\mathrm{max}}}
        \frac{\tilde{u}(f) \, \tilde{v}^*(f)}{S_n(f)} \, df
        $$

        By utilizing the orthonormality of the spin-weighted spherical harmonics:
        $$
        \int_{S^2} {}^{-2}Y_{\ell m}^*(\Omega) \, {}^{-2}Y_{\ell' m'}(\Omega) \, d\Omega
        = \delta_{\ell \ell'} \, \delta_{m m'}
        $$
        the angular integral decouples, simplifying the sphere-integrated inner product
        into a simple sum over all common modes $(\ell, m)$:
        $$
        \int_{S^2} \langle h_1(t, \Omega) \mid h_2(t, \Omega) \rangle_t \, d\Omega
        = \sum_{\ell, m} \langle h_{1, \ell m} \mid h_{2, \ell m} \rangle_t
        $$

        Coordinate Frame Optimization
        -----------------------------
        Because the two waveforms may be defined in different coordinate systems (source frames)
        and have arbitrary reference times/phases, we align the target waveform $h_2$ to $h_1$
        by active/passive rigid rotation $R \in SO(3)$, time translation $t_c$, and coalescence phase shift $\phi_c$:
        1. **Rotation ($R$)**: Rotates the modes using Wigner D-matrices:
           $$
           h_{2, \ell m}^{\mathrm{rot}}(t) = \sum_{m'=-\ell}^{\ell} h_{2, \ell m'}(t) \, D^{\ell}_{m' m}(R)
           $$

        2. **Time Shift ($t_c$)**: Shifts time via $t \to t - t_c$,
           implemented efficiently as a linear phase in the frequency domain.
        3. **Phase Shift ($\phi_c$)**: Twist around the rotated $z$-axis via:
           $$
           h_{2, \ell m}^{\mathrm{rot, shifted}}(t) \to e^{-i m \phi_c} \, h_{2, \ell m}^{\mathrm{rot}}(t - t_c)
           $$

        The method then returns the maximized match (overlap):

        $$
        \mathcal{O}_{\mathrm{max}} = \max_{t_c, \phi_c, R \in SO(3)} \left[
        \frac{
            \sum_{\ell, m} \langle h_{1, \ell m} \mid
            h_{2, \ell m}^{\mathrm{rot, shifted}}(t_c, \phi_c, R) \rangle_t
        }{
            \sqrt{
                \left( \sum_{\ell, m} \langle h_{1, \ell m} \mid
                h_{1, \ell m} \rangle_t \right)
                \left( \sum_{\ell, m} \langle h_{2, \ell m} \mid
                h_{2, \ell m} \rangle_t \right)
            }
        } \right]
        $$

        The maximization over $t_c$ is performed efficiently using Fast Fourier Transforms (FFTs),
        $\phi_c$ is maximized analytically, and the SO(3) rotation $R$ (parameterized by
        Euler angles $\alpha, \beta, \gamma$) is optimized using the differential evolution algorithm.

        Parameters
        ----------
        other : WaveformModes or dict
            The second waveform to compare against. Can be a `WaveformModes` object or a dict
            of PyCBC TimeSeries modes.
        psd : pycbc.types.FrequencySeries
            One-sided noise power spectral density (PSD).
        f_lower : float
            Lower frequency cutoff in Hz.
        f_upper : float, optional
            Upper frequency cutoff in Hz. If None, the Nyquist frequency of the PSD is used.
        delta_t : float, optional
            Sample spacing in physical seconds (default 1/4096).
        return_rotation : bool, optional
            If True, returns a tuple `(match, R_opt)` containing the maximum match and the
            optimal quaternionic rotation.
        total_mass : float, optional
            Total mass of the binary system in solar masses (default 1.0).
        distance : float, optional
            Luminosity distance to the source in Mpc (default 1.0).

        Returns
        -------
        float or tuple
            If `return_rotation` is False, returns the maximum match value in $[0, 1]$.
            If `return_rotation` is True, returns `(match, R_opt)` where `R_opt` is the
            optimal `quaternionic.array` unit quaternion representing the rotation.
        """
        import numpy as np
        from scipy.optimize import differential_evolution
        from scipy.fft import fft, ifft
        import quaternionic
        import spherical

        # Compute overlapping frequency range
        df = psd.delta_f
        low_idx = int(f_lower / df) if f_lower else 0
        high_idx = int(np.ceil(f_upper / df)) if f_upper else len(psd)

        if isinstance(other, dict):
            other_LM = list(other.keys())
        else:
            other_LM = list(map(tuple, other.LM))

        common_modes = set(map(tuple, self.LM)) & set(other_LM)
        if not common_modes:
            return (0.0, None) if return_rotation else 0.0

        h1_ts_dict = {}
        h2_ts_dict = {}

        # Load modes and align lengths
        for ell, m in common_modes:
            h1_ts_dict[(ell, m)] = self.get_mode(
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

        wigner = spherical.Wigner(self.ell_max)
        ells_in_common = set(ell for ell, m in common_modes)

        def objective_function(x):
            phi_c, alpha, beta, gamma = x
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
                    term = (
                        h1_f_dict[(ell, m)] * np.conj(h2_rot_matrix[:, j])
                    ) / psd_full
                    term *= np.exp(1j * m * phi_c)
                    I_f_full += term

            _q = ifft(I_f_full)
            max_inner_prod = df * N_pad * np.max(np.real(_q))

            overlap = max_inner_prod / np.sqrt(total_norm1_sq * total_norm2_sq)
            if np.isnan(overlap):
                return 1.0

            return 1.0 - overlap

        bounds = [(0, 2 * np.pi), (0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi)]

        identity_mismatch = objective_function([0.0, 0.0, 0.0, 0.0])
        logger.info(
            f"      [DEBUG] Sphere-averaged match at Identity Rotation: {1.0 - identity_mismatch:.6f}"
        )

        result = differential_evolution(
            objective_function,
            bounds,
            popsize=10,
            maxiter=50,
            tol=1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
        )
        match = 1.0 - result.fun

        if return_rotation:
            R_opt = quaternionic.array.from_euler_angles(
                result.x[1], result.x[2], result.x[3]
            )
            return match, R_opt
        return match

    def match_sphere_averaged_bms_maximized(
        self,
        other,
        psd,
        f_lower,
        f_upper=None,
        j_max=1,
    ):
        r"""Calculate the match maximized over BMS supertranslations in addition to
        standard time shift, phase shift, and SO(3) rotation.

        BMS Supertranslation Mathematical Formulation
        ---------------------------------------------
        At null infinity $\mathcal{I}^+$, the asymptotic symmetry group of General Relativity
        is the infinite-dimensional Bondi-Metzner-Sachs (BMS) group. This group is the
        semi-direct product of the Lorentz group and the abelian group of **supertranslations**,
        which correspond to direction-dependent shifts in the retarded time coordinate $u$:
        $$
        u' = u - \alpha(\theta, \phi)
        $$
        where the supertranslation field $\alpha(\theta, \phi)$ is an arbitrary smooth real
        function on the sphere, decomposed into scalar spherical harmonics $Y_{j k}$:
        $$
        \alpha(\theta, \phi) = \sum_{j=0}^{j_{\mathrm{max}}} \sum_{k=-j}^{j} \alpha_{j k} \, Y_{j k}(\theta, \phi)
        $$
        Here, $j=0$ corresponds to a global time translation ($t_c$), $j=1$ corresponds to
        spatial translations (origin shifts), and $j \ge 2$ modes correspond to proper
        supertranslations.

        Under a small supertranslation, the strain waveform modes $h_{\ell m}(u)$ undergo
        first-order mode mixing:
        $$
        h'_{\ell m}(u) \approx h_{\ell m}(u) -
        \sum_{j=0}^{j_{\mathrm{max}}} \sum_{k=-j}^{j} \sum_{p, q}
        \alpha_{j k} \, \mathcal{G}^{\ell m}_{j k, p q} \, \dot{h}_{p q}(u)
        $$
        where $\dot{h}_{p q}(u) = \partial h_{p q} / \partial u$, and $\mathcal{G}^{\ell m}_{j k, p q}$
        are the spin-weighted Gaunt coefficients (integrals of products of three spherical harmonics):
        $$
        \mathcal{G}^{\ell m}_{j k, p q} = \int_{S^2}
        {}^{-2}Y_{\ell m}^*(\Omega) \, Y_{j k}(\Omega) \,
        {}^{-2}Y_{p q}(\Omega) \, d\Omega
        $$

        This method optimizes both the rigid rotation $R \in SO(3)$, time translation $t_c$,
        phase shift $\phi_c$, and the supertranslation coefficients $\alpha_{j k}$ for $j \ge 1$
        up to `j_max` using the Nelder-Mead downhill simplex algorithm to minimize the mismatch
        (maximize the overlap).

        Parameters
        ----------
        other : WaveformModes
            The second waveform to compare against.
        psd : pycbc.types.FrequencySeries
            One-sided noise power spectral density (PSD).
        f_lower : float
            Lower frequency cutoff in Hz.
        f_upper : float, optional
            Upper frequency cutoff in Hz. If None, the Nyquist frequency of the PSD is used.
        j_max : int, optional
            Maximum spherical-harmonic order of the supertranslation field to optimize
            (default 1, which corresponds to time translation + spatial translation).

        Returns
        -------
        float
            Maximum match value in $[0, 1]$.
        """
        from scipy.optimize import minimize

        try:
            import scri
        except ImportError as e:
            raise ImportError(
                "The 'scri' package is required for BMS supertranslation optimization. "
                "Install it with: pip install scri"
            ) from e

        alpha_jk_indices = [
            (j, k) for j in range(1, j_max + 1) for k in range(-j, j + 1)
        ]

        max_len = 0
        for ell, m in self.LM:
            max_len = max(
                max_len,
                len(self.get_mode(ell, m, to_pycbc=True, delta_t_seconds=1 / 4096)),
            )

        ref_mode_ts = self.get_mode(2, 2, to_pycbc=True, delta_t_seconds=1 / 4096)
        ref_mode_ts.resize(max_len)
        ref_fs = ref_mode_ts.to_frequencyseries()
        freqs = ref_fs.sample_frequencies
        delta_f = ref_fs.delta_f

        self_modes_tilde = {}
        self_modes_dot_tilde = {}
        for ell, m in self.LM:
            h_ts = self.get_mode(ell, m, to_pycbc=True, delta_t_seconds=1 / 4096)
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

            common_modes = set(map(tuple, self.LM)) & set(map(tuple, other_rot.LM))

            self_modes_tilde_st = {}
            for ell, m in common_modes:
                h1_tilde = self_modes_tilde[(ell, m)]
                st_correction = np.zeros_like(h1_tilde.data, dtype=complex)

                for (j, k), alpha_jk in alpha_jk_coeffs.items():
                    for p, q in self.LM:
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

            overlap = np.abs(total_inner_prod) / np.sqrt(
                total_norm1_sq * total_norm2_sq
            )
            return 1.0 - overlap

        x0 = [0.0] * (5 + len(alpha_jk_indices))
        result = minimize(objective_function, x0, method="Nelder-Mead")
        return 1.0 - result.fun
