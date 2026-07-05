"""Utility constants, cache-path definitions, and download helpers.

Module-level constants
----------------------
nrcatalog_cache_dir : pathlib.Path
    Root cache directory; defaults to ``~/.cache/`` but can be overridden
    by setting the ``NR_CATALOG_CACHE`` environment variable.

rit_catalog_info : dict
    Cache paths, base URLs, filename format strings, and parameter ranges
    for the RIT catalog.

maya_catalog_info : dict
    Cache paths and base URLs for the MAYA/GT catalog.

sxs_catalog_info : dict
    Cache paths and base URLs for the SXS catalog (supplemental; the
    ``sxs`` package manages its own cache internally).

Public functions
----------------
url_exists(link, num_retries, verbosity)
    HEAD-request check with exponential-backoff retries.

download_file(url, path, progress, if_newer, num_retries, verbosity)
    Download a URL to a local path; tries the ``sxs`` downloader first and
    falls back to ``requests``.

call_with_timeout(myfunc, args, kwargs, timeout)
    Run a callable in a subprocess with a hard wall-clock timeout.

time_to_physical(M)
    Convert dimensionless NR time units to seconds.

amp_to_physical(M, D)
    Scale dimensionless NR strain amplitude to SI units.

amplitude_phase_frequency_from_complex_mode(hlm)
    Compute instantaneous amplitude, phase, and frequency from a complex
    PyCBC TimeSeries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import functools
import os
import pathlib
import shutil
import time
import lal
import requests

import sxs

if os.getenv("NR_CATALOG_CACHE"):
    nrcatalog_cache_dir = (
        pathlib.Path(os.getenv("NR_CATALOG_CACHE")).expanduser().resolve()
    )
else:
    nrcatalog_cache_dir = pathlib.Path("~/.cache/").expanduser().resolve()
nr_group_tags = {}
nr_group_tags["SXS"] = "SXS"
nr_group_tags["RIT"] = "RIT"
nr_group_tags["MAYA"] = "MAYA"
nr_group_tags["UNKNOWN"] = "UNKNOWN"

rit_catalog_info = {}
rit_catalog_info["cache_dir"] = nrcatalog_cache_dir / "RIT"
rit_catalog_info["metadata_dir"] = rit_catalog_info["cache_dir"] / "metadata/"
rit_catalog_info["data_dir"] = rit_catalog_info["cache_dir"] / "data/"
rit_catalog_info["url"] = "https://ccrgpages.rit.edu/~RITCatalog/"
rit_catalog_info["metadata_url"] = rit_catalog_info["url"] + "/Metadata/"
rit_catalog_info["data_url"] = rit_catalog_info["url"] + "/Data/"
rit_catalog_info["possible_resolutions"] = [
    100,
    120,
    88,
    84,
    118,
    130,
    140,
    144,
    160,
    200,
]
rit_catalog_info["metadata_file_fmts"] = [
    "RIT:BBH:{:04d}-n{:03d}-id{:d}_Metadata.txt",
    "RIT:eBBH:{:04d}-n{:03d}-ecc_Metadata.txt",
]
rit_catalog_info["waveform_file_fmts"] = [
    "ExtrapStrain_RIT-BBH-{:04d}-n{:03d}.h5",
    "ExtrapStrain_RIT-eBBH-{:04d}-n{:03d}.h5",
]
rit_catalog_info["psi4_file_fmts"] = [
    "ExtrapPsi4_RIT-BBH-{:04d}-n{:03d}-id{:d}.tar.gz",
    "ExtrapPsi4_RIT-eBBH-{:04d}-n{:03d}-ecc.tar.gz",
]
rit_catalog_info["max_id_val"] = 6

maya_catalog_info = {
    "cache_dir": nrcatalog_cache_dir / "MAYA",
    "data_url": "https://cgpstorage.ph.utexas.edu/",
    "metadata_url": "https://cgpstorage.ph.utexas.edu/MAYAmetadata.pkl",
}
maya_catalog_info["data_dir"] = maya_catalog_info["cache_dir"] / "data/"
maya_catalog_info["metadata_dir"] = maya_catalog_info["cache_dir"] / "metadata"

sxs_catalog_info = {
    "cache_dir": nrcatalog_cache_dir / "SXS",
    "data_url": "https://www.black-holes.org/waveforms/",
    "metadata_url": "https://www.black-holes.org/waveforms/metadata.json",
}
sxs_catalog_info["data_dir"] = sxs_catalog_info["cache_dir"] / "data/"
sxs_catalog_info["metadata_dir"] = sxs_catalog_info["cache_dir"] / "metadata"


def url_exists(link: str, num_retries: int = 5, verbosity: int = 0) -> bool:
    """Check if a given URL exists on the web.

    Retries up to ``num_retries`` times on network errors, with exponential
    backoff (``2**attempt`` seconds, capped at 30 s).  A non-OK HTTP status
    is returned immediately as ``False`` without retrying (the URL exists but
    is not accessible / not found — no point retrying).

    Args:
        link (str): Complete web URL.
        num_retries (int): Maximum number of attempts. Defaults to 5.
        verbosity (int): Print retry progress when > 0. Defaults to 0.

    Returns:
        bool: True if the URL returned HTTP 200, False otherwise.
    """
    requests.packages.urllib3.disable_warnings()
    for attempt in range(num_retries):
        try:
            response = requests.head(link, verify=False, timeout=10)
            return response.status_code == requests.codes.ok
        except Exception as exc:
            if isinstance(exc, requests.exceptions.Timeout):
                import warnings

                warnings.warn(f"Request to {link} timed out after 10 seconds.")
            if attempt < num_retries - 1:
                delay = min(2**attempt, 30)
                if verbosity > 0:
                    logger.info(
                        f"url_exists: attempt {attempt + 1}/{num_retries} failed"
                        f" for {link}; retrying in {delay}s"
                    )
                time.sleep(delay)
    if verbosity > 0:
        logger.info(f"url_exists: all {num_retries} attempts failed for {link}")
    return False


def download_file(
    url: str,
    path: str | pathlib.Path,
    progress: bool = False,
    if_newer: bool = True,
    num_retries: int = 5,
    verbosity: int = 0,
) -> pathlib.Path:
    """
    Download a file from the given URL to the specified local path.

    This function attempts to download the file at `url` and save it to `path`.
    It first tries to use the `sxs.utilities.downloads.download_file` utility (if available).
    If that fails, it falls back to using the Python `requests` package, with SSL
    verification disabled and up to ``num_retries`` attempts with exponential
    backoff (``2**attempt`` seconds, capped at 30 s).

    Args:
        url (str): The URL to download the file from.
        path (str or pathlib.Path): The destination path where the file should be saved.
        progress (bool, optional): Whether to show a progress bar if supported.
        if_newer (bool, optional): Only download if the remote file is newer than the local file.
        num_retries (int): Maximum number of fallback attempts. Defaults to 5.
        verbosity (int): Print retry progress when > 0. Defaults to 0.

    Returns:
        The path (as a pathlib.Path object) to the downloaded file.

    Raises:
        ConnectionError: If the file could not be fetched after all retry attempts.
        RuntimeError: If the server returned a non-200 status.
    """
    if url_exists(url, num_retries=num_retries, verbosity=verbosity):
        try:
            return sxs.utilities.downloads.download_file(
                url, path, progress=progress, if_newer=if_newer
            )
        except Exception:
            requests.packages.urllib3.disable_warnings()
            last_exc = None
            r = None
            for attempt in range(num_retries):
                try:
                    r = requests.get(
                        url, verify=False, stream=True, allow_redirects=True, timeout=10
                    )
                    break
                except Exception as exc:
                    if isinstance(exc, requests.exceptions.Timeout):
                        import warnings

                        warnings.warn(f"Request to {url} timed out after 10 seconds.")
                    last_exc = exc
                    if attempt < num_retries - 1:
                        delay = min(2**attempt, 30)
                        if verbosity > 0:
                            logger.info(
                                f"download_file: attempt {attempt + 1}/{num_retries}"
                                f" failed for {url}; retrying in {delay}s"
                            )
                        time.sleep(delay)
            else:
                raise ConnectionError(
                    f"Failed to download '{url}' after {num_retries} attempts"
                ) from last_exc
            if r.status_code != 200:
                logger.info(f"An error occurred when trying to access <{url}>.")
                try:
                    logger.info(r.json())
                except Exception:
                    pass
                r.raise_for_status()
                raise RuntimeError()  # Will only happen if the response was not strictly an error
            r.raw.read = functools.partial(r.raw.read, decode_content=True)
            path = pathlib.Path(path).expanduser().resolve()
            with path.open("wb") as f:
                shutil.copyfileobj(r.raw, f)
    return path


def call_with_timeout(
    myfunc: object, args: tuple = (), _kwargs: dict = {}, timeout: float = 5
) -> object:
    """
    Call a function with a time limit in a separate process.

    Executes the provided function `myfunc` with given positional (`args`) and keyword
    arguments (`kwargs`) in a separate process. If the function does not complete
    within `timeout` seconds, the process is terminated and an exception is raised.
    If the function completes within the timeout, its result is returned.

    Args:
        myfunc (callable): The function to execute.
        args (tuple, optional): Positional arguments to pass to `myfunc`. Defaults to ().
        _kwargs (dict, optional): Reserved; keyword arguments are not currently
                                  forwarded to the subprocess. Defaults to {}.
        timeout (int or float, optional): Maximum allowed execution time in seconds. Defaults to 5.

    Returns:
        The result of `myfunc(*args, **kwargs)` if completed within the timeout.

    Raises:
        Exception: If the function does not complete within the specified timeout.
    """

    from multiprocessing import Process, Queue

    def funcwrapper(p, *args, **kwargs) -> None:
        """
        This thin wrapper calls the user-provided function, and puts
        its result into the multiprocessing `Queue`  so that it can be
        obtained via `Queue().get()`.
        """
        res = myfunc(*args, **kwargs)
        p.put(res)

    queue = Queue()
    task = Process(target=funcwrapper, args=(queue, *args))
    task.start()
    task.join(timeout=timeout)
    task.terminate()
    try:
        result = queue.get(timeout=0)
        return result
    except Exception:
        raise Exception("Timeout")


def time_to_physical(M: float) -> float:
    """
    Factor to convert time from dimensionless units to SI units

    parameters
    ----------
    M: mass of system in the units of solar mass

    Returns
    -------
    converting factor
    """

    return M * lal.MTSUN_SI


def amp_to_physical(M: float, D: float) -> float:
    """
    Factor to rescale strain to mass M and distance D convert from
    dimensionless units to SI units

    parameters
    ----------
    M: mass of the system in units of solar mass
    D: Luminosity distance in units of megaparsecs

    Returns
    -------
    Scaling factor
    """

    return lal.G_SI * M * lal.MSUN_SI / (lal.C_SI**2 * D * 1e6 * lal.PC_SI)


def amplitude_phase_frequency_from_complex_mode(hlm: object) -> tuple:
    """
    Compute amplitude, phase, and instantaneous frequency from a complex mode time series.

    Parameters
    ----------
    hlm : tuple of (real, imag) pycbc.types.TimeSeries, or a single complex pycbc.types.TimeSeries
        Either a tuple of real and imaginary parts of a mode (as PyCBC TimeSeries with matching sample_times),
        or a single complex-valued PyCBC TimeSeries.

    Returns
    -------
    amp : pycbc.types.TimeSeries
        The instantaneous amplitude as a function of time.
    phase : pycbc.types.TimeSeries
        The instantaneous phase as a function of time.
    freq : pycbc.types.TimeSeries
        The instantaneous frequency (cycles per unit time) as a function of time.
    """
    import numpy as np
    from pycbc.types import TimeSeries

    # Check if hlm is a tuple with PyCBC TimeSeries (real, imag)
    if isinstance(hlm, tuple) and len(hlm) == 2:
        re, im = hlm
        h_complex = re.numpy() + 1j * im.numpy()
        # Assume re/im have sample_times attribute if PyCBC TimeSeries
        if hasattr(re, "sample_times"):
            re.sample_times.numpy()
        else:
            raise AttributeError(
                "Real/Imag PyCBC TimeSeries objects must have sample_times."
            )
        delta_t = re.delta_t
    else:
        # Accept complex-valued PyCBC TimeSeries input
        if isinstance(hlm, TimeSeries) and np.iscomplexobj(hlm):
            h_complex = hlm.numpy()
            delta_t = hlm.delta_t
            re = hlm  # use for .start_time and .delta_t
        else:
            raise ValueError(
                "Input must be a tuple of PyCBC TimeSeries (re, im), or a complex-valued PyCBC TimeSeries."
            )

    # Compute amplitude
    amp_arr = np.abs(h_complex)

    amp = TimeSeries(amp_arr, delta_t=delta_t, epoch=re.start_time)

    # Compute phase
    phase_arr = np.unwrap(np.angle(h_complex))
    phase = TimeSeries(phase_arr, delta_t=delta_t, epoch=re.start_time)

    # Compute dphase/dt (frequency) as a TimeSeries (using uniform time steps)
    # (careful: np.gradient(y, dx) is d(y)/d(x) for uniform x spacing dx)
    dphase_dt_arr = np.gradient(phase_arr, re.delta_t)
    freq = TimeSeries(dphase_dt_arr / 2 / np.pi, delta_t=delta_t, epoch=re.start_time)
    return amp, phase, freq
