"""MAYA/GT catalog interface.

Wraps the Georgia Tech MAYA catalog of numerical-relativity BBH waveforms.
Metadata is distributed as a pickled Pandas DataFrame
(``MAYAmetadata.pkl``) downloaded from the UT Austin CGP storage server
and cached locally as a bzip2-compressed ZIP archive.  Waveform data files
are individual HDF5 files downloaded on demand.

Key design decisions
--------------------
* **Singleton pattern** – ``load()`` stores its result in
  ``_maya_catalog_singleton`` so that repeated calls reuse the parsed
  metadata.  Pass ``download=True`` or call ``reload()`` to force a fresh
  download.

* **LVCNR conversion** – ``download_waveform_data()`` can optionally
  convert MAYA-format HDF5 files to LVCNR catalog format via the
  ``mayawaves`` package before caching.

* **PSI4 not available** – the MAYA catalog only distributes strain
  waveforms; all ``psi4_*`` methods raise ``NotImplementedError``.

Public classes
--------------
MayaCatalog
    Registered under the tag ``"MAYA"`` in the catalog plugin registry.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import collections
import functools
import os
import zipfile
import pandas as pd
from nrcats import catalog, utils
from nrcats.registry import register_catalog

# Module-level singleton — same stale-result fix as RITCatalog.
_maya_catalog_singleton = None


@register_catalog("MAYA")
class MayaCatalog(catalog.CatalogBase):
    """Catalog interface for the MAYA / Georgia-Tech NR waveform collection.

    Downloads a pickle-format metadata file (``MAYAmetadata.pkl``) from
    the UT Austin CGP storage and uses the ``mayawaves`` package to load
    individual waveform HDF5 files.  Key design points:

    - Metadata is cached as ``~/.cache/MAYA/data/catalog.zip`` on first
      load.  Subsequent calls read from cache unless ``download=True``.
    - Waveform files are loaded via ``mayawaves.Coalescence`` and then
      wrapped in a ``WaveformModes`` object.
    - Psi4 is not available for this catalog; the corresponding methods
      raise ``NotImplementedError``.
    - A module-level singleton prevents redundant catalog loads when
      ``load()`` is called multiple times in the same process.

    Example:
        >>> import nrcats as nrcat
        >>> cat = nrcat.MayaCatalog.load()
        >>> wfm = cat.get("GT0001")
    """

    CATALOG_TYPE = "MAYA"

    def __init__(
        self,
        catalog: dict | None = None,
        use_cache: bool = True,
        verbosity: int = 0,
        **kwargs,
    ) -> None:
        """This class wraps the catalog infrastructure provided by the
        groups performing NR simulations using the MAYA code, and provides
        an interface that is derived from the `sxs.catalog` class.

        Args:
            catalog (MayaCatalog or CatalogBase, optional): If a catalog object
                already exist, pass it here to transfer its ownership to the new
                object. Defaults to None.
            use_cache (bool, optional): Whether to download all metadata and
                data on the fly or use cached versionis. Defaults to True.
            verbosity (int, optional): Verbose output levels [0=quiet,
                10=most verbose]. Defaults to 0.
        """
        if catalog is not None:
            super().__init__(catalog["simulations"])
        else:
            obj = type(self).load(verbosity=verbosity, **kwargs)
            super().__init__(obj._simulations)
        self._verbosity = verbosity

        # Other info
        self.num_of_sims = 0
        self.cache_dir = utils.maya_catalog_info["cache_dir"]
        self.use_cache = use_cache

        self.metadata = pd.DataFrame.from_dict(self._simulations)
        self.metadata_url = utils.maya_catalog_info["metadata_url"]
        self.metadata_dir = utils.maya_catalog_info["metadata_dir"]

        self.waveform_data = {}
        self.waveform_data_url = utils.maya_catalog_info["data_url"]
        self.waveform_data_dir = utils.maya_catalog_info["data_dir"]

        self._add_paths_to_metadata()

        internal_dirs = [self.cache_dir, self.metadata_dir, self.waveform_data_dir]
        for d in internal_dirs:
            d.mkdir(parents=True, exist_ok=True)

    def clear_cache(self) -> None:
        """Remove the cached catalog ZIP file so the next ``load()`` re-downloads it.

        The file removed is ``~/.cache/MAYA/catalog.zip`` (or the path
        configured via ``NR_CATALOG_CACHE``).  The module-level singleton is
        *not* cleared by this method; call ``MayaCatalog.reload()`` to force a
        full re-download and replace the singleton.
        """
        cache_path = utils.maya_catalog_info["cache_dir"] / "catalog.zip"
        if cache_path.exists():
            os.remove(cache_path)

    @classmethod
    def load(
        cls,
        download: bool | None = None,
        verbosity: int = 0,
        show_progress: bool = True,
    ) -> MayaCatalog:
        """Load the MAYA catalog.

        Downloads the pickled metadata from the UT Austin CGP storage server,
        compresses it to a local bzip2 ZIP cache, and parses it into a
        ``MayaCatalog`` singleton.  Subsequent calls return the singleton
        without re-parsing, unless ``download=True``.

        Args:
            download (bool or None): If ``False``, only the local cache is
                used and an error is raised if it is absent.  If ``True``, a
                fresh download is always attempted.  If ``None`` (default), a
                download is attempted and the cache is used as a fallback.
            verbosity (int): Verbosity level (0 = quiet). Defaults to 0.
            show_progress (bool): Show a download progress bar. Defaults to True.

        Returns:
            MayaCatalog: The loaded (possibly cached) catalog instance.

        Raises:
            RuntimeError: If ``download=True`` and the download fails.
            ValueError: If the catalog cache file is missing or corrupt.
        """
        global _maya_catalog_singleton
        if _maya_catalog_singleton is not None and download is not True:
            return _maya_catalog_singleton

        # Create cache dir if it does not exist
        utils.maya_catalog_info["cache_dir"].mkdir(parents=True, exist_ok=True)

        metadata_url = utils.maya_catalog_info["metadata_url"]
        cache_path = utils.maya_catalog_info["cache_dir"] / "catalog.zip"
        if cache_path.exists():
            if_newer = cache_path
        else:
            if_newer = False

        if download or download is None:
            # 1. Download the full pickle file (zipped in flight, but auto-decompressed on arrival)
            # 2. Zip to a temporary file (using bzip2, which is better than the in-flight compression)
            # 3. Replace the original catalog.zip with the temporary zip file
            # 4. Remove the full pickle file
            # 5. Make sure the temporary zip file is gone too
            temp_pkl = cache_path.with_suffix(".temp.pkl")
            temp_zip = cache_path.with_suffix(".temp.zip")
            try:
                try:
                    utils.download_file(
                        metadata_url,
                        temp_pkl,
                        progress=show_progress,
                        if_newer=if_newer,
                    )
                except Exception as e:
                    if download:
                        raise RuntimeError(
                            f"Failed to download '{metadata_url}'; If you don't have a "
                            "network connection, try setting `download=False`"
                        ) from e
                    download_failed = e  # We'll try the cache
                else:
                    download_failed = False
                    if temp_pkl.exists():
                        with zipfile.ZipFile(
                            temp_zip, "w", compression=zipfile.ZIP_BZIP2
                        ) as catalog_zip:
                            catalog_zip.write(temp_pkl, arcname="catalog.pkl")
                        temp_zip.replace(cache_path)
            finally:
                # The `missing_ok` argument to `unlink` would be much nicer, but was added in python 3.8
                try:
                    temp_pkl.unlink()
                except FileNotFoundError:
                    pass
                try:
                    temp_zip.unlink()
                except FileNotFoundError:
                    pass

        if not cache_path.exists():
            if download_failed:
                raise ValueError(
                    f"Catalog not found in '{cache_path}' and download failed"
                ) from download_failed
            elif (
                download is False
            ):  # Test if it literally *is* False, rather than just casts to False
                raise ValueError(
                    f"The catalog was not found in '{cache_path}', and downloading was turned off"
                )
            else:
                raise ValueError(
                    f"Catalog not found in '{cache_path}' for unknown reasons"
                )

        try:
            with zipfile.ZipFile(cache_path, "r") as catalog_zip:
                try:
                    with catalog_zip.open("catalog.pkl") as catalog_pickle:
                        try:
                            catalog_df = pd.read_pickle(catalog_pickle)
                        except Exception as e:
                            raise ValueError(
                                f"Failed to parse 'catalog.json' in '{cache_path}'"
                            ) from e
                except Exception as e:
                    raise ValueError(
                        f"Failed to open 'catalog.pkl' in '{cache_path}'"
                    ) from e
        except Exception as e:
            raise ValueError(f"Failed to open '{cache_path}' as a ZIP file") from e

        # Fill in the catalog object
        catalog_dict = {}
        catalog_dict["GTID"] = [s.strip() for s in list(catalog_df.index)]

        for col_name in catalog_df.columns:
            column = list(catalog_df[col_name])
            if "name" in col_name:
                catalog_dict["GT_Tag"] = [s.strip() for s in column]
            else:
                catalog_dict[col_name.strip()] = [
                    float(s.strip().replace("-", "NAN")) if type(s) is str else float(s)
                    for s in column
                ]
        catalog_df = pd.DataFrame(catalog_dict)
        catalog_obj = {}
        simulations = {}
        for idx, row in catalog_df.iterrows():
            name = row["GTID"]
            metadata_dict = row.to_dict()
            simulations[name] = metadata_dict
        catalog_obj["simulations"] = simulations
        _maya_catalog_singleton = cls(catalog=catalog_obj, verbosity=verbosity)
        return _maya_catalog_singleton

    @classmethod
    def reload(cls, **kwargs) -> MayaCatalog:
        """Force a fresh download and replace the cached singleton.

        Equivalent to ``MayaCatalog.load(download=True, **kwargs)``.
        """
        global _maya_catalog_singleton
        _maya_catalog_singleton = None
        return cls.load(download=True, **kwargs)

    def _add_paths_to_metadata(self):
        """Populate each simulation's metadata dict with path and URL columns.

        Adds the following keys to every simulation's metadata dict if they
        are not already present:

        - ``metadata_link`` – remote URL (same for all simulations; points to
          the catalog-level metadata pickle).
        - ``metadata_location`` – absolute local path for the per-simulation
          metadata text file.
        - ``waveform_data_link`` – remote URL for the waveform HDF5 file.
        - ``waveform_data_location`` – absolute local path for the waveform
          HDF5 file.

        This method is idempotent and safe to call multiple times.
        """
        metadata_dict = self._simulations
        existing_cols = list(metadata_dict[list(metadata_dict.keys())[0]].keys())
        new_cols = [
            "metadata_link",
            "metadata_location",
            "waveform_data_link",
            "waveform_data_location",
        ]

        if any([col not in existing_cols for col in new_cols]):
            for sim_name in metadata_dict:
                if "metadata_location" not in existing_cols:
                    metadata_dict[sim_name][
                        "metadata_location"
                    ] = self.metadata_filepath_from_simname(sim_name)
                if "metadata_link" not in existing_cols:
                    metadata_dict[sim_name]["metadata_link"] = self.metadata_url
                if "waveform_data_link" not in existing_cols:
                    metadata_dict[sim_name]["waveform_data_link"] = (
                        self.waveform_data_url + "/" + f"{sim_name}.h5"
                    )
                if "waveform_data_location" not in existing_cols:
                    metadata_dict[sim_name][
                        "waveform_data_location"
                    ] = self.waveform_filepath_from_simname(sim_name)

    @property
    @functools.lru_cache()
    def simulations_dataframe(self) -> object:
        """All simulations as a Pandas DataFrame indexed by simulation name.

        Columns correspond to the MAYA metadata fields (mass ratio, spins,
        orbital frequency, etc.) plus the path/link columns added by
        ``_add_paths_to_metadata()``.

        Returns:
            pandas.DataFrame: DataFrame with one row per simulation.
        """
        df = pd.DataFrame(self.simulations).transpose()
        df.rename(columns={"GTID": "simulation_name"}, inplace=True)
        return df

    @property
    @functools.lru_cache()
    def files(self) -> dict:
        """Map of waveform filenames to file-info dicts.

        Each value is a dict with keys:
        ``checksum`` (None), ``filename``, ``filesize`` (bytes, 0 if not
        cached), ``download`` (remote URL), and ``truepath`` (canonical
        local filename after deduplication).

        Returns:
            dict[str, dict]: Mapping from bare filename to file-info dict.
        """
        file_infos = {}
        for _, row in self.simulations_dataframe.iterrows():
            waveform_data_location = row["waveform_data_location"]
            path_str = os.path.basename(waveform_data_location)
            if os.path.exists(waveform_data_location):
                file_size = os.path.getsize(waveform_data_location)
            else:
                file_size = 0
            file_info = {
                "checksum": None,
                "filename": os.path.basename(waveform_data_location),
                "filesize": file_size,
                "download": row["waveform_data_link"],
            }
            file_infos[path_str] = file_info

        unique_files = collections.defaultdict(list)
        for k, v in file_infos.items():
            unique_files[f"{v['checksum']}{v['filesize']}"].append(k)

        original_paths = {k: min(v) for k, v in unique_files.items()}

        for v in file_infos.values():
            v["truepath"] = original_paths[f"{v['checksum']}{v['filesize']}"]

        return file_infos

    def metadata_filename_from_simname(self, sim_name: str) -> str:
        """Return the bare filename for the per-simulation metadata file.

        Args:
            sim_name (str): MAYA simulation name, e.g. ``"GT0001"``.

        Returns:
            str: Filename, e.g. ``"GT0001.txt"``.
        """
        return os.path.basename(self.metadata_filepath_from_simname(sim_name))

    def metadata_filepath_from_simname(self, sim_name: str, ext: str = "txt") -> str:
        """Return the absolute local path for the per-simulation metadata file.

        Args:
            sim_name (str): MAYA simulation name, e.g. ``"GT0001"``.
            ext (str): File extension. Defaults to ``"txt"``.

        Returns:
            str: Absolute path under ``~/.cache/MAYA/metadata/``.
        """
        return str(self.metadata_dir / f"{sim_name}.{ext}")

    def metadata_url_from_simname(self, _sim_name: str) -> None:
        """MAYA does not expose per-simulation metadata URLs; returns None.

        The parameter is accepted for interface compatibility with
        ``CatalogABC`` but is not used.

        Returns:
            None
        """
        return

    def waveform_filename_from_simname(self, sim_name: str) -> str:
        """Return the bare HDF5 filename for *sim_name*.

        Args:
            sim_name (str): MAYA simulation name, e.g. ``"GT0001"``.

        Returns:
            str: Filename, e.g. ``"GT0001.h5"``.
        """
        return sim_name + ".h5"

    def waveform_filepath_from_simname(self, sim_name: str) -> str:
        """Return the absolute local path for the waveform HDF5 file.

        Args:
            sim_name (str): MAYA simulation name, e.g. ``"GT0001"``.

        Returns:
            str: POSIX-style absolute path under ``~/.cache/MAYA/data/``.
        """
        file_path = self.waveform_data_dir / self.waveform_filename_from_simname(
            sim_name
        )
        if not os.path.exists(file_path):
            if self._verbosity > 2:
                logger.info(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
        return file_path.as_posix()

    def waveform_url_from_simname(
        self, sim_name: str, maya_format: bool = False
    ) -> str:
        """Return the remote URL for the waveform HDF5 file.

        Args:
            sim_name (str): MAYA simulation name, e.g. ``"GT0001"``.
            maya_format (bool): If True, return the URL for the native MAYA
                format file instead of the LVCNR-format file. Defaults to False.

        Returns:
            str: Full HTTP(S) URL.
        """
        if maya_format:
            format = "maya_format"
        else:
            format = "lvcnr_format"
        return f"{self.waveform_data_url}/{format}/{self.waveform_filename_from_simname(sim_name)}"

    def download_waveform_data(
        self, sim_name: str, maya_format: bool = False, use_cache: bool | None = None
    ) -> None:
        """Download the waveform HDF5 file for *sim_name* into the local cache.

        By default downloads the LVCNR-format file.  If ``maya_format=True``
        the native MAYA format is downloaded and then converted to LVCNR
        format using the ``mayawaves`` package before the original is removed.

        Args:
            sim_name (str): MAYA simulation name, e.g. ``"GT0001"``.
            maya_format (bool): Download in native MAYA format and convert to
                LVCNR. Requires ``mayawaves`` to be installed. Defaults to False.
            use_cache (bool or None): Whether to skip download if a non-empty
                local file already exists.  If ``None``, falls back to the
                instance-level ``self.use_cache`` setting.
        """
        if maya_format:
            logger.info(
                "...WARNING: you have requested download of data in MAYA format"
            )
        if use_cache is None:
            use_cache = self.use_cache
        file_name = self.waveform_filename_from_simname(sim_name)
        file_path_web = self.waveform_url_from_simname(
            sim_name, maya_format=maya_format
        )
        local_file_path = self.waveform_data_dir / file_name
        if (
            use_cache
            and os.path.exists(local_file_path)
            and os.path.getsize(local_file_path) > 0
        ):
            if self._verbosity > 2:
                logger.info("...can read from cache: {}".format(str(local_file_path)))
            pass
        elif os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
            pass
        else:
            if self._verbosity > 2:
                logger.info("...writing to cache: {}".format(str(local_file_path)))
            if utils.url_exists(file_path_web):
                if self._verbosity > 2:
                    logger.info("...downloading {}".format(file_path_web))
                utils.download_file(file_path_web, local_file_path)
                if maya_format:
                    if self._verbosity > 2:
                        logger.info("...exporting to LVCNR catalog format")
                    try:
                        from mayawaves import coalescence as maya_coalescence
                        from mayawaves.utils import (
                            postprocessingutils as maya_postprocessingutils,
                        )
                    except ImportError as exc:
                        raise ImportError(
                            "mayawaves is required to convert MAYA-format files. "
                            "Install it with: pip install mayawaves"
                        ) from exc
                    maya_postprocessingutils.export_to_lvcnr_catalog(
                        maya_coalescence.Coalescence(local_file_path),
                        self.waveform_data_dir,
                        name=sim_name + "_LVCNR",
                        NR_group="UT Austin",
                        NR_code="MAYA",
                        bibtex_keys="Jani:2016wkt",
                        contact_email="email@email.com",
                        center_of_mass_correction=True,
                    )
                    if self._verbosity > 2:
                        logger.info("...removing maya format file")
                    os.remove(local_file_path)
                    if self._verbosity > 2:
                        logger.info("...renaming LVCNR format file in the cache")
                    os.rename(
                        self.waveform_data_dir / (sim_name + "_LVCNR.h5"),
                        local_file_path,
                    )
            else:
                if self._verbosity > 2:
                    logger.info(
                        "... ... but couldnt find link: {}".format(str(file_path_web))
                    )

    def psi4_filename_from_simname(self, _sim_name: str) -> str:
        """Not implemented; MAYA distributes strain waveforms only.

        Raises:
            NotImplementedError: Always. Use ``get(sim_name)`` for strain data.
        """
        raise NotImplementedError(
            "PSI4 data is not available for the MAYA catalog. "
            "Use the strain waveform data instead."
        )

    def psi4_filepath_from_simname(self, _sim_name: str) -> str:
        """Not implemented; MAYA distributes strain waveforms only.

        Raises:
            NotImplementedError: Always. Use ``get(sim_name)`` for strain data.
        """
        raise NotImplementedError(
            "PSI4 data is not available for the MAYA catalog. "
            "Use the strain waveform data instead."
        )

    def psi4_url_from_simname(self, _sim_name: str) -> str:
        """Not implemented; MAYA distributes strain waveforms only.

        Raises:
            NotImplementedError: Always. Use ``get(sim_name)`` for strain data.
        """
        raise NotImplementedError(
            "PSI4 data is not available for the MAYA catalog. "
            "Use the strain waveform data instead."
        )

    def download_psi4_data(self, _sim_name: str) -> None:
        """Not implemented; MAYA distributes strain waveforms only.

        Raises:
            NotImplementedError: Always. Use ``download_waveform_data()`` instead.
        """
        raise NotImplementedError(
            "PSI4 data is not available for the MAYA catalog. "
            "Use the strain waveform data instead."
        )
