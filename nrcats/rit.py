"""RIT catalog interface.

Provides access to the Rochester Institute of Technology (RIT) catalog of
numerical-relativity BBH waveforms generated with the LazEv code.

Metadata is scraped from the RIT web server
(``https://ccrgpages.rit.edu/~RITCatalog/Metadata/``) as individual
``*_Metadata.txt`` files, aggregated into a Pandas DataFrame, and cached
at ``~/.cache/RIT/metadata/metadata.csv``.  Waveform (HDF5) and psi4
(tar.gz) data are downloaded on demand.

Both quasicircular BBH (``RIT:BBH:NNNN-nRRR-idI``) and eccentric BBH
(``RIT:eBBH:NNNN-nRRR-ecc``) simulation names are supported.

Key design decisions
--------------------
* **Singleton pattern** – ``RITCatalog.load()`` stores its result in a
  module-level singleton to avoid the ``lru_cache`` stale-result bug
  (keyed on all arguments, a ``load(download=True)`` call after an earlier
  ``load(download=False)`` would have returned the stale cached result).

* **Two-class design** – ``RITCatalog`` exposes the public
  ``CatalogBase`` interface and delegates all scraping/caching/file-naming
  logic to ``RITCatalogHelper``, which can be instantiated and tested
  independently.

Public classes
--------------
RITCatalog
    Registered under the tag ``"RIT"`` in the catalog plugin registry.
RITCatalogHelper
    Internal helper; handles metadata scraping, caching, and filename
    conventions.  Not part of the public API but documented here because
    it is the main complexity in this module.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import collections
import functools
import glob
import os
import subprocess
import time

import pandas as pd
import requests
from tqdm import tqdm

from nrcats import catalog, utils
from nrcats.registry import register_catalog

# Module-level singleton — avoids the stale-result bug that lru_cache caused:
# lru_cache keyed on all arguments, so load(download=True) after an earlier
# load(download=False) returned the first (possibly incomplete) result.
_rit_catalog_singleton = None


@register_catalog("RIT")
class RITCatalog(catalog.CatalogBase):
    """Catalog interface for the RIT (LazEv) NR waveform collection.

    Delegates all file-naming, URL-construction, caching, and web-crawling
    logic to ``RITCatalogHelper``.  ``RITCatalog`` itself provides the
    ``CatalogBase``-compatible public API (``load()``, ``get()``,
    ``get_metadata()``, ``get_parameters()``, etc.).

    Key design points:

    - Metadata is scraped from ``https://ccrgpages.rit.edu/~RITCatalog/``
      on first load and aggregated into ``~/.cache/RIT/metadata/metadata.csv``.
    - Waveform HDF5 files (``ExtrapStrain_RIT-BBH-*.h5``) are downloaded
      to ``~/.cache/RIT/data/`` on demand.
    - Psi4 data is available as ``.tar.gz`` archives via
      ``download_psi4_data()`` / ``psi4_filepath_from_simname()``.
    - A module-level singleton prevents redundant catalog loads when
      ``load()`` is called multiple times in the same process.

    Example:
        >>> import nrcats as nrcat
        >>> cat = nrcat.RITCatalog.load(verbosity=0)
        >>> wfm = cat.get("RIT:BBH:0001-n100-id3")
    """

    CATALOG_TYPE = "RIT"

    def __init__(self, catalog=None, helper=None, verbosity: int = 3, **kwargs) -> None:
        """Initialise RITCatalog, loading metadata if *catalog* is None.

        Args:
            catalog: Pre-built catalog dict.  Pass ``None`` (default) to call
                :meth:`load` automatically.
            helper: ``RITCatalogHelper`` instance.  Populated automatically
                when *catalog* is None.
            verbosity (int): Logging verbosity level. Defaults to 3.
            **kwargs: Forwarded to :meth:`load`.
        """
        if catalog is not None:
            super().__init__(catalog["simulations"])
        else:
            obj = type(self).load(verbosity=verbosity, **kwargs)
            super().__init__(obj._simulations)
            helper = obj._helper
        self._helper = helper
        self._verbosity = verbosity

    @classmethod
    def load(
        cls,
        download: bool | None = None,
        num_sims_to_crawl: int = 2000,
        acceptable_scraping_fraction: float = 0.7,
        verbosity: int = 0,
    ) -> RITCatalog:
        """Load the RIT catalog.

        The result is cached in a module-level singleton after the first
        successful load.  Subsequent calls return the cached instance without
        re-reading disk or the network, unless ``download=True`` is passed.

        Pass ``download=True`` to force a fresh download and replace the
        singleton, or call ``RITCatalog.reload()`` for the same effect.

        Parameters
        ----------
        download : {None, bool}, optional
            If False, this function will look for the catalog in the cache and
            raise an error if it is not found.  If True, this function will
            download the catalog and raise an error if the download fails.
            If None (the default), it will try to download the file, warn but
            fall back to the cache if that fails, and only raise an error if
            the catalog is not found in the cache.

        See Also
        --------
        RITCatalog.reload : Force a fresh download and replace the singleton.
        nrcats.utils.rit_catalog_info : Catalog info, including cache directory.
        """
        global _rit_catalog_singleton
        # Return the cached instance unless the caller explicitly wants a
        # fresh download.  This avoids the lru_cache bug where
        # load(download=True) after load(download=False) returned a stale result.
        if _rit_catalog_singleton is not None and download is not True:
            return _rit_catalog_singleton

        helper = RITCatalogHelper(use_cache=True, verbosity=verbosity)
        if verbosity > 2:
            logger.info("..Going to read RIT catalog metadata from cache.")
        catalog_df = helper.read_metadata_df_from_disk()
        if len(catalog_df) == 0:
            if verbosity > 2:
                logger.info(
                    "..Catalog metadata not found on disk. Going to refresh from cache."
                )
            catalog_df = helper.refresh_metadata_df_on_disk(
                num_sims_to_crawl=num_sims_to_crawl
            )
        elif len(catalog_df) < acceptable_scraping_fraction * num_sims_to_crawl:
            if verbosity > 2:
                logger.info(
                    """..Catalog metadata on disk is likely incomplete with only {} sims.
                    ...Going to refresh from cache.
                    """.format(
                        len(catalog_df)
                    )
                )
            catalog_df = helper.refresh_metadata_df_on_disk(
                num_sims_to_crawl=num_sims_to_crawl
            )

        if len(catalog_df) < acceptable_scraping_fraction * num_sims_to_crawl:
            if verbosity > 2:
                logger.info(
                    "Refreshing catalog metadata from cache did not work.",
                    "...Falling back to downloading metadata for the full",
                    "...catalog. This will take some time.",
                )
            if download:
                catalog_df = helper.download_metadata_for_catalog(
                    num_sims_to_crawl=num_sims_to_crawl
                )
            else:
                raise ValueError(
                    "Catalog not found in {}. Please set `download=True`".format(
                        helper.metadata_dir
                    )
                )
        # Build the catalog dict from the helper's DataFrame
        catalog_dict = {}
        simulations = {}
        for idx, row in catalog_df.iterrows():
            name = row["simulation_name"]
            metadata_dict = row.to_dict()
            simulations[name] = metadata_dict
        catalog_dict["simulations"] = simulations
        _rit_catalog_singleton = cls(
            catalog=catalog_dict, helper=helper, verbosity=verbosity
        )
        return _rit_catalog_singleton

    @classmethod
    def reload(cls, **kwargs) -> RITCatalog:
        """Force a fresh download and replace the cached singleton.

        Equivalent to ``RITCatalog.load(download=True, **kwargs)``.
        """
        global _rit_catalog_singleton
        _rit_catalog_singleton = None
        return cls.load(download=True, **kwargs)

    @property
    @functools.lru_cache()
    def simulations_dataframe(self) -> object:
        """All simulations as a Pandas DataFrame indexed by simulation name.

        Removes any unnamed index columns left over from CSV round-trips and
        sets ``simulation_name`` as both the index and an explicit column.

        Returns:
            pandas.DataFrame: DataFrame with one row per simulation and
            metadata fields as columns.
        """
        df = self._helper.metadata
        for col_name in list(df.columns):
            if "Unnamed" in col_name:
                df = df.drop(columns=[col_name])
                break
        self._helper.metadata = df
        df = df.set_index("simulation_name")
        df.index.names = [None]
        df["simulation_name"] = df.index.to_list()
        return df

    @property
    @functools.lru_cache()
    def files(self) -> dict:
        """Map of waveform and psi4 filenames to file-info dicts.

        Each value is a dict with keys: ``checksum`` (None), ``filename``,
        ``filesize`` (bytes; 0 if not cached), ``download`` (remote URL), and
        ``truepath`` (canonical local filename after deduplication).

        Returns:
            dict[str, dict]: Mapping from bare filename to file-info dict.
        """
        file_infos = {}
        for _, row in self.simulations_dataframe.iterrows():
            psi4_data_location = row["psi4_data_location"]
            path_str = os.path.basename(psi4_data_location)
            if os.path.exists(psi4_data_location):
                file_size = os.path.getsize(psi4_data_location)
            else:
                file_size = 0
            file_info = {
                "checksum": None,
                "filename": os.path.basename(psi4_data_location),
                "filesize": file_size,
                "download": row["psi4_data_link"],
            }
            file_infos[path_str] = file_info

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
        """Return the bare filename of the RIT metadata text file for *sim_name*.

        Args:
            sim_name: RIT simulation name tag, e.g. ``"RIT:BBH:0001-n100-id3"``.

        Returns:
            Filename string, e.g. ``"RIT:BBH:0001-n100-id3_Metadata.txt"``.
        """
        return self._helper.metadata_filename_from_simname(sim_name)

    def metadata_filepath_from_simname(self, sim_name: str) -> str:
        """Return the absolute local path to the metadata file for *sim_name*.

        Args:
            sim_name: RIT simulation name tag, e.g. ``"RIT:BBH:0001-n100-id3"``.

        Returns:
            Absolute path string to the cached ``.txt`` metadata file.

        Raises:
            RuntimeError: If the path stored in the metadata dict does not exist
                on disk.
        """
        file_path = self.get_metadata(sim_name)["metadata_location"]
        if not os.path.exists(file_path):
            raise RuntimeError(
                f"Could not resolve path for {sim_name}"
                f"..best calculated path = {file_path}"
            )
        return str(file_path)

    def metadata_url_from_simname(self, sim_name: str) -> str:
        """Return the remote URL for the metadata text file for *sim_name*.

        Args:
            sim_name: RIT simulation name tag.

        Returns:
            Full URL string on the RIT web server.
        """
        return (
            self._helper.metadata_url
            + "/"
            + self.metadata_filename_from_simname(sim_name)
        )

    def waveform_filename_from_simname(self, sim_name: str) -> str:
        """Return the bare filename of the RIT waveform HDF5 file for *sim_name*.

        Args:
            sim_name: RIT simulation name tag.

        Returns:
            Filename string, e.g. ``"ExtrapStrain_RIT-BBH-0001-n100.h5"``.
        """
        return self._helper.waveform_filename_from_simname(sim_name)

    def waveform_filepath_from_simname(self, sim_name: str) -> str:
        """Return the absolute local cache path to the waveform HDF5 for *sim_name*.

        Falls back to re-anchoring the filename under the current cache directory
        when the stored path (from a shared CSV) belongs to a different machine.

        Args:
            sim_name: RIT simulation name tag.

        Returns:
            Absolute path string to the local ``.h5`` waveform file.  Returns the
            stored (possibly stale) path if the file is not yet downloaded.
        """
        file_path = self.get_metadata(sim_name)["waveform_data_location"]
        if not os.path.exists(file_path):
            # The stored path may be an absolute path from a different machine
            # (e.g. a committed CSV with developer-local paths).  Re-anchor the
            # filename under the current cache directory.
            canonical = self._helper.data_dir / os.path.basename(file_path)
            if os.path.exists(canonical):
                return str(canonical)
            if self._verbosity > 2:
                logger.info(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
        return str(file_path)

    def waveform_url_from_simname(self, sim_name: str) -> str:
        """Return the remote URL of the waveform HDF5 file for *sim_name*.

        Args:
            sim_name: RIT simulation name tag.

        Returns:
            Full URL string on the RIT web server.
        """
        return (
            self._helper.waveform_data_url
            + "/"
            + self.waveform_filename_from_simname(sim_name)
        )

    def refresh_metadata_df_on_disk(self, num_sims_to_crawl: int = 2000) -> object:
        """Delegate to ``RITCatalogHelper.refresh_metadata_df_on_disk()``.

        Args:
            num_sims_to_crawl (int): Upper bound on the simulation index.
                Defaults to 2000.

        Returns:
            pandas.DataFrame: Refreshed aggregated metadata DataFrame.
        """
        return self._helper.refresh_metadata_df_on_disk(
            num_sims_to_crawl=num_sims_to_crawl
        )

    def download_data_for_catalog(
        self,
        num_sims_to_crawl: int = 2000,
        which_data: str = "waveform",
        possible_res: list | None = None,
        max_id_in_name: int = -1,
        use_cache: bool = True,
    ) -> dict:
        """Download waveform or psi4 data for all simulations in the catalog.

        Args:
            num_sims_to_crawl (int): Maximum number of simulations to process.
                Defaults to 2000.
            which_data (str): ``"waveform"`` or ``"psi4"``. Defaults to
                ``"waveform"``.
            possible_res (list[int] or None): Resolution values to try.
                Defaults to the list in ``utils.rit_catalog_info``.
            max_id_in_name (int): Maximum ID suffix to search for. Defaults to
                ``-1`` (uses the value in ``utils.rit_catalog_info``).
            use_cache (bool): Skip download if a non-empty file exists locally.
                Defaults to True.

        Returns:
            dict[str, pathlib.Path]: Mapping from simulation name to the
            local file path for each successfully downloaded file.
        """
        return self._helper.download_data_for_catalog(
            num_sims_to_crawl=num_sims_to_crawl,
            which_data=which_data,
            possible_res=possible_res if possible_res is not None else [],
            max_id_in_name=max_id_in_name,
            use_cache=use_cache,
        )

    def write_metadata_df_to_disk(self) -> None:
        """Write the current aggregated metadata DataFrame to ``metadata.csv``.

        Delegates to ``RITCatalogHelper.write_metadata_df_to_disk()``.
        """
        return self._helper.write_metadata_df_to_disk()

    def download_waveform_data(
        self, sim_name: str, use_cache: bool | None = None
    ) -> bool:
        """Download the waveform HDF5 file for *sim_name*.

        Args:
            sim_name (str): RIT simulation name tag,
                e.g. ``"RIT:BBH:0001-n100-id3"``.
            use_cache (bool or None): Use the cached file if present.
                Defaults to None (uses helper's default).

        Returns:
            bool: True if the file is available locally after the call.
        """
        return self._helper.download_waveform_data(sim_name, use_cache=use_cache)

    def psi4_filename_from_simname(self, sim_name: str) -> str:
        """Return the bare filename of the RIT psi4 tar.gz archive for *sim_name*.

        Args:
            sim_name: RIT simulation name tag.

        Returns:
            Filename string, e.g.
            ``"ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz"``.
        """
        return self._helper.psi4_filename_from_simname(sim_name)

    def psi4_filepath_from_simname(self, sim_name: str) -> str:
        """Return the absolute local path to the psi4 archive for *sim_name*.

        Args:
            sim_name: RIT simulation name tag.

        Returns:
            Absolute path string, or an empty string if the file is not yet
            downloaded.
        """
        file_path = self.get_metadata(sim_name)["psi4_data_location"]
        if not os.path.exists(file_path):
            if self._verbosity > 2:
                logger.info(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
            return ""
        return str(file_path)

    def psi4_url_from_simname(self, sim_name: str) -> str:
        """Return the remote URL of the psi4 archive for *sim_name*.

        Args:
            sim_name: RIT simulation name tag.

        Returns:
            Full URL string on the RIT web server.
        """
        return (
            self._helper.psi4_data_url + "/" + self.psi4_filename_from_simname(sim_name)
        )

    def download_psi4_data(self, sim_name: str, use_cache: bool | None = None) -> bool:
        """Download the psi4 tar.gz archive for *sim_name*.

        Args:
            sim_name: RIT simulation name tag.
            use_cache: Skip download if a non-empty file exists locally.
                Defaults to None (uses helper's default).

        Returns:
            True if the file is available locally after the call.
        """
        return self._helper.download_psi4_data(sim_name, use_cache=use_cache)


class RITCatalogHelper(object):
    """Internal helper for RIT catalog scraping, caching, and file naming.

    Handles all the catalog-specific complexity that ``RITCatalog`` delegates:

    - **File naming** – converts simulation name tags (e.g.
      ``"RIT:BBH:0001-n100-id3"``) to metadata filenames, waveform filenames,
      and psi4 filenames in the formats used on the RIT web server.
    - **Metadata scraping** – downloads ``*_Metadata.txt`` files from the RIT
      web server one-by-one, parses them with ``parse_metadata_txt()``, and
      aggregates them into a Pandas DataFrame.
    - **Disk caching** – reads/writes the aggregated DataFrame as
      ``~/.cache/RIT/metadata/metadata.csv`` and stores individual
      ``*_Metadata.txt`` files in ``~/.cache/RIT/metadata/``.
    - **Data downloads** – downloads waveform HDF5 and psi4 tar.gz files via
      ``wget`` into ``~/.cache/RIT/data/``.

    This class is not part of the public API; use ``RITCatalog`` instead.
    """

    def __init__(
        self, _catalog=None, use_cache: bool = True, verbosity: int = 0
    ) -> None:
        """Initialise cache paths, URLs, and format strings for the RIT catalog.

        Args:
            _catalog: Unused; present for interface compatibility.
            use_cache (bool): Read from local cache when available.
                Defaults to True.
            verbosity (int): Logging verbosity level. Defaults to 0.
        """
        self.verbosity = verbosity
        self.catalog_url = utils.rit_catalog_info["url"]
        self.use_cache = use_cache
        self.cache_dir = utils.rit_catalog_info["cache_dir"]

        self.num_of_sims = 0

        self.metadata = pd.DataFrame.from_dict({})
        self.metadata_url = utils.rit_catalog_info["metadata_url"]
        self.metadata_file_fmts = utils.rit_catalog_info["metadata_file_fmts"]
        self.metadata_dir = utils.rit_catalog_info["metadata_dir"]

        self.psi4_data = {}
        self.psi4_data_url = utils.rit_catalog_info["data_url"]
        self.psi4_file_fmts = utils.rit_catalog_info["psi4_file_fmts"]

        self.waveform_data = {}
        self.waveform_data_url = utils.rit_catalog_info["data_url"]
        self.waveform_file_fmts = utils.rit_catalog_info["waveform_file_fmts"]

        self.data_dir = utils.rit_catalog_info["data_dir"]
        self.waveform_data_dir = self.data_dir
        self.psi4_data_dir = self.data_dir

        self.possible_res = utils.rit_catalog_info["possible_resolutions"]
        self.max_id_val = utils.rit_catalog_info["max_id_val"]

        internal_dirs = [
            self.cache_dir,
            self.metadata_dir,
            self.psi4_data_dir,
            self.waveform_data_dir,
        ]
        for d in internal_dirs:
            d.mkdir(parents=True, exist_ok=True)

    def metadata_filenames(self, idx: int, res: int, id_val: int) -> list:
        """Return all candidate metadata filenames for simulation *idx*.

        Returns one name for the quasicircular BBH format and one for the
        eccentric BBH format, since a given index may correspond to either.

        Args:
            idx (int): Four-digit simulation index (e.g. ``1`` for ``0001``).
            res (int): Numerical resolution tag (e.g. ``100``).
            id_val (int): ID suffix for quasicircular simulations (e.g. ``3``).

        Returns:
            list[str]: Two candidate filenames:
            ``["RIT:BBH:NNNN-nRRR-idI_Metadata.txt",
            "RIT:eBBH:NNNN-nRRR-ecc_Metadata.txt"]``.
        """
        return [
            self.metadata_file_fmts[0].format(idx, res, id_val),
            self.metadata_file_fmts[1].format(idx, res),
        ]

    def sim_info_from_metadata_filename(self, file_name: str) -> tuple:
        """
        Input:
        ------
        file_name: name (not path) of metadata file as hosted on the web

        Output:
        -------
        - simulation number
        - resolution as indicated with an integer
        - ID value (only for non-eccentric simulations)
        """
        sim_number = int(file_name.split("-")[0][-4:])
        res_number = int(file_name.split("-")[1][1:])
        try:
            id_val = int(file_name.split("-")[2].split("_")[0][2:])
        except Exception:
            id_val = -1
        return (sim_number, res_number, id_val)

    def simname_from_metadata_filename(self, filename: str) -> str:
        """
        Input:
        ------
        - filename: name (not path) of metadata file as hosted on the web

        Output:
        -------
        - Simulation Name Tag (Class uses this tag for internal indexing)
        """
        return filename.split("_Meta")[0]

    def metadata_filename_from_simname(self, sim_name: str) -> str:
        """
        We assume the sim names are either of the format:
        (1) RIT:eBBH:1109-n100-ecc
        (2) RIT:BBH:1109-n100-id1
        """
        txt = sim_name.split(":")[-1]
        idx = int(txt[:4])
        res = int(txt.split("-")[1][1:])
        if "eBBH" not in sim_name:
            # If this works, its a quasicircular sim
            id_val = int(txt[-1])
            return self.metadata_file_fmts[0].format(idx, res, id_val)
        else:
            return self.metadata_file_fmts[1].format(idx, res)

    def metadata_filename_from_cache(self, idx: int) -> str:
        """Return the path of the cached metadata file for simulation *idx*.

        Searches the local metadata cache directory for any file whose name
        starts with either the BBH or eBBH sim-tag prefix for this index.

        Args:
            idx (int): Four-digit simulation index.

        Returns:
            str: Full path of the first matching cached file, or ``""`` if no
            cached file is found.
        """
        possible_sim_tags = self.simtags(idx)
        file_name = ""
        for sim_tag in possible_sim_tags:
            mf = self.metadata_dir / sim_tag
            poss_files = glob.glob(str(mf) + "*")
            if len(poss_files) == 0:
                if self.verbosity > 4:
                    logger.info("...found no files matching {}".format(str(mf) + "*"))
                continue
            file_name = poss_files[0]
        return file_name

    def psi4_filename_from_simname(self, sim_name: str) -> str:
        """
        We assume the sim names are either of the format:
        (1) RIT:eBBH:1109-n100-ecc
        (2) RIT:BBH:1109-n100-id1
        """
        txt = sim_name.split(":")[-1]
        idx = int(txt[:4])
        res = int(txt.split("-")[1][1:])
        if "eBBH" not in sim_name:
            # If this works, its a quasicircular sim
            id_val = int(txt[-1])
            return self.psi4_file_fmts[0].format(idx, res, id_val)
        else:
            return self.psi4_file_fmts[1].format(idx, res)

    def psi4_filename_from_cache(self, idx: int) -> str:
        """Return the psi4 filename for simulation *idx* via the cache.

        Looks up the cached metadata filename for *idx*, derives the
        simulation name from it, and then computes the psi4 filename.

        Args:
            idx (int): Four-digit simulation index.

        Returns:
            str: Psi4 filename, e.g.
            ``"ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz"``.
        """
        return self.psi4_filename_from_simname(
            self.simname_from_metadata_filename(self.metadata_filename_from_cache(idx))
        )

    def waveform_filename_from_simname(self, sim_name: str) -> str:
        """
        ExtrapStrain_RIT-BBH-0005-n100.h5 -->
        ExtrapStrain_RIT-eBBH-1843-n100.h5
        RIT:eBBH:1843-n100-ecc_Metadata.txt
        """
        txt = sim_name.split(":")[-1]
        idx = int(txt[:4])
        res = int(txt.split("-")[1][1:])
        try:
            # If this works, its a quasicircular sim
            id_val = int(txt[-1])
            mf = self.metadata_file_fmts[0].format(idx, res, id_val)
        except Exception:
            mf = self.metadata_file_fmts[1].format(idx, res)
        parts = mf.split(":")
        return (
            "ExtrapStrain_"
            + parts[0]
            + "-"
            + parts[1]
            + "-"
            + parts[2].split("_")[0].split("-")[0]
            + "-"
            + parts[2].split("_")[0].split("-")[1]
            + ".h5"
        )

    def waveform_filename_from_cache(self, idx: int) -> str:
        """Return the waveform HDF5 filename for simulation *idx* via the cache.

        Looks up the cached metadata filename for *idx*, derives the
        simulation name from it, and then computes the waveform filename.

        Args:
            idx (int): Four-digit simulation index.

        Returns:
            str: Waveform filename, e.g.
            ``"ExtrapStrain_RIT-BBH-0001-n100.h5"``.
        """
        return self.waveform_filename_from_simname(
            self.simname_from_metadata_filename(self.metadata_filename_from_cache(idx))
        )

    def simname_from_cache(self, idx: int) -> str:
        """Return the simulation name tag for *idx* by inspecting the cache.

        Searches the metadata cache directory for a file matching either the
        BBH or eBBH prefix, then derives the simulation name from the
        filename.

        Args:
            idx (int): Four-digit simulation index.

        Returns:
            str: Simulation name tag (e.g. ``"RIT:BBH:0001-n100-id3"``), or
            ``""`` if no cached metadata file is found for *idx*.
        """
        possible_sim_tags = self.simtags(idx)
        for sim_tag in possible_sim_tags:
            mf = self.metadata_dir / sim_tag
            poss_files = glob.glob(str(mf) + "*")
            if len(poss_files) == 0:
                if self.verbosity > 4:
                    logger.info("...found no files matching {}".format(str(mf) + "*"))
                continue
            file_path = poss_files[0]  # glob gives full paths
            file_name = os.path.basename(file_path)
            return self.simname_from_metadata_filename(file_name)
        return ""

    def simnames(self, idx: int, res: int, id_val: int) -> list:
        """Return candidate simulation name tags for *idx* at *res* and *id_val*.

        Args:
            idx (int): Four-digit simulation index.
            res (int): Numerical resolution tag.
            id_val (int): ID suffix for quasicircular simulations.

        Returns:
            list[str]: Two candidate simulation names (BBH and eBBH formats).
        """
        return [
            self.simname_from_metadata_filename(mf)
            for mf in self.metadata_filenames(idx, res, id_val)
        ]

    def simtags(self, idx: int) -> list:
        """Return the filename-prefix tags used to glob-search for *idx*.

        Returns one prefix for the BBH format and one for the eBBH format,
        which are used to search for cached metadata files with
        ``glob(prefix + "*")``.

        Args:
            idx (int): Four-digit simulation index.

        Returns:
            list[str]: Two prefix strings, e.g.
            ``["RIT:BBH:0001", "RIT:eBBH:0001"]``.
        """
        return [
            self.metadata_file_fmts[0].split("-")[0].format(idx),
            self.metadata_file_fmts[1].split("-")[0].format(idx),
        ]

    def parse_metadata_txt(self, raw: list) -> tuple:
        """Parses raw RIT metadata

        Args:
            raw (list(str)): List of lines read in from RIT metadata

        Returns:
           list(str): Original metadata with empty lines removed
           dict     : Parsed metadata as a dictionary
        """
        derived_fields = [
            "freq-start-22",
            "freq-start-22-Hz-1Msun",
            "number-of-cycles-22",
            "number-of-orbits",
            "peak-omega-22",
            "peak-ampl-22",
            "Msun",
            "eccentricity",
        ]
        nxt = [s for s in raw if len(s) > 0 and s[0].isalpha()]
        opts = {}
        for s in nxt:
            kv = s.split("=")
            try:
                opts[kv[0].strip()] = float("=".join(kv[1:]).strip())
            except Exception:
                # If any of the following fields are empty in metadata, they are
                # set to 0 here
                reasonable_value_set = False
                for xy in derived_fields:
                    if (kv[0].strip() == xy) and (xy not in opts):
                        opts[xy] = 0.0
                        reasonable_value_set = True
                        break
                if not reasonable_value_set:
                    opts[kv[0].strip()] = str("=".join(kv[1:]).strip())

        # Note: often when some spin components are 0, they are not
        # even included in the metadata file. We set them to 0 here.
        if "relaxed-chi1z" in opts:
            for xy in ["relaxed-chi1x", "relaxed-chi1y"]:
                if xy not in opts and (
                    opts["system-type"].lower() == "aligned"
                    or opts["system-type"].lower() == "nonspinning"
                ):
                    opts[xy] = 0.0
        if "relaxed-chi2z" in opts:
            for xy in ["relaxed-chi2x", "relaxed-chi2y"]:
                if xy not in opts and (
                    opts["system-type"].lower() == "aligned"
                    or opts["system-type"].lower() == "nonspinning"
                ):
                    opts[xy] = 0.0

        if "initial-bh-chi1z" in opts:
            for xy in ["initial-bh-chi1x", "initial-bh-chi1y"]:
                if xy not in opts and (
                    opts["system-type"].lower() == "aligned"
                    or opts["system-type"].lower() == "nonspinning"
                ):
                    opts[xy] = 0.0
        if "initial-bh-chi2z" in opts:
            for xy in ["initial-bh-chi2x", "initial-bh-chi2y"]:
                if xy not in opts and (
                    opts["system-type"].lower() == "aligned"
                    or opts["system-type"].lower() == "nonspinning"
                ):
                    opts[xy] = 0.0

        # derived fields might not be populated at all. In that case, they are
        # set to 0 here.
        if "number-of-cycles-22" in opts:
            if "number-of-orbits" not in opts:
                opts["number-of-orbits"] = opts["number-of-cycles-22"] / 2.0
            if opts["number-of-orbits"] == 0.0:
                opts["number-of-orbits"] = opts["number-of-cycles-22"] / 2.0

        if "number-of-orbits" in opts:
            if "number-of-cycles-22" not in opts:
                opts["number-of-cycles-22"] = opts["number-of-orbits"] * 2.0
            if opts["number-of-cycles-22"] == 0.0:
                opts["number-of-cycles-22"] = opts["number-of-orbits"] * 2.0

        for xy in derived_fields:
            if xy not in opts:
                opts[xy] = 0.0

        return nxt, opts

    def metadata_from_link(
        self, link: str, save_to: object = None, num_retries: int = 5
    ) -> tuple:
        """Fetch and parse a single RIT metadata file from a URL.

        If *save_to* is given, downloads the file to disk and then parses it
        with ``metadata_from_file()``.  Otherwise performs an in-memory HTTP
        GET and parses the response body directly.

        Args:
            link (str): Full HTTP(S) URL to the ``*_Metadata.txt`` file.
            save_to (str or pathlib.Path or None): If provided, save the
                downloaded text to this path before parsing. Defaults to None.
            num_retries (int): Number of request attempts with exponential
                backoff. Defaults to 5.

        Returns:
            tuple[list[str], dict]: The output of ``parse_metadata_txt()``:
            a list of non-empty metadata lines and a dict of parsed fields.

        Raises:
            ConnectionError: If all retry attempts fail.
        """
        if save_to is not None:
            utils.download_file(link, save_to, progress=True)
            return self.metadata_from_file(save_to)
        else:
            requests.packages.urllib3.disable_warnings()
            last_exc = None
            response = None
            for attempt in range(num_retries):
                try:
                    response = requests.get(link, verify=False, timeout=10)
                    break
                except Exception as exc:
                    if isinstance(exc, requests.exceptions.Timeout):
                        import warnings

                        warnings.warn(f"Request to {link} timed out after 10 seconds.")
                    last_exc = exc
                    if attempt < num_retries - 1:
                        delay = min(2**attempt, 30)
                        if self.verbosity > 0:
                            logger.info(
                                f"metadata_from_link: attempt {attempt + 1}/{num_retries}"
                                f" failed for {link}; retrying in {delay}s"
                            )
                        time.sleep(delay)
            if response is None:
                raise ConnectionError(
                    f"Failed to fetch metadata from '{link}' after {num_retries} attempts"
                ) from last_exc
            return self.parse_metadata_txt(response.content.decode().split("\n"))

    def metadata_from_file(self, file_path: object) -> tuple:
        """Parse a locally cached RIT metadata text file.

        Args:
            file_path (str or pathlib.Path): Path to the ``*_Metadata.txt``
                file on disk.

        Returns:
            tuple[list[str], dict]: The output of ``parse_metadata_txt()``:
            a list of non-empty metadata lines and a dict of parsed fields.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        return self.parse_metadata_txt(lines)

    def metadata_from_cache(self, idx: int) -> object:
        """Build a single-row DataFrame from a cached metadata file for *idx*.

        Searches the local metadata cache for any file matching the BBH or
        eBBH prefix for *idx*.  If found, parses it and enriches the result
        with the simulation name, metadata/psi4/waveform URLs and local paths.

        Args:
            idx (int): Four-digit simulation index.

        Returns:
            pandas.DataFrame: Single-row DataFrame with all metadata fields
            plus ``simulation_name``, ``metadata_link``, ``metadata_location``,
            ``psi4_data_link``, ``psi4_data_location``, ``waveform_data_link``,
            and ``waveform_data_location`` columns.  Returns an empty DataFrame
            if no cached file is found for *idx*.
        """
        possible_sim_tags = self.simtags(idx)
        for sim_tag in possible_sim_tags:
            mf = self.metadata_dir / sim_tag
            poss_files = glob.glob(str(mf) + "*")
            if len(poss_files) == 0:
                if self.verbosity > 4:
                    logger.info("...found no files matching {}".format(str(mf) + "*"))
                continue
            file_path = poss_files[0]  # glob gives full paths
            file_name = os.path.basename(file_path)
            file_path_web = self.metadata_url + "/" + file_name
            psi4_file_name = self.psi4_filename_from_cache(idx)
            psi4_file_path_web = self.psi4_data_url + "/" + psi4_file_name
            wf_file_name = self.waveform_filename_from_cache(idx)
            wf_file_path_web = self.waveform_data_url + "/" + wf_file_name
            _, metadata_dict = self.metadata_from_file(file_path)

            if len(metadata_dict) > 0:
                metadata_dict["simulation_name"] = [
                    self.simname_from_metadata_filename(file_name)
                ]
                metadata_dict["metadata_link"] = [file_path_web]
                metadata_dict["metadata_location"] = [file_path]
                metadata_dict["psi4_data_link"] = [psi4_file_path_web]
                metadata_dict["psi4_data_location"] = [
                    str(
                        self.psi4_data_dir
                        / self.psi4_filename_from_simname(
                            metadata_dict["simulation_name"][0]
                        )
                    )
                ]
                metadata_dict["waveform_data_link"] = [wf_file_path_web]
                metadata_dict["waveform_data_location"] = [
                    str(
                        self.waveform_data_dir
                        / self.waveform_filename_from_simname(
                            metadata_dict["simulation_name"][0]
                        )
                    )
                ]
                return pd.DataFrame.from_dict(metadata_dict)
        return pd.DataFrame({})

    def download_metadata(self, idx: int, res: int, id_val: int = -1) -> object:
        """Download (or read from cache) metadata for one simulation.

        Tries the BBH filename format first, then the eBBH format.  For each
        candidate filename, checks the local cache first (if ``use_cache`` is
        True) before making an HTTP request.  The downloaded file is saved
        to the metadata cache directory.

        Args:
            idx (int): Four-digit simulation index.
            res (int): Numerical resolution tag (e.g. ``100``).
            id_val (int): ID suffix for quasicircular simulations. Defaults to
                ``-1`` (triggers the eccentric filename format as fallback).

        Returns:
            pandas.DataFrame: Single-row DataFrame (same schema as
            ``metadata_from_cache()``), or an empty DataFrame if neither
            filename is found locally or remotely.
        """
        possible_file_names = [
            self.metadata_file_fmts[0].format(idx, res, id_val),
            self.metadata_file_fmts[1].format(idx, res),
        ]
        metadata_txt, metadata_dict = "", {}

        for file_name in possible_file_names:
            if self.verbosity > 2:
                logger.info("...beginning search for {}".format(file_name))
            file_path_web = self.metadata_url + "/" + file_name
            mf = self.metadata_dir / file_name
            psi4_file_name = self.psi4_filename_from_simname(
                self.simname_from_metadata_filename(file_name)
            )
            psi4_file_path_web = self.psi4_data_url + "/" + psi4_file_name
            wf_file_name = self.waveform_filename_from_simname(
                self.simname_from_metadata_filename(file_name)
            )
            wf_file_path_web = self.waveform_data_url + "/" + wf_file_name

            if self.use_cache:
                if os.path.exists(mf) and os.path.getsize(mf) > 0:
                    if self.verbosity > 2:
                        logger.info("...reading from cache: {}".format(str(mf)))
                    metadata_txt, metadata_dict = self.metadata_from_file(mf)

            if len(metadata_dict) == 0:
                if utils.url_exists(file_path_web):
                    if self.verbosity > 2:
                        logger.info("...found {}".format(file_path_web))
                    metadata_txt, metadata_dict = self.metadata_from_link(
                        file_path_web, save_to=mf
                    )
                else:
                    if self.verbosity > 3:
                        logger.info(
                            "...tried and failed to find {}".format(file_path_web)
                        )

            if len(metadata_dict) > 0:
                # Convert to DataFrame and break loop
                metadata_dict["simulation_name"] = [
                    self.simname_from_metadata_filename(file_name)
                ]
                metadata_dict["metadata_link"] = [file_path_web]
                metadata_dict["metadata_location"] = [mf]
                metadata_dict["psi4_data_link"] = [psi4_file_path_web]
                metadata_dict["psi4_data_location"] = [
                    str(
                        self.psi4_data_dir
                        / self.psi4_filename_from_simname(
                            metadata_dict["simulation_name"][0]
                        )
                    )
                ]
                metadata_dict["waveform_data_link"] = [wf_file_path_web]
                metadata_dict["waveform_data_location"] = [
                    str(
                        self.waveform_data_dir
                        / self.waveform_filename_from_simname(
                            metadata_dict["simulation_name"][0]
                        )
                    )
                ]
                break

        return pd.DataFrame.from_dict(metadata_dict)

    def download_metadata_for_catalog(
        self,
        num_sims_to_crawl: int = 2000,
        possible_res: list = [],
        max_id_in_name: int = -1,
    ) -> object:
        """
        We crawl the webdirectory where RIT metadata usually lives,
        and try to read metadata for as many simulations as we can
        """
        if len(possible_res) == 0:
            possible_res = self.possible_res
        if max_id_in_name <= 0:
            max_id_in_name = self.max_id_val
        import pandas as pd

        sims = pd.DataFrame({})

        if self.use_cache:
            metadata_df_fpath = self.metadata_dir / "metadata.csv"
            if (
                os.path.exists(metadata_df_fpath)
                and os.path.getsize(metadata_df_fpath) > 0
            ):
                if self.verbosity > 2:
                    logger.info("Opening file {}".format(metadata_df_fpath))
                self.metadata = pd.read_csv(metadata_df_fpath)
                if len(self.metadata) >= (num_sims_to_crawl - 1):
                    # return self.metadata
                    return self.metadata.iloc[: num_sims_to_crawl - 1]
                else:
                    sims = self.metadata
        if self.verbosity > 2:
            logger.info("Found metadata for {} sims".format(len(sims)))

        import concurrent.futures

        def _fetch_metadata(idx):
            found = False
            possible_sim_tags = self.simtags(idx)

            if self.verbosity > 3:
                logger.info("\nHunting for sim with idx: {}".format(idx))

            # First, check if metadata present as file on disk
            if not found and self.use_cache:
                if self.verbosity > 3:
                    logger.info("checking for metadata file on disk")
                sim_data = self.metadata_from_cache(idx)
                if len(sim_data) > 0:
                    found = True
                    if self.verbosity > 3:
                        logger.info("...metadata found on disk for {}".format(idx))

            # Second, check if metadata present already in DataFrame
            if len(sims) > 0 and not found:
                if self.verbosity > 1:
                    logger.info("Checking existing dataframe")
                for _, row in sims.iterrows():
                    name = row["simulation_name"]
                    for sim_tag in possible_sim_tags:
                        if sim_tag in name:
                            found = True
                            f_idx, res, id_val = self.sim_info_from_metadata_filename(
                                name
                            )
                            if f_idx != idx:
                                continue  # just ignore mismatch in parallel
                            if self.verbosity > 3:
                                logger.info(
                                    "...metadata found in DF for {}, {}, {}".format(
                                        idx, res, id_val
                                    )
                                )
                            sim_data = pd.DataFrame.from_dict(row.to_dict(), index=[0])
                            break

            # If not already present, fetch metadata the hard way
            if not found:
                for res in possible_res[::-1]:
                    for id_val in range(max_id_in_name):
                        # If not already present, fetch metadata
                        sim_data = self.download_metadata(idx, res, id_val)
                        if len(sim_data) > 0:
                            found = True
                            if self.verbosity > 3:
                                logger.info(
                                    "...metadata txt file found for {}, {}, {}".format(
                                        idx, res, id_val
                                    )
                                )
                            break
                        else:
                            if self.verbosity > 3:
                                logger.info(
                                    "...metadata not found for {}, {}, {}".format(
                                        idx, res, id_val
                                    )
                                )
                    if found:
                        break
            if found:
                return sim_data
            else:
                if self.verbosity > 3:
                    logger.info(
                        "...metadata for {} NOT FOUND.".format(possible_sim_tags)
                    )
                return None

        sim_data_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                tqdm(
                    executor.map(_fetch_metadata, range(1, 1 + num_sims_to_crawl)),
                    total=num_sims_to_crawl,
                )
            )
            for res in results:
                if res is not None:
                    sim_data_list.append(res)

        if sim_data_list:
            if len(sims) > 0:
                # remove overlap before concat
                sims = pd.concat([sims] + sim_data_list)
                # deduplicate
                sims = sims.drop_duplicates(subset=["simulation_name"], keep="last")
            else:
                sims = pd.concat(sim_data_list)

        self.metadata = sims
        if self.use_cache:
            self.write_metadata_df_to_disk()

        self.num_of_sims = len(sims)
        return self.metadata

    def write_metadata_df_to_disk(self) -> None:
        """Write the current ``self.metadata`` DataFrame to ``metadata.csv``.

        Saves to ``~/.cache/RIT/metadata/metadata.csv`` (or the path
        configured via ``NR_CATALOG_CACHE``).  Called automatically after
        each simulation's metadata is downloaded during a catalog crawl.
        """
        metadata_df_fpath = self.metadata_dir / "metadata.csv"
        with open(metadata_df_fpath, "w+") as f:
            try:
                self.metadata.to_csv(f)
            except Exception:
                self.metadata.reset_index(drop=True, inplace=True)
                self.metadata.to_csv(f)

    def refresh_metadata_df_on_disk(self, num_sims_to_crawl: int = 2000) -> object:
        """Rebuild the metadata CSV from cached ``*_Metadata.txt`` files.

        Iterates over simulation indices 1 … *num_sims_to_crawl*, reads each
        simulation's metadata from the local file cache (does **not** make
        network requests), concatenates the results, and writes the aggregated
        DataFrame to ``metadata.csv``.

        Args:
            num_sims_to_crawl (int): Upper bound on the simulation index to
                scan. Defaults to 2000.

        Returns:
            pandas.DataFrame: The refreshed aggregated metadata DataFrame.
        """
        import concurrent.futures

        def _fetch(idx):
            return self.metadata_from_cache(idx)

        sims_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for sim_data in tqdm(
                executor.map(_fetch, range(1, 1 + num_sims_to_crawl)),
                total=num_sims_to_crawl,
            ):
                if len(sim_data) > 0:
                    sims_list.append(sim_data)

        if sims_list:
            sims = pd.concat(sims_list)
        else:
            sims = pd.DataFrame([])
        sims.reset_index(drop=True, inplace=True)
        metadata_df_fpath = self.metadata_dir / "metadata.csv"
        with open(metadata_df_fpath, "w") as f:
            sims.to_csv(f)
        self.metadata = sims  # set this member
        return self.metadata

    def read_metadata_df_from_disk(self) -> object:
        """Load the aggregated metadata DataFrame from ``metadata.csv``.

        If the CSV file does not exist or is empty, sets ``self.metadata`` to
        an empty DataFrame and returns it.

        Returns:
            pandas.DataFrame: The previously saved aggregated metadata, or an
            empty DataFrame if the cache file is absent.
        """
        metadata_df_fpath = self.metadata_dir / "metadata.csv"
        if os.path.exists(metadata_df_fpath) and os.path.getsize(metadata_df_fpath) > 0:
            self.metadata = pd.read_csv(metadata_df_fpath)
        else:
            self.metadata = pd.DataFrame([])
        return self.metadata

    def download_psi4_data(self, sim_name: str, use_cache: bool | None = True) -> bool:
        """Download the psi4 tar.gz file for *sim_name* via ``wget``.

        Skips the download if ``use_cache`` is True and a non-empty local
        file already exists.

        Args:
            sim_name (str): RIT simulation name tag, e.g.
                ``"RIT:BBH:0001-n100-id3"``.
            use_cache (bool or None): Use cached file if present.  If
                ``None``, falls back to the instance-level ``self.use_cache``.
                Defaults to True.

        Returns:
            bool: True if the file is available locally (either from cache or
            after a successful download), False if the URL was not found.
        """
        if use_cache is None:
            use_cache = self.use_cache
        file_name = self.psi4_filename_from_simname(sim_name)
        file_path_web = self.psi4_data_url + "/" + file_name
        local_file_path = self.psi4_data_dir / file_name
        if (
            use_cache
            and os.path.exists(local_file_path)
            and os.path.getsize(local_file_path) > 0
        ):
            if self.verbosity > 2:
                logger.info("...can read from cache: {}".format(str(local_file_path)))
            return True
        else:
            if self.verbosity > 2:
                logger.info("...writing to cache: {}".format(str(local_file_path)))
            if utils.url_exists(file_path_web):
                if self.verbosity > 2:
                    logger.info("...downloading {}".format(file_path_web))
                subprocess.call(
                    [
                        "wget",
                        "--no-check-certificate",
                        str(file_path_web),
                        "-O",
                        str(local_file_path),
                    ]
                )
                return True
            else:
                if self.verbosity > 2:
                    logger.info(
                        "... ... but couldnt find link: {}".format(str(file_path_web))
                    )
                return False

    def download_waveform_data(
        self, sim_name: str, use_cache: bool | None = True
    ) -> bool:
        """Download the waveform HDF5 file for *sim_name* via ``wget``.

        Skips the download if ``use_cache`` is True and a non-empty local
        file already exists.

        Possible file formats:

        - ``https://ccrgpages.rit.edu/~RITCatalog/Data/ExtrapStrain_RIT-BBH-0193-n100.h5``
        - ``https://ccrgpages.rit.edu/~RITCatalog/Data/ExtrapStrain_RIT-eBBH-1911-n100.h5``

        Args:
            sim_name (str): RIT simulation name tag, e.g.
                ``"RIT:BBH:0001-n100-id3"``.
            use_cache (bool or None): Use cached file if present.  If
                ``None``, falls back to the instance-level ``self.use_cache``.
                Defaults to True.

        Returns:
            bool: True if the file is available locally (either from cache or
            after a successful download), False if the URL was not found.
        """
        if use_cache is None:
            use_cache = self.use_cache
        file_name = self.waveform_filename_from_simname(sim_name)
        file_path_web = self.waveform_data_url + "/" + file_name
        local_file_path = self.waveform_data_dir / file_name
        if (
            use_cache
            and os.path.exists(local_file_path)
            and os.path.getsize(local_file_path) > 0
        ):
            if self.verbosity > 2:
                logger.info("...can read from cache: {}".format(str(local_file_path)))
            return True
        else:
            if self.verbosity > 2:
                logger.info("...writing to cache: {}".format(str(local_file_path)))
            if utils.url_exists(file_path_web):
                if self.verbosity > 2:
                    logger.info("...downloading {}".format(file_path_web))
                subprocess.call(
                    [
                        "wget",
                        "--no-check-certificate",
                        str(file_path_web),
                        "-O",
                        str(local_file_path),
                    ]
                )
                return True
            else:
                if self.verbosity > 2:
                    logger.info(
                        "... ... but couldnt find link: {}".format(str(file_path_web))
                    )
                return False

    def fetch_waveform_data_from_cache(self, idx: int) -> object:
        """Not yet implemented.

        Args:
            idx (int): Four-digit simulation index.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()

    def download_data_for_catalog(
        self,
        num_sims_to_crawl: int = 2000,
        which_data: str = "waveform",
        possible_res: list = [],
        max_id_in_name: int = -1,
        use_cache: bool = True,
    ) -> dict:
        """Download waveform or psi4 data for all simulations in the catalog.

        Crawls the RIT web directory for waveform or psi4 data files and
        downloads each one.  Refreshes the on-disk metadata DataFrame first
        if it is out of date.

        Args:
            num_sims_to_crawl (int): Maximum number of simulations to process.
                Defaults to 2000.
            which_data (str): ``"waveform"`` or ``"psi4"``. Defaults to
                ``"waveform"``.
            possible_res (list): Resolution values to try. Defaults to the
                list in ``utils.rit_catalog_info``.
            max_id_in_name (int): Maximum ID suffix to search. Defaults to
                ``-1`` (uses the value in ``utils.rit_catalog_info``).
            use_cache (bool): Skip download if a non-empty file exists locally.
                Defaults to True.

        Returns:
            dict[str, pathlib.Path]: Mapping from simulation name to the
            local file path for each successfully downloaded file.
        """
        if len(possible_res) == 0:
            possible_res = self.possible_res
        if max_id_in_name <= 0:
            max_id_in_name = self.max_id_val
        if use_cache is None:
            use_cache = self.use_cache

        try:
            x = os.popen("/bin/ls {}/*.txt | wc -l".format(str(self.metadata_dir)))
            num_metadata_txt_files = int(x.read().strip())
            x = os.popen(
                "/bin/cat {}/metadata.csv | wc -l".format(str(self.metadata_dir))
            )
            num_metadata_df = int(x.read().strip())
        except Exception:
            # dummy values to force refresh below
            num_metadata_txt_files, num_metadata_df = 10, 0

        if num_metadata_df - 1 < num_metadata_txt_files:
            metadata = self.refresh_metadata_df_on_disk()
        else:
            metadata = self.read_metadata_df_from_disk()
        sims = {}

        if which_data == "waveform":
            filename_from_simname = self.waveform_filename_from_simname
            download_data = self.download_waveform_data
            data_dir = self.waveform_data_dir
        elif which_data == "psi4":
            filename_from_simname = self.psi4_filename_from_simname
            download_data = self.download_psi4_data
            data_dir = self.psi4_data_dir

        for idx, sim_name in tqdm(enumerate(metadata["simulation_name"])):
            if idx + 1 > num_sims_to_crawl:
                break
            file_name = filename_from_simname(sim_name)
            local_file_path = data_dir / file_name
            rv = download_data(sim_name, use_cache=use_cache)
            if rv:
                sims[sim_name] = local_file_path

        return sims
