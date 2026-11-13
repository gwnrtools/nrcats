"""Dynamic threshold-based classification and organization of numerical relativity catalogs.

Exposes the ``NRCatalogClassifier`` class which categorises simulations in
SXS, RIT, and MAYA catalogs into six mutually exclusive categories based on spin and
eccentricity thresholds, and supports filtering by NRSur7dq4 training simulations.
"""

from __future__ import annotations

import os
import numpy as np


class NRCatalogClassifier:
    """Classifies simulations in SXS, RIT, and MAYA catalogs based on spin and eccentricity thresholds.

    Provides methods to query simulations belonging to different physical categories dynamically.
    """

    CATEGORY_MAPPING = {
        "a": "non-spinning eccentric",
        "b": "non-spinning non-eccentric",
        "c": "aligned-spin eccentric",
        "d": "aligned-spin non-eccentric",
        "e": "precessing-spin eccentric",
        "f": "precessing-spin non-eccentric",
    }

    def __init__(
        self,
        spin_threshold: float = 0.001,
        ecc_threshold: float | dict[str, float] = 0.005,
    ):
        """Initialize the classifier with threshold limits.

        Parameters
        ----------
        spin_threshold : float, optional
            Threshold below which spin components are treated as 0 (default 0.001).
        ecc_threshold : float or dict, optional
            Threshold below which the reference eccentricity is treated as 0
            (default 0.005).

            A **mapping** ``{catalog: threshold}`` applies a different value to
            each catalog, which is the physically correct thing to do and the
            reason this accepts one.  The three catalogs do not publish the same
            quantity: SXS reports ``reference_eccentricity``, RIT and MAYA report
            an ``eccentricity`` measured by their own procedures at their own
            epochs, and one number applied to all three does not select the same
            population from each.  Catalog names are matched case-insensitively.

            The scalar form is retained and remains the default: it is the
            documented API, other callers depend on it, and a uniform threshold
            is still the right choice when the question is "what does the
            conventional cut give?" rather than "which simulations are
            measurably eccentric?".

        Raises
        ------
        ValueError
            If a mapping is given and a catalog being classified is absent from
            it.  Falling back to a default would silently apply one catalog's
            threshold to another, which is the error this parameter exists to
            prevent.
        """
        self.spin_threshold = spin_threshold
        if isinstance(ecc_threshold, dict):
            self.ecc_threshold = {k.upper(): float(v) for k, v in ecc_threshold.items()}
        else:
            self.ecc_threshold = float(ecc_threshold)
        self._sxs_catalog = None
        self._rit_catalog = None
        self._maya_catalog = None
        self._nrsur_sims = None
        self._classifications = {"SXS": {}, "RIT": {}, "MAYA": {}}

    def ecc_threshold_for(self, catalog_name: str) -> float:
        """The eccentricity threshold applied to *catalog_name*.

        Returns the scalar unchanged when a scalar was configured, so the
        uniform case costs nothing and reads the same.
        """
        if not isinstance(self.ecc_threshold, dict):
            return self.ecc_threshold
        tag = catalog_name.upper()
        try:
            return self.ecc_threshold[tag]
        except KeyError:
            raise ValueError(
                f"no eccentricity threshold for catalog '{catalog_name}': the "
                f"classifier was given a per-catalog mapping covering "
                f"{sorted(self.ecc_threshold)}. Add an entry for '{tag}' or pass "
                f"a scalar. Defaulting here would apply one catalog's threshold "
                f"to another, which is what the mapping exists to prevent."
            ) from None

    def load_catalog(self, catalog_name: str):
        """Lazy load a catalog by name tag.

        Parameters
        ----------
        catalog_name : str
            One of 'SXS', 'RIT', 'MAYA' (case-insensitive).

        Returns
        -------
        CatalogBase subclass
        """
        tag = catalog_name.upper()
        if tag == "SXS":
            if self._sxs_catalog is None:
                import nrcats as nrcat

                self._sxs_catalog = nrcat.SXSCatalog.load(download=False)
            return self._sxs_catalog
        elif tag == "RIT":
            if self._rit_catalog is None:
                import nrcats as nrcat

                self._rit_catalog = nrcat.RITCatalog.load()
            return self._rit_catalog
        elif tag == "MAYA":
            if self._maya_catalog is None:
                import nrcats as nrcat

                self._maya_catalog = nrcat.MayaCatalog.load()
            return self._maya_catalog
        else:
            raise ValueError(
                f"Unknown catalog '{catalog_name}'. Supported: 'SXS', 'RIT', 'MAYA'."
            )

    def load_nrsur_calibration_sims(self) -> set[str]:
        """Load and return the set of SXS simulations used to train/calibrate NRSur7dq4.

        Extracts the calibration set from `catalog_organization/sxs_classification.json`.

        Returns
        -------
        set[str]
            Simulation name tags (e.g. 'SXS:BBH:0001').
        """
        if self._nrsur_sims is not None:
            return self._nrsur_sims

        import nrcats
        import json

        base_dir = os.path.dirname(os.path.dirname(nrcats.__file__))
        path = os.path.join(base_dir, "catalog_organization", "sxs_classification.json")

        sims = set()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for category_data in data.values():
                    if (
                        isinstance(category_data, dict)
                        and "simulations" in category_data
                    ):
                        for sim_entry in category_data["simulations"]:
                            if isinstance(sim_entry, dict) and sim_entry.get(
                                "nrsur7dq4_calibration"
                            ):
                                sims.add(sim_entry["id"])
            except Exception:
                pass

        self._nrsur_sims = sims
        return self._nrsur_sims

    def classify_simulation(self, catalog_name: str, sim_name: str) -> str:
        """Classify a single simulation into one of the six categories.

        Parameters
        ----------
        catalog_name : str
            One of 'SXS', 'RIT', 'MAYA'.
        sim_name : str
            Name tag of the simulation.

        Returns
        -------
        str
            Category name.
        """
        tag = catalog_name.upper()
        catalog = self.load_catalog(tag)
        meta = catalog.get_metadata(sim_name)

        # 1. Eccentricity extraction and cleaning
        ecc = 0.0
        if tag == "SXS":
            ecc = meta.get("reference_eccentricity", 0.0)
        else:  # RIT and MAYA
            ecc = meta.get("eccentricity", 0.0)

        try:
            if isinstance(ecc, str):
                ecc = float(ecc.replace("<", "").replace("~", ""))
            elif ecc is None or np.isnan(ecc):
                ecc = 0.0
            else:
                ecc = float(ecc)
        except Exception:
            ecc = 0.0

        # 2. Spin extraction and cleaning
        s1x, s1y, s1z, s2x, s2y, s2z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        def clean_val(v):
            if v is None or np.isnan(v):
                return 0.0
            return float(v)

        if tag == "SXS":
            spin1 = meta.get("reference_dimensionless_spin1", [0.0, 0.0, 0.0])
            spin2 = meta.get("reference_dimensionless_spin2", [0.0, 0.0, 0.0])
            # Handle case where spin1/spin2 is None
            if spin1 is not None and len(spin1) >= 3:
                s1x, s1y, s1z = (
                    clean_val(spin1[0]),
                    clean_val(spin1[1]),
                    clean_val(spin1[2]),
                )
            if spin2 is not None and len(spin2) >= 3:
                s2x, s2y, s2z = (
                    clean_val(spin2[0]),
                    clean_val(spin2[1]),
                    clean_val(spin2[2]),
                )
        elif tag == "RIT":
            s1x = clean_val(
                meta.get("relaxed-chi1x", meta.get("initial-bh-chi1x", 0.0))
            )
            s1y = clean_val(
                meta.get("relaxed-chi1y", meta.get("initial-bh-chi1y", 0.0))
            )
            s1z = clean_val(
                meta.get("relaxed-chi1z", meta.get("initial-bh-chi1z", 0.0))
            )
            s2x = clean_val(
                meta.get("relaxed-chi2x", meta.get("initial-bh-chi2x", 0.0))
            )
            s2y = clean_val(
                meta.get("relaxed-chi2y", meta.get("initial-bh-chi2y", 0.0))
            )
            s2z = clean_val(
                meta.get("relaxed-chi2z", meta.get("initial-bh-chi2z", 0.0))
            )
        elif tag == "MAYA":
            s1x = clean_val(meta.get("a1x", 0.0))
            s1y = clean_val(meta.get("a1y", 0.0))
            s1z = clean_val(meta.get("a1z", 0.0))
            s2x = clean_val(meta.get("a2x", 0.0))
            s2y = clean_val(meta.get("a2y", 0.0))
            s2z = clean_val(meta.get("a2z", 0.0))

        # 3. Apply thresholds
        s1x_approx = 0.0 if abs(s1x) < self.spin_threshold else s1x
        s1y_approx = 0.0 if abs(s1y) < self.spin_threshold else s1y
        s1z_approx = 0.0 if abs(s1z) < self.spin_threshold else s1z
        s2x_approx = 0.0 if abs(s2x) < self.spin_threshold else s2x
        s2y_approx = 0.0 if abs(s2y) < self.spin_threshold else s2y
        s2z_approx = 0.0 if abs(s2z) < self.spin_threshold else s2z

        ecc_approx = 0.0 if abs(ecc) < self.ecc_threshold_for(tag) else ecc

        is_ecc = ecc_approx > 0.0

        # Determine classification category
        has_spin = (
            s1x_approx != 0.0
            or s1y_approx != 0.0
            or s1z_approx != 0.0
            or s2x_approx != 0.0
            or s2y_approx != 0.0
            or s2z_approx != 0.0
        )

        if not has_spin:
            return "non-spinning eccentric" if is_ecc else "non-spinning non-eccentric"

        is_precessing = (
            s1x_approx != 0.0
            or s1y_approx != 0.0
            or s2x_approx != 0.0
            or s2y_approx != 0.0
        )

        if is_precessing:
            return (
                "precessing-spin eccentric"
                if is_ecc
                else "precessing-spin non-eccentric"
            )
        else:
            return "aligned-spin eccentric" if is_ecc else "aligned-spin non-eccentric"

    def classify_all(self, catalog_name: str):
        """Precompute and cache classifications for all simulations in a catalog.

        Parameters
        ----------
        catalog_name : str
            One of 'SXS', 'RIT', 'MAYA'.
        """
        tag = catalog_name.upper()
        if self._classifications[tag]:
            return

        catalog = self.load_catalog(tag)
        categories = {cat_name: [] for cat_name in self.CATEGORY_MAPPING.values()}

        for sim in catalog.simulations_list:
            try:
                cat = self.classify_simulation(tag, sim)
                categories[cat].append(sim)
            except Exception:
                # If parsing fails, skip that simulation
                pass

        self._classifications[tag] = categories

    def get_simulations(
        self, catalog_name: str, category: str, only_nrsur_calibration: bool = False
    ) -> list[str]:
        """Get the list of simulation name tags under a given catalog and category.

        Parameters
        ----------
        catalog_name : str
            One of 'SXS', 'RIT', 'MAYA' (case-insensitive).
        category : str
            One of 'a', 'b', 'c', 'd', 'e', 'f' or their full name values.
        only_nrsur_calibration : bool, optional
            If True, only returns simulations used to calibrate NRSur7dq4 (SXS only).

        Returns
        -------
        list[str]
            Simulation name tags matching the category.
        """
        tag = catalog_name.upper()

        if only_nrsur_calibration and tag != "SXS":
            raise ValueError(
                "only_nrsur_calibration=True is only supported for the 'SXS' catalog."
            )

        self.classify_all(tag)

        category_lower = category.lower()
        if category_lower in self.CATEGORY_MAPPING:
            resolved_category = self.CATEGORY_MAPPING[category_lower]
        elif category_lower in self.CATEGORY_MAPPING.values():
            resolved_category = category_lower
        else:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Supported keys: {list(self.CATEGORY_MAPPING.keys())} "
                f"or values: {list(self.CATEGORY_MAPPING.values())}."
            )

        sims = self._classifications[tag][resolved_category]

        if only_nrsur_calibration:
            calibration_sims = self.load_nrsur_calibration_sims()
            return [sim for sim in sims if sim in calibration_sims]

        return list(sims)
