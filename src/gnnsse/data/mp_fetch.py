"""Materials Project bulk downloader with incremental re-fetch and retry logic."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "bulk_modulus",
    "shear_modulus",
    "elements",
    "symmetry",
    "is_stable",
    "last_updated",
]


class MPFetcher:
    """Download and cache Materials Project entries for target working ions.

    Parameters
    ----------
    api_key:
        MP API key; falls back to ``MP_API_KEY`` env-var.
    snapshot_dir:
        Directory where Parquet snapshots and metadata are written.
    batch_size:
        Number of documents per API page.
    max_retries:
        Number of exponential-backoff retries on transient errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        snapshot_dir: str | Path = "data/mp_snapshots",
        batch_size: int = 1000,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or os.environ["MP_API_KEY"]
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        working_ions: list[str] | None = None,
        incremental: bool = False,
    ) -> Path:
        """Download entries and write a date-stamped Parquet snapshot.

        Parameters
        ----------
        working_ions:
            Subset of elements to require in the formula (e.g. ["Li", "Na", "Mg"]).
        incremental:
            If True, only fetch records updated since the last snapshot.

        Returns
        -------
        Path
            Path to the written Parquet file.
        """
        if working_ions is None:
            working_ions = ["Li", "Na", "Mg"]

        since = self._last_snapshot_date() if incremental else None
        if incremental and since:
            logger.info("Incremental fetch: only records updated after %s", since)

        records: list[dict[str, Any]] = []
        for ion in working_ions:
            logger.info("Fetching MP entries containing '%s'…", ion)
            batch = self._fetch_ion(ion, since=since)
            logger.info("  → %d entries", len(batch))
            records.extend(batch)

        # Deduplicate by material_id (ions may overlap, e.g. LiNa compounds)
        df = pd.DataFrame(records).drop_duplicates(subset="material_id")
        logger.info("Total unique entries: %d", len(df))

        out_path = self._write_snapshot(df)
        self._write_meta(out_path, working_ions, len(df))
        return out_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_ion(
        self,
        ion: str,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all MP documents that contain *ion* in their formula."""
        entries = []
        for attempt in range(self.max_retries):
            try:
                with MPRester(self.api_key) as mpr:
                    docs = mpr.materials.summary.search(
                        elements=[ion],
                        fields=_REQUIRED_FIELDS,
                        chunk_size=self.batch_size,
                    )
                for doc in docs:
                    entry = {
                        "material_id": doc.material_id,
                        "formula": doc.formula_pretty,
                        "band_gap": doc.band_gap,
                        "formation_energy_per_atom": doc.formation_energy_per_atom,
                        "energy_above_hull": doc.energy_above_hull,
                        "bulk_modulus": getattr(doc.bulk_modulus, "vrh", None)
                        if doc.bulk_modulus
                        else None,
                        "shear_modulus": getattr(doc.shear_modulus, "vrh", None)
                        if doc.shear_modulus
                        else None,
                        "is_stable": doc.is_stable,
                        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
                        "crystal_system": str(doc.symmetry.crystal_system.value) if doc.symmetry and doc.symmetry.crystal_system else None,
                        "structure_json": doc.structure.to_json() if doc.structure else None,
                        "working_ion": ion,
                        "last_updated": str(doc.last_updated) if hasattr(doc, "last_updated") else None,
                    }
                    entries.append(entry)
                return entries
            except Exception as exc:
                wait = 2**attempt
                logger.warning("Attempt %d failed (%s). Retrying in %ds…", attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch ion '{ion}' after {self.max_retries} attempts")

    def _write_snapshot(self, df: pd.DataFrame) -> Path:
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.snapshot_dir / f"mp_snapshot_{date_tag}.parquet"
        df.to_parquet(path, index=False)
        # Also write a symlink "latest"
        latest = self.snapshot_dir / "mp_snapshot_latest.parquet"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
        logger.info("Snapshot written → %s", path)
        return path

    def _write_meta(self, path: Path, working_ions: list[str], n_entries: int) -> None:
        meta = {
            "snapshot_path": str(path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "working_ions": working_ions,
            "n_entries": n_entries,
            "checksum_sha256": _sha256(path),
        }
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))

    def _last_snapshot_date(self) -> datetime | None:
        metas = sorted(self.snapshot_dir.glob("*.json"))
        if not metas:
            return None
        last_meta = json.loads(metas[-1].read_text())
        return datetime.fromisoformat(last_meta["timestamp_utc"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
