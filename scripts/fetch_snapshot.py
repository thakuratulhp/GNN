"""Fetch the Li/Na/Mg Materials Project snapshot. Key comes from MP_API_KEY env var."""
import logging, sys
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from gnnsse.data.mp_fetch import MPFetcher

f = MPFetcher(snapshot_dir="data/mp_snapshots")
path = f.fetch(working_ions=["Li", "Na", "Mg"])
print("SNAPSHOT_WRITTEN:", path)
