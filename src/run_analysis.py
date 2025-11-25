"""
Main entry point for the distributed H->ZZ->4l analysis.
"""

import argparse
import glob
import os
import yaml
from hist import Hist
import hist
import vector
import numpy as np
import awkward as ak

from dask import delayed, compute

from src.analysis.io import load_events
from src.analysis.physics import build_four_vector, invariant_mass
from src.analysis.selection import basic_lepton_selection, four_lepton_event_selection
from src.distributed.executor import create_local_client


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed ATLAS H->ZZ->4l analysis with Dask."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def process_file(filename, config):
    """
    Per-file H->ZZ->4l analysis.

    Steps:
    - Read ROOT file
    - Apply basic lepton selection
    - Select 4-lepton events
    - Build four-vectors
    - Compute m4l
    - Fill and return a partial histogram
    """

    # 1) Load events from the ROOT file
    arrays = load_events(filename)

    # 2) Basic lepton selection (pT and |eta|)
    arrays = basic_lepton_selection(
        arrays,
        pt_min=config["selection"]["pt_min"],
        eta_max=config["selection"]["eta_max"],
    )

    # 3) Require exactly four leptons per event (simple placeholder)
    arrays = four_lepton_event_selection(arrays)

    # If no events pass, return an empty histogram
    nbins = config["hist"]["nbins"]
    hmin = config["hist"]["min"]
    hmax = config["hist"]["max"]

    m4l_axis = hist.axis.Regular(
        nbins, hmin, hmax, name="m4l", label=r"$m_{4\ell}\,\mathrm{[GeV]}$"
    )
    h = Hist(m4l_axis)

    if len(arrays) == 0:
        return h, {"filename": filename, "n_events": 0}

    # 4) Build four-vectors using the vector library
    vecs = vector.Array(
    ak.zip(
        {
            "pt": arrays["lep_pt"],
            "eta": arrays["lep_eta"],
            "phi": arrays["lep_phi"],
            "E": arrays["lep_E"],
        }
    )
)

    # 5) Sum the four-vectors to get the Higgs candidate per event
    higgs_vec = ak.sum(vecs, axis=1)
    m4l = higgs_vec.mass

    # 6) Fill the histogram (weight=1 for now)
    h.fill(m4l)

    # Return the partial histogram and some metadata
    return h, {"filename": filename, "n_events": len(arrays)}


def main():
    args = parse_args()
    config = load_config(args.config)

    # Discover files
    pattern = os.path.join(config["data_dir"], config["file_pattern"])
    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(f"No input files found for pattern {pattern}")

    # Start local Dask client (later we can add remote/scheduler support)
    client = create_local_client(
        n_workers=config["dask"]["n_workers"],
        threads_per_worker=config["dask"]["threads_per_worker"],
    )

    # Build delayed tasks
    tasks = [delayed(process_file)(filename, config) for filename in files]

    results = compute(*tasks)

    client.close()

    # TODO: reduce histograms and write outputs
    print(f"Processed {len(results)} files.")


if __name__ == "__main__":
    main()
