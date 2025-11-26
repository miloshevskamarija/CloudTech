"""
Main entry point for a simple H->ZZ->4l-style analysis on the ATLAS
Open Data mini-ntuples.

This script:
  * Reads all *.4lep.root files listed in config/config.yaml
  * Applies basic lepton kinematic cuts
  * Requires at least 4 leptons per event
  * Builds lepton four-vectors and reconstructs a 4-lepton invariant mass
    for each event (using the four highest-pT leptons)
  * Fills and saves a 1D histogram of m4l to outputs/m4l_counts.npy,
    outputs/m4l_edges.npy, and outputs/m4l.png
"""

import argparse
import glob
import os
import yaml
import awkward as ak
import numpy as np
import vector
import matplotlib.pyplot as plt
from hist import Hist
import hist

from src.analysis.io import load_events
from src.analysis.selection import (
    basic_lepton_selection,
    four_lepton_event_selection,
)


# Argument parsing and config loading
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


# Per-file analysis
def process_file(filename, config):
    """
    Vectorised H→ZZ→4ℓ-like analysis.
    Uses awkward broadcasting + vector Lorentz sums
    """

    # Load leptons
    arrays = load_events(filename)

    # Basic selection
    arrays = basic_lepton_selection(
        arrays,
        pt_min=config["selection"]["pt_min"],
        eta_max=config["selection"]["eta_max"],
    )

    # Require >= 4 leptons
    arrays = four_lepton_event_selection(arrays)

    # If no events pass, return an empty histogram
    nbins = config["hist"]["nbins"]
    hmin = config["hist"]["min"]
    hmax = config["hist"]["max"]

    m4l_axis = hist.axis.Regular(
        nbins, hmin, hmax, name="m4l", label=r"$m_{4\ell}\,[GeV]$"
    )
    h = Hist(m4l_axis)

    if len(arrays) == 0:
        return h, {"filename": filename, "n_events": 0}

    # Build lorentz vectors
    leptons = vector.Array(
        ak.zip(
            {
                "pt": arrays["lep_pt"],
                "eta": arrays["lep_eta"],
                "phi": arrays["lep_phi"],
                "E":   arrays["lep_E"],
            }
        )
    )

    # Sort leptons per event by pT descending
    idx = ak.argsort(leptons.pt, axis=1, ascending=False)

    # Reorder leptons
    leptons_sorted = leptons[idx]

    # Take the first 4 leptons per event
    top4 = leptons_sorted[:, :4]        # shape: (events, 4)

    # Vectorized 4-vector sum: sum over axis=1
    higgs = ak.sum(top4, axis=1)

    # Convert MeV → GeV
    m4l = higgs.mass / 1000.0

    # Fill histogram
    h.fill(m4l.to_numpy())

    return h, {"filename": filename, "n_events": len(arrays)}


def main():
    args = parse_args()
    config = load_config(args.config)

    # Discover files
    pattern = os.path.join(config["data_dir"], config["file_pattern"])
    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(f"No input files found for pattern {pattern}")

    print(f"Found {len(files)} input files.")

    results = []
    for i, fname in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing {fname} ...")
        h, info = process_file(fname, config)
        results.append((h, info))

    # Separate histograms and metadata
    hists, infos = zip(*results)

    # Filter out any histograms that are not Hist instances (just in case)
    hists = [h for h in hists if isinstance(h, Hist)]

    if not hists:
        raise RuntimeError("No histograms were produced!")

    # Merge histograms by adding them bin-by-bin
    total_hist = hists[0].copy()
    for h in hists[1:]:
        total_hist += h

    # Ensure output directory exists
    outdir = config["output_dir"]
    os.makedirs(outdir, exist_ok=True)

    # Save histogram contents: bin counts and edges
    counts = total_hist.values()
    edges = total_hist.axes[0].edges

    np.save(os.path.join(outdir, "m4l_counts.npy"), counts)
    np.save(os.path.join(outdir, "m4l_edges.npy"), edges)

    # Plot
    fig, ax = plt.subplots()
    ax.step(edges[:-1], counts, where="post")
    ax.set_xlabel(r"$m_{4\ell}\,\mathrm{[GeV]}$")
    ax.set_ylabel("Events")
    ax.set_title("Four-lepton invariant mass")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "m4l.png"))
    plt.close(fig)

    print(f"Processed {len(files)} files.")
    print(f"Saved histogram arrsays and plot to: {outdir}")


if __name__ == "__main__":
    main()
