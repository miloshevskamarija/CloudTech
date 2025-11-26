"""
Main entry point for the H->ZZ->4l analysis.

Reads ATLAS Open Data mini-ntuples, applies a simple lepton
selection, reconstructs a four-lepton candidate in each event,
and fills an m4l histogram.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Per-file 4-lepton analysis.

    Steps:
      1. Load branches needed for lepton kinematics.
      2. Apply basic lepton cuts: pT and |eta|.
      3. Require >= 4 leptons in the event.
      4. For each event, sort leptons by pT and take the 4 leading.
      5. Sum their four-vectors to form a Higgs candidate.
      6. Fill an m4l histogram (in GeV).
    """

    # 1) Load events
    arrays = load_events(filename)

    # 2) Basic lepton selection
    arrays = basic_lepton_selection(
        arrays,
        pt_min=config["selection"]["pt_min"],
        eta_max=config["selection"]["eta_max"],
    )

    # 3) Require >= 4 leptons
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

    # 4) Build lepton four-vectors (still in MeV)
    leptons = vector.Array(
        ak.zip(
            {
                "pt": arrays["lep_pt"],
                "eta": arrays["lep_eta"],
                "phi": arrays["lep_phi"],
                "E": arrays["lep_E"],
            }
        )
    )

    # 5) Sort leptons in each event by pT, descending, and keep leading 4
    idx = ak.argsort(leptons.pt, axis=1, ascending=False)
    leptons_sorted = leptons[idx]
    leptons4 = leptons_sorted[:, :4]

    # 6) Sum the four leading leptons to form Higgs candidates
    #    (Awkward reduction over the event axis)
    higgs = ak.sum(leptons4, axis=1)  # still MeV
    m4l = higgs.mass / 1000.0      # convert to GeV

    m4l_np = ak.to_numpy(m4l)

    if m4l_np.size > 0:
        h.fill(m4l_np)

    info = {"filename": filename, "n_events": len(arrays)}
    return h, info


def safe_process_file(fname, config):
    """
    Wrapper so that a bad file doesn't kill the whole job.
    """
    try:
        return process_file(fname, config)
    except Exception as e:
        print(f"[WARN] Error in file {fname}: {e}")
        return None

def main():
    args = parse_args()
    config = load_config(args.config)

    pattern = os.path.join(config["data_dir"], config["file_pattern"])
    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(f"No input files found for pattern {pattern}")

    print(f"Found {len(files)} input files.")

    use_parallel = config.get("parallel", False)
    n_workers = int(config.get("n_workers", os.cpu_count() or 2))

    results = []

    if use_parallel:
        print(f"Running in parallel with {n_workers} workers.")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_file = {
                pool.submit(safe_process_file, fname, config): fname
                for fname in files
            }
            n_total = len(future_to_file)
            for i, fut in enumerate(as_completed(future_to_file), start=1):
                fname = future_to_file[fut]
                try:
                    out = fut.result()
                    if out is not None:
                        results.append(out)
                        print(f"[{i}/{n_total}] Done {fname}")
                    else:
                        print(f"[{i}/{n_total}] Skipped {fname}")
                except Exception as e:
                    print(f"[ERROR] {fname} raised: {e}")
    else:
        for i, fname in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] Processing {fname} ...")
            out = safe_process_file(fname, config)
            if out is not None:
                results.append(out)
    if not results:
        raise RuntimeError("No successful per-file results; nothing to merge.")

    # Separate histograms and metadata
    hists, infos = zip(*results)

    # Keep only proper histograms
    hists = [h for h in hists if isinstance(h, Hist)]

    if not hists:
        raise RuntimeError("No histograms were produced!")

    # Merge histograms by adding them bin-by-bin
    total_hist = hists[0].copy()
    for h in hists[1:]:
        total_hist += h

    outdir = config["output_dir"]
    os.makedirs(outdir, exist_ok=True)

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

    # zoomed plot around Z and Higgs region
    fig, ax = plt.subplots()
    ax.step(edges[:-1], counts, where="post")
    ax.set_xlim(70, 180)
    ax.set_xlabel(r"$m_{4\ell}\,\mathrm{[GeV]}$")
    ax.set_ylabel("Events")
    ax.set_title("Four-lepton invariant mass (zoom)")
    ax.axvline(91.2, linestyle="--", alpha=0.7, label="Z mass")
    ax.axvline(125.0, linestyle="--", alpha=0.7, label="Higgs mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "m4l_zoom.png"))
    plt.close(fig)

    print(f"Processed {len(results)} files.")
    print(f"Saved histogram arrays and plots in: {outdir}")


if __name__ == "__main__":
    main()
