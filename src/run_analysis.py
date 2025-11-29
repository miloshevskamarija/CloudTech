"""
Main entry point for the H->ZZ->4l analysis.

Reads ATLAS Open Data mini-ntuples, applies a simple lepton
selection, reconstructs a four-lepton candidate in each event,
and fills physics histograms (m4l, Z candidates, lepton kinematics).

Supports both serial execution and local multi-process parallelism
via ProcessPoolExecutor.
"""

import argparse
import glob
import os
import time
import multiprocessing
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
        description="Parallel ATLAS H->ZZ->4l analysis using multiple worker processes."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel file processing.",
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
    h_m4l = Hist(m4l_axis)

    analysis_cfg = config.get("analysis", {})
    compute_extra = analysis_cfg.get("compute_extra_observables", True)

    # If no events remain after selection, return empty structures
    if len(arrays) == 0:
        info = {
            "filename": filename,
            "n_events": 0,
            "leading_pt_GeV": np.array([], dtype=float),
            "eta": np.array([], dtype=float),
            "m12_GeV": np.array([], dtype=float),
            "m34_GeV": np.array([], dtype=float),
            "dR_lep01": np.array([], dtype=float),
        }
        return h_m4l, info

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
    
    # handle files with zero 4-lepton events
    if len(leptons4) == 0:
        return h_m4l, {
        "filename": filename,
        "n_events": 0,
        "leading_pt_GeV": np.array([], dtype=float),
        "eta": np.array([], dtype=float),
        "m12_gev": np.array([], dtype=float),
        "m34_gev": np.array([], dtype=float),
        "dR_lep01": np.array([], dtype=float),
    }

    # 6) Sum the four leading leptons to form Higgs candidates
    higgs = ak.sum(leptons4, axis=1)  # still MeV
    m4l = higgs.mass / 1000.0      # convert to GeV

    m4l_np = ak.to_numpy(m4l)

    if m4l_np.size > 0:
        h_m4l.fill(m4l_np)

    # Extra observables (optional for performance runs)
    if compute_extra:
        # Leading lepton pT in GeV
        leading_pt_GeV = ak.to_numpy(leptons4[:, 0].pt / 1000.0)

        # All selected leptons' pseudorapidity
        eta_flat = ak.to_numpy(ak.flatten(leptons.eta))

        # Simple Z candidates: (0,1) and (2,3) of the leading four
        z1 = leptons4[:, 0] + leptons4[:, 1]
        z2 = leptons4[:, 2] + leptons4[:, 3]
        m12_GeV = ak.to_numpy(z1.mass / 1000.0)
        m34_GeV = ak.to_numpy(z2.mass / 1000.0)

        # ΔR between the two leading leptons
        dR_lep01 = ak.to_numpy(leptons4[:, 0].deltaR(leptons4[:, 1]))
    else:
        # Empty arrays when we are not interested in detailed observables
        leading_pt_GeV = np.array([], dtype=float)
        eta_flat = np.array([], dtype=float)
        m12_GeV = np.array([], dtype=float)
        m34_GeV = np.array([], dtype=float)
        dR_lep01 = np.array([], dtype=float)

    info = {
        "filename": filename,
        "n_events": len(arrays),
        "leading_pt_GeV": leading_pt_GeV,
        "eta": eta_flat,
        "m12_GeV": m12_GeV,
        "m34_GeV": m34_GeV,
        "dR_lep01": dR_lep01,
    }

    return h_m4l, info


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

    # Analysis / plotting switches from config
    analysis_cfg = config.get("analysis", {})
    compute_extra = analysis_cfg.get("compute_extra_observables", True)
    make_plots = analysis_cfg.get("make_plots", True)

    # Decide how many workers to use
    n_workers = config.get("n_workers", args.n_workers)
    max_procs = multiprocessing.cpu_count() or 1
    if n_workers > max_procs:
        print(
            f"[INFO] Requested {n_workers} workers but only {max_procs} cores available; "
            f"using {max_procs}."
        )
        n_workers = max_procs

    print(f"Using {n_workers} worker process(es).")

    start_time = time.perf_counter()

    results = []
    total_events = 0

    # Serial path for N=1: avoids multiprocessing overhead for the reference timing
    if n_workers == 1:
        for i, fname in enumerate(files, start=1):
            out = safe_process_file(fname, config)
            if out is not None:
                h, info = out
                results.append((h, info))
                total_events += info.get("n_events", 0)
            print(f"[{i}/{len(files)}] Completed {fname}")
    else:
        # Multi-process path
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_file = {
                pool.submit(safe_process_file, fname, config): fname
                for fname in files
            }
            for i, future in enumerate(as_completed(future_to_file), start=1):
                fname = future_to_file[future]
                try:
                    out = future.result()
                except Exception as e:
                    print(f"[ERROR] {fname}: {e}")
                    continue
                if out is not None:
                    h, info = out
                    results.append((h, info))
                    total_events += info.get("n_events", 0)
                print(f"[{i}/{len(files)}] Completed {fname}")

    end_time = time.perf_counter()
    wall_time = end_time - start_time

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

    def concat_from_infos(key):
        arrays = [info[key] for info in infos if info[key].size > 0]
        if arrays:
            return np.concatenate(arrays)
        else:
            return np.array([], dtype=float)

    # empty if compute_extra=False
    leading_pt_all = concat_from_infos("leading_pt_GeV") if compute_extra else np.array([], dtype=float)
    eta_all = concat_from_infos("eta") if compute_extra else np.array([], dtype=float)
    m12_all = concat_from_infos("m12_GeV") if compute_extra else np.array([], dtype=float)
    m34_all = concat_from_infos("m34_GeV") if compute_extra else np.array([], dtype=float)
    dR_all = concat_from_infos("dR_lep01") if compute_extra else np.array([], dtype=float)

    # m4l histogram and stats
    counts = total_hist.values()
    edges = total_hist.axes[0].edges

    # Bin centres and statistical (Poisson) errors
    centers = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts)

    total_entries = counts.sum()
    if total_entries > 0:
        mean_m4l = np.sum(centers * counts) / total_entries
        var_m4l = np.sum(((centers - mean_m4l) ** 2) * counts) / total_entries
        rms_m4l = np.sqrt(var_m4l)
    else:
        mean_m4l = float("nan")
        rms_m4l = float("nan")

    # Define simple windows around Z and H in GeV
    z_window = (80.0, 100.0)
    h_window = (115.0, 135.0)

    z_mask = (centers >= z_window[0]) & (centers <= z_window[1])
    h_mask = (centers >= h_window[0]) & (centers <= h_window[1])

    z_yield = counts[z_mask].sum()
    h_yield = counts[h_mask].sum()

    # Save m4l histogram data as numpy arrays (counts and bin edges)
    np.save(os.path.join(outdir, "m4l_counts.npy"), counts)
    np.save(os.path.join(outdir, "m4l_edges.npy"), edges)

    # Plotting (can be disabled via analysis.make_plots = false)
    if make_plots:
        # m4l with error bars
        fig, ax = plt.subplots()
        ax.step(edges[:-1], counts, where="post", label="Events")
        ax.errorbar(
            centers,
            counts,
            yerr=errors,
            fmt=".",
            markersize=2,
            linewidth=0.5,
            label="Statistical errors",
        )
        ax.set_xlabel(r"$m_{4\ell}\,\mathrm{[GeV]}$")
        ax.set_ylabel("Events")
        ax.set_title("Four-lepton invariant mass")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "m4l.png"))
        plt.close(fig)

        # zoomed plot around Z and Higgs region
        fig, ax = plt.subplots()
        ax.step(edges[:-1], counts, where="post", label="Events")
        ax.errorbar(
            centers,
            counts,
            yerr=errors,
            fmt=".",
            markersize=2,
            linewidth=0.5,
            label="Statistical errors",
        )
        ax.set_xlim(70, 180)
        ax.set_xlabel(r"$m_{4\ell}\,\mathrm{[GeV]}$")
        ax.set_ylabel("Events")
        ax.set_title("Four-lepton invariant mass (zoom)")

        mZ = 91.1876
        mH = 125.0
        ax.axvline(mZ, linestyle="--", label="Z boson mass")
        ax.axvline(mH, linestyle="--", label="Higgs boson mass")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "m4l_zoom.png"))
        plt.close(fig)

        # m4l with log-y scale
        fig, ax = plt.subplots()
        ax.step(edges[:-1], counts, where="post", label="Events")
        ax.errorbar(
            centers,
            counts,
            yerr=errors,
            fmt=".",
            markersize=2,
            linewidth=0.5,
            label="Statistical errors",
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{4\ell}\,\mathrm{[GeV]}$")
        ax.set_ylabel("Events")
        ax.set_title("Four-lepton invariant mass (log scale)")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "m4l_log.png"))
        plt.close(fig)

        # Leading lepton pT
        if leading_pt_all.size > 0:
            pt_bins = np.linspace(0.0, 200.0, 51)  # 0–200 GeV
            pt_counts, pt_edges = np.histogram(leading_pt_all, bins=pt_bins)
            pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
            pt_errors = np.sqrt(pt_counts)

            fig, ax = plt.subplots()
            ax.step(pt_edges[:-1], pt_counts, where="post", label="Events")
            ax.errorbar(
                pt_centers,
                pt_counts,
                yerr=pt_errors,
                fmt=".",
                markersize=2,
                linewidth=0.5,
                label="Statistical errors",
            )
            ax.set_xlabel(r"Leading lepton $p_T$ [GeV]")
            ax.set_ylabel("Events")
            ax.set_title("Leading lepton transverse momentum")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "leading_pt.png"))
            plt.close(fig)

        # Lepton eta distribution
        if eta_all.size > 0:
            eta_bins = np.linspace(-2.5, 2.5, 51)
            eta_counts, eta_edges = np.histogram(eta_all, bins=eta_bins)
            eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
            eta_errors = np.sqrt(eta_counts)

            fig, ax = plt.subplots()
            ax.step(eta_edges[:-1], eta_counts, where="post", label="Events")
            ax.errorbar(
                eta_centers,
                eta_counts,
                yerr=eta_errors,
                fmt=".",
                markersize=2,
                linewidth=0.5,
                label="Statistical errors",
            )
            ax.set_xlabel(r"Lepton $\eta$")
            ax.set_ylabel("Events")
            ax.set_title("Lepton pseudorapidity distribution")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "lepton_eta.png"))
            plt.close(fig)

        # m12 and m34 1D distributions + 2D correlation
        if m12_all.size > 0 and m34_all.size > 0:
            m_bins = np.linspace(50.0, 150.0, 51)

            m12_counts, m12_edges = np.histogram(m12_all, bins=m_bins)
            m34_counts, m34_edges = np.histogram(m34_all, bins=m_bins)

            m_centers = 0.5 * (m_bins[:-1] + m_bins[1:])
            m12_errors = np.sqrt(m12_counts)
            m34_errors = np.sqrt(m34_counts)

            # m12
            fig, ax = plt.subplots()
            ax.step(m12_edges[:-1], m12_counts, where="post", label="Events")
            ax.errorbar(
                m_centers,
                m12_counts,
                yerr=m12_errors,
                fmt=".",
                markersize=2,
                linewidth=0.5,
                label="Statistical errors",
            )
            ax.set_xlabel(r"$m_{12}$ [GeV]")
            ax.set_ylabel("Events")
            ax.set_title("Leading Z candidate mass $m_{12}$")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "m12.png"))
            plt.close(fig)

            # m34
            fig, ax = plt.subplots()
            ax.step(m34_edges[:-1], m34_counts, where="post", label="Events")
            ax.errorbar(
                m_centers,
                m34_counts,
                yerr=m34_errors,
                fmt=".",
                markersize=2,
                linewidth=0.5,
                label="Statistical errors",
            )
            ax.set_xlabel(r"$m_{34}$ [GeV]")
            ax.set_ylabel("Events")
            ax.set_title("Subleading Z candidate mass $m_{34}$")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "m34.png"))
            plt.close(fig)

            # 2D histogram m12 vs m34
            H2, xedges, yedges = np.histogram2d(
                m12_all,
                m34_all,
                bins=(m_bins, m_bins),
            )

            fig, ax = plt.subplots()
            X, Y = np.meshgrid(xedges, yedges)
            pcm = ax.pcolormesh(X, Y, H2.T)
            ax.set_xlabel(r"$m_{12}$ [GeV]")
            ax.set_ylabel(r"$m_{34}$ [GeV]")
            ax.set_title(r"$m_{12}$ vs $m_{34}$")
            fig.colorbar(pcm, ax=ax, label="Events")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "m12_m34_2D.png"))
            plt.close(fig)

        # ΔR between leading leptons
        if dR_all.size > 0:
            dR_bins = np.linspace(0.0, 5.0, 51)
            dR_counts, dR_edges = np.histogram(dR_all, bins=dR_bins)
            dR_centers = 0.5 * (dR_edges[:-1] + dR_edges[1:])
            dR_errors = np.sqrt(dR_counts)

            fig, ax = plt.subplots()
            ax.step(dR_edges[:-1], dR_counts, where="post", label="Events")
            ax.errorbar(
                dR_centers,
                dR_counts,
                yerr=dR_errors,
                fmt=".",
                markersize=2,
                linewidth=0.5,
                label="Statistical errors",
            )
            ax.set_xlabel(r"$\Delta R(\ell_0,\ell_1)$")
            ax.set_ylabel("Events")
            ax.set_title(r"Angular separation $\Delta R$ of leading leptons")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "deltaR_leading_leptons.png"))
            plt.close(fig)

    # Final summary
    print(f"Processed {len(results)} files.")
    print(f"Total selected events (after cuts): {total_events}")
    print(f"Total histogram entries: {int(total_entries)}")
    print(f"⟨m4ℓ⟩ = {mean_m4l:.2f} GeV, RMS = {rms_m4l:.2f} GeV")
    print(f"Yield in Z window {z_window}: {int(z_yield)} events")
    print(f"Yield in H window {h_window}: {int(h_yield)} events")
    print(f"Total wall time: {wall_time:.2f} s")
    if wall_time > 0:
        rate = total_events / wall_time
        print(f"Average processing rate: {rate:.1f} events/s")
    print(f"Saved outputs to {outdir}")


if __name__ == "__main__":
    main()
