"""
Main entry point for the distributed H->ZZ->4l analysis.
"""

import argparse
import glob
import os
import yaml

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
    Per-file analysis function.

    This will be turned into a Dask task. It should:
    - load events,
    - apply selection,
    - build four-vectors,
    - compute m4l,
    - fill and return a partial histogram.
    """
    arrays = load_events(filename)
    arrays = basic_lepton_selection(
        arrays,
        pt_min=config["selection"]["pt_min"],
        eta_max=config["selection"]["eta_max"],
    )
    arrays = four_lepton_event_selection(arrays)

    # TODO: build four-vectors and compute m4l, fill hist (for now we just return a placeholder)
    return None, {"filename": filename, "n_events": len(arrays)}


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
