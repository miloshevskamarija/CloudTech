"""
I/O utilities for reading ATLAS Open Data ROOT files with uproot
"""

import uproot
import awkward as ak


DEFAULT_BRANCHES = [
    "lep_pt",
    "lep_eta",
    "lep_phi",
    "lep_E",
    "lep_charge",
    "lep_type",
    "jet_n",
    "runNumber",
    "eventNumber",
]


def _find_tree(file):
    """
    Detect the correct TTree inside the ROOT file.

    Logic:
    1. If 'mini' exists, use it.
    2. Otherwise, search for exactly one TTree.
    3. Otherwise, search for a TTree inside subkeydirectories.
    """
    # Direct match
    if "mini" in file.keys():
        return file["mini"]

    # Match with ';1' versioning
    if "mini;1" in file.keys():
        return file["mini;1"]

    # If there is exactly one TTree in the root file:
    tt_keys = [k for k, v in file.classnames().items() if v == "TTree"]
    if len(tt_keys) == 1:
        return file[tt_keys[0]]

    # Search inside directories
    for key in file.keys():
        try:
            object = file[key]
            for subkey in object.keys():
                full = f"{key}/{subkey}"
                if file[full].classname == "TTree":
                    return file[full]
        except Exception:
            continue

    raise RuntimeError(f"No TTree found in file {file.file_path}")


def load_events(filename, branches=None):
    """
    Load selected branches into an Awkward Array.
    Automatically detects the correct TTree name.
    """
    if branches is None:
        branches = DEFAULT_BRANCHES

    with uproot.open(filename) as f:
        tree = _find_tree(f)
        arrays = tree.arrays(branches, library="ak")

    return arrays
