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
]


def _find_tree(file):
    """
    Detect the correct TTree inside the ROOT file.

    Logic:
    1. If 'mini' exists, use it.
    2. Otherwise, search for exactly one TTree.
    3. Otherwise, search for a TTree inside subdirectories.
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
            subkeys = object.keys()
            for subkey in subkeys:
                if file[f"{key}/{subkey}"].classname == "TTree":
                    return file[f"{key}/{subkey}"]
        except Exception:
            continue

    raise RuntimeError(f"Could not find a TTree in file {file.file_path}. "
                       f"Available keys: {file.keys()}")


def load_events(filename, branches=None):
    """
    Load selected branches from a ROOT file into an Awkward Array.
    Automatically detects the correct TTree name.

    Parameters
    ----------
    filename : str
        Path to the ROOT file.
    branches : list of str or None
        Branches to read.

    Returns
    -------
    ak.Array
        Awkward array with the requested branches.
    """
    if branches is None:
        branches = DEFAULT_BRANCHES

    f = uproot.open(filename)
    tree = _find_tree(f)

    return tree.arrays(branches, library="ak")
