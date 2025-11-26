"""
Selection logic for the H -> ZZ* -> 4l analysis.

This module defines event and object selection functions, such as
basic kinematic cuts and four-lepton candidate selection.
"""

import awkward as ak


def basic_lepton_selection(arrays, pt_min=5.0, eta_max=2.5):
    """
    Basic lepton-level cuts.
    Mask is jagged, so we apply it branch-by-branch.
    """

    # Extract branches
    pt     = arrays["lep_pt"]
    eta    = arrays["lep_eta"]
    phi    = arrays["lep_phi"]
    E      = arrays["lep_E"]
    charge = arrays["lep_charge"]
    ltype  = arrays["lep_type"]

    # Thresholds:
    pt_cut = pt_min * 1000.0   # GeV → MeV

    # Per-lepton jagged mask
    mask = (pt > pt_cut) & (abs(eta) < eta_max)

    # Rewrite masked branches
    arrays = ak.with_field(arrays, pt[mask],     "lep_pt")
    arrays = ak.with_field(arrays, eta[mask],    "lep_eta")
    arrays = ak.with_field(arrays, phi[mask],    "lep_phi")
    arrays = ak.with_field(arrays, E[mask],      "lep_E")
    arrays = ak.with_field(arrays, charge[mask], "lep_charge")
    arrays = ak.with_field(arrays, ltype[mask],  "lep_type")

    return arrays


def four_lepton_event_selection(arrays):
    """
    Keep only events with at least 4 leptons.
    """
    n_leps = ak.num(arrays["lep_pt"], axis=1)
    event_mask = (n_leps >= 4)
    return arrays[event_mask]
