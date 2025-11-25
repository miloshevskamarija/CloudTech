"""
Selection logic for the H -> ZZ* -> 4l analysis.

This module defines event and object selection functions, such as
basic kinematic cuts and four-lepton candidate selection.
"""

import awkward as ak


def basic_lepton_selection(arrays, pt_min=5.0, eta_max=2.5):
    """
    Apply basic lepton-level cuts on pT and |eta|.

    Parameters
    ----------
    arrays : ak.Array
        Awkward array containing lepton branches, e.g. lep_pt, lep_eta.
    pt_min : float
        Minimum transverse momentum [GeV].
    eta_max : float
        Maximum absolute pseudorapidity.

    Returns
    -------
    ak.Array
        A mask or filtered arrays reflecting the basic lepton selection.
    """
    pt = arrays["lep_pt"]
    eta = arrays["lep_eta"]

    mask = (pt > pt_min) & (abs(eta) < eta_max)
    # Return a filtered version of the full arrays
    return arrays[mask]


def four_lepton_event_selection(arrays):
    """
    Placeholder for the full 4-lepton event selection.

    In a full implementation this would:
    - require exactly four selected leptons,
    - check charge and flavour combinations for two Z candidates,
    - choose the best pairing (closest to m_Z).

    Parameters
    ----------
    arrays : ak.Array
        Awkward array after basic lepton cuts.

    Returns
    -------
    ak.Array
        Subset of events passing the 4-lepton selection.
    """
    # For now we simply require events with exactly four leptons left
    n_leptons = ak.num(arrays["lep_pt"], axis=1)
    event_mask = (n_leptons == 4)
    return arrays[event_mask]
