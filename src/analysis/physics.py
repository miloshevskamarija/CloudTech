"""
Physics utilities for four-lepton analyses.

This module provides basic four-vector operations and invariant mass
calculations using NumPy and Awkward Arrays.
"""

import numpy as np
import awkward as ak


def build_four_vector(pt, eta, phi, energy):
    """
    Construct four-vectors from (pt, eta, phi, E).

    Parameters
    ----------
    pt : array-like (Awkward or NumPy)
        Transverse momentum of the particles [GeV].
    eta : array-like
        Pseudorapidity of the particles.
    phi : array-like
        Azimuthal angle of the particles [radians].
    energy : array-like
        Energy of the particles [GeV].

    Returns
    -------
    dict of arrays
        A dictionary with components 'E', 'px', 'py', 'pz'.
        Each entry has the same jagged/array structure as the inputs.
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    return {
        "E": energy,
        "px": px,
        "py": py,
        "pz": pz,
    }


def invariant_mass(E, px, py, pz):
    """
    Compute invariant mass m = sqrt(E^2 - |p|^2) with c = 1.

    Parameters
    ----------
    E, px, py, pz : array-like
        Components of the four-vector(s). Can be Awkward Arrays, so
        this works event-by-event for many particles at once.

    Returns
    -------
    array-like
        Invariant mass values with the same structure as the inputs.
    """
    p2 = px**2 + py**2 + pz**2
    m2 = E**2 - p2
    # guard against small negative values from numerical precision
    m2 = ak.where(m2 < 0, 0, m2)
    return np.sqrt(m2)
