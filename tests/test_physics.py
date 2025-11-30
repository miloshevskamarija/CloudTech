import numpy as np
import pytest
ak = pytest.importorskip("awkward")
from src.analysis import physics

def test_build_four_vector_numpy():
    pt = np.array([10.0])
    eta = np.array([0.0])
    phi = np.array([0.0])
    energy = np.array([10.0])

    four = physics.build_four_vector(pt, eta, phi, energy)

    # Check the standard cylindrical → Cartesian mapping:
    # px = pt cos φ, py = pt sin φ, pz = pt sinh η
    assert np.allclose(four["E"], energy)
    assert np.allclose(four["px"], pt * np.cos(phi))
    assert np.allclose(four["py"], pt * np.sin(phi))
    assert np.allclose(four["pz"], pt * np.sinh(eta))

def test_build_four_vector_awkward_structure_preserved():
    pt = ak.Array([[10.0, 20.0], [30.0]])
    eta = ak.Array([[0.1, -0.2], [0.0]])
    phi = ak.Array([[0.0, np.pi / 2], [np.pi]])
    energy = ak.Array([[50.0, 80.0], [40.0]])

    four = physics.build_four_vector(pt, eta, phi, energy)

    # Same jagged structure: 2 leptons in first event, 1 in second
    assert ak.to_list(ak.num(four["px"], axis=1)) == [2, 1]
    # Energies are passed through unchanged
    assert ak.all(four["E"] == energy)

def test_invariant_mass_scalar_and_precision_guard():
    # Simple timelike four-vector at rest: m = sqrt(E^2 - |p|^2) = E
    E = np.array([2.0])
    px = np.array([0.0])
    py = np.array([0.0])
    pz = np.array([0.0])

    m = physics.invariant_mass(E, px, py, pz)
    assert np.allclose(m, 2.0)

    # Unphysical vector with E^2 - |p|^2 < 0 should be clipped to m^2 = 0
    E2 = np.array([1.0])
    px2 = np.array([1.01])
    py2 = np.array([0.0])
    pz2 = np.array([0.0])

    m2 = physics.invariant_mass(E2, px2, py2, pz2)
    assert np.all(m2 >= 0.0)
    assert np.allclose(m2, 0.0)
