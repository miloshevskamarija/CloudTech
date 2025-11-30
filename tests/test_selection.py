import pytest
ak = pytest.importorskip("awkward")
from src.analysis import selection

def _make_lepton_arrays():
    # lep_pt etc are in MeV in the ATLAS mini-ntuples
    return ak.Array(
        {
            "lep_pt": [
                [4000.0, 6000.0], # 1 lepton above 5 GeV
                [4000.0], # all below 5 GeV
                [7000.0, 8000.0], # both above 5 GeV
            ],
            "lep_eta": [
                [0.1, 0.2],
                [0.0],
                [2.4, -2.4],
            ],
            "lep_phi": [
                [0.0, 1.0],
                [2.0],
                [3.0, 0.5],
            ],
            "lep_E": [
                [10.0, 20.0],
                [30.0],
                [40.0, 50.0],
            ],
            "lep_charge": [
                [1, -1],
                [-1],
                [1, -1],
            ],
            "lep_type": [
                [11, 13],
                [11],
                [13, 13],
            ],
        }
    )


def test_basic_lepton_selection_pt_and_eta_cuts():
    arrays = _make_lepton_arrays()

    selected = selection.basic_lepton_selection(arrays, pt_min=5.0, eta_max=2.5)

    # pt_min = 5 GeV → 5000 MeV, so 4000 fails, 6000/7000/8000 pass
    pts = ak.to_list(selected["lep_pt"])
    assert pts == [[6000.0], [], [7000.0, 8000.0]]

    # All other branches must be masked identically
    for field in ["lep_eta", "lep_phi", "lep_E", "lep_charge", "lep_type"]:
        values = ak.to_list(selected[field])
        assert len(values) == 3
        assert len(values[0]) == 1  # matches first event leptons
        assert len(values[1]) == 0  # second event empty
        assert len(values[2]) == 2  # third event has two leptons


def test_four_lepton_event_selection_keeps_only_rich_events():
    arrays = ak.Array(
        {
            "lep_pt": [
                [1, 2, 3, 4], # 4 leptons  -> keep
                [1, 2, 3], # 3 leptons  -> drop
                [], # 0 leptons  -> drop
                [1, 2, 3, 4, 5], # 5 leptons  -> keep
            ]
        }
    )

    selected = selection.four_lepton_event_selection(arrays)

    # Only events 0 and 3 survive (n_leps ≥ 4)
    counts = ak.to_list(ak.num(selected["lep_pt"], axis=1))
    assert counts == [4, 5]
