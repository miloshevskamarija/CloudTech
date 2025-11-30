import sys
import numpy as np
import pytest
ak = pytest.importorskip("awkward")
pytest.importorskip("hist")
pytest.importorskip("vector")
from hist import Hist
from src import run_analysis

def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = run_analysis.parse_args()
    assert args.config == "config/config.yaml"
    assert args.n_workers == 1

def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--config", "mycfg.yaml", "--n-workers", "4"],
    )
    args = run_analysis.parse_args()
    assert args.config == "mycfg.yaml"
    assert args.n_workers == 4

def test_process_file_returns_empty_hist_when_no_events(monkeypatch):
    empty = ak.Array([])

    def fake_load_events(_filename):
        return empty

    def fake_basic_lepton_selection(arrays, pt_min, eta_max):
        return arrays

    def fake_four_lepton_event_selection(arrays):
        return arrays

    monkeypatch.setattr(run_analysis, "load_events", fake_load_events)
    monkeypatch.setattr(
        run_analysis, "basic_lepton_selection", fake_basic_lepton_selection
    )
    monkeypatch.setattr(
        run_analysis, "four_lepton_event_selection", fake_four_lepton_event_selection
    )

    config = {
        "selection": {"pt_min": 5.0, "eta_max": 2.5},
        "hist": {"nbins": 10, "min": 100.0, "max": 160.0},
        "analysis": {"compute_extra_observables": True},
    }

    h, info = run_analysis.process_file("dummy.root", config)

    assert isinstance(h, Hist)
    # No events passed the selection
    assert info["n_events"] == 0
    assert info["leading_pt_GeV"].size == 0
    assert info["eta"].size == 0
    assert info["m12_GeV"].size == 0
    assert info["m34_GeV"].size == 0

def test_process_file_single_event_pipeline(monkeypatch):
    # One event with four leptons passing all selections
    arrays = ak.Array(
        {
            "lep_pt": [[6000.0, 7000.0, 8000.0, 9000.0]],
            "lep_eta": [[0.0, 0.1, -0.2, 0.3]],
            "lep_phi": [[0.0, 1.0, 2.0, 3.0]],
            "lep_E": [[10000.0, 11000.0, 12000.0, 13000.0]],
            "lep_charge": [[1, -1, 1, -1]],
            "lep_type": [[11, 11, 13, 13]],
        }
    )

    def fake_load_events(_filename):
        return arrays

    def identity_basic_selection(a, pt_min, eta_max):
        return a

    def identity_event_selection(a):
        return a

    monkeypatch.setattr(run_analysis, "load_events", fake_load_events)
    monkeypatch.setattr(
        run_analysis, "basic_lepton_selection", identity_basic_selection
    )
    monkeypatch.setattr(
        run_analysis, "four_lepton_event_selection", identity_event_selection
    )

    config = {
        "selection": {"pt_min": 5.0, "eta_max": 2.5},
        "hist": {"nbins": 20, "min": 50.0, "max": 200.0},
        "analysis": {"compute_extra_observables": True},
    }

    h, info = run_analysis.process_file("dummy.root", config)

    # We should have processed exactly one event
    assert info["n_events"] == 1

    # Leading lepton pT, in GeV, is just max(pt) / 1000
    assert np.allclose(info["leading_pt_GeV"], np.array([9.0]))

    # Histogram has a single entry corresponding to that event's m4ℓ
    assert isinstance(h, Hist)
    # hist.Hist stores values in a NumPy array accessible via .values()
    assert np.isclose(np.sum(h.values(flow=True)), 1.0)
