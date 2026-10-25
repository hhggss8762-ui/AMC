"""Checks for the new physical generators and intentionally unequal counts."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from amc.dataset20 import load_config20, expected_counts
from amc.synthesis20 import CLASSES20, constellation, generate_ofdm, generate_sample20, _new_digital

CONFIG = Path(__file__).resolve().parents[1]/"configs"/"amc20_v1.yaml"


def test_counts_and_32qam_geometry():
    c = load_config20(CONFIG)
    from amc.config import load_config
    assert load_config(CONFIG) == c
    assert sum(expected_counts(c,"train").values())*7 == 154000
    assert sum(expected_counts(c,"validation").values())*7 == 17500
    assert sum(expected_counts(c,"test").values())*7 == 35000
    for name, order in (("32QAM",32),("256QAM",256)):
        points = constellation(name)
        assert len(np.unique(points)) == order
        assert np.mean(np.abs(points)**2) == pytest.approx(1.)
    points = constellation("32QAM")
    assert not np.any((np.abs(points.real) == np.max(np.abs(points.real))) &
                      (np.abs(points.imag) == np.max(np.abs(points.imag))))


@pytest.mark.parametrize("group", ["generic","wlan_style","lte_style","multiscale","sparse_pilot","partial_band"])
def test_cyclic_prefix_copies_own_symbol_tail(group):
    c = load_config20(CONFIG)
    w = generate_ofdm(np.random.default_rng(400),group,c)
    n = w.parameters["source_fft_samples"]
    offset = 0
    for cp in w.parameters["cp_samples_per_symbol"]:
        block = w.iq[offset:offset+cp+n]
        np.testing.assert_array_equal(block[:cp],block[n:n+cp])
        offset += cp+n
    assert offset == len(w.iq)
    # Metadata must remain writable for the variable-CP branches, too.
    json.dumps(w.parameters)


def test_variable_cp_metadata_serializes():
    c = deepcopy(load_config20(CONFIG))
    c["synthesis"]["ofdm"]["groups"]["lte_style"]["cp_fractions"] = [.0703125]
    w = generate_ofdm(np.random.default_rng(42),"lte_style",c)
    assert w.parameters["variable_cyclic_prefix"] is True
    assert len(set(w.parameters["cp_samples_per_symbol"])) == 2
    json.dumps(w.parameters)


@pytest.mark.parametrize("name", CLASSES20)
def test_sample_contract_and_seed_reproducibility(name):
    c = load_config20(CONFIG)
    subtype = "sparse_pilot" if name == "OFDM" else c["dataset"]["subtypes"].get(name,[name])[0]
    seed = [20261025,20,99,CLASSES20.index(name),3,0]
    a,m,r = generate_sample20(name,subtype,15.,seed,c)
    b,n,_ = generate_sample20(name,subtype,15.,seed,c)
    np.testing.assert_array_equal(a,b)
    assert m == n
    json.dumps(m)
    assert a.shape == (2,2048) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.mean(np.sum(a.astype(float)**2,axis=0)) == pytest.approx(1,abs=2e-6)
    assert m["measured_inband_snr_db"] == pytest.approx(15,abs=1e-8)
    np.testing.assert_allclose(a,r["clean"]+r["noise"],atol=4e-7)


def test_msk_uses_half_index():
    c = load_config20(CONFIG)
    w = _new_digital("MSK","MSK",np.random.default_rng(4),c)
    phase_increments = np.angle(w.iq[1:]*w.iq[:-1].conj())
    np.testing.assert_allclose(np.abs(phase_increments),np.pi/(2*16),atol=2e-12)
    assert w.parameters["modulation_index"] == .5


def test_barker_dbpsk_spreading_and_overlap_disclosure():
    c = load_config20(CONFIG)
    _,m,_ = generate_sample20("DBPSK","DIFFERENTIAL_BARKER11",20,[123,20,90,1],c)
    assert m["source_parameters"]["spreading_factor"] == 11
    assert m["statistically_overlapping_label_group"] is None
    _,m,_ = generate_sample20("DQPSK","DQPSK",20,[123,20,90,2],c)
    assert m["statistically_overlapping_label_group"] == "QPSK_DQPSK_random_payload"
