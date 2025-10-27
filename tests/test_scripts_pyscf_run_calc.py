import os
import subprocess
from pathlib import Path

import pytest

from fanpy.scripts.pyscf import run_calc


def test_write_pyscf_py_creates_file(tmp_path, monkeypatch):
    # isolate make_pyscf_input to avoid heavy dependencies
    monkeypatch.setattr(run_calc, "make_pyscf_input", lambda coords, **kwargs: f"COORDS: {coords}")

    monkeypatch.chdir(tmp_path)
    coords = "H 0 0 0; H 0 0 0.74"
    run_calc.write_pyscf_py("inpdir", coords, basis="sto-3g", memory="1GB", charge=0, spin=0, units="B")

    out_file = tmp_path / "inpdir" / "calculate.py"
    assert out_file.exists()
    content = out_file.read_text()
    assert "COORDS" in content
    assert "H 0 0 0.74" in content

def test_write_pyscf_py_errors():
    # if memory not in MB or GB
    with pytest.raises(ValueError):
        run_calc.write_pyscf_py("inpdir", "H 0 0 0; H 0 0 0.74", basis="sto-3g", memory="1MQ")

def test_make_wfn_dirs_creates_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = "parent_dir"
    wfn_name = "mywfn"
    run_calc.make_wfn_dirs(parent, wfn_name, num_runs=3, rep_dirname_prefix="rep")

    for i in range(3):
        d = tmp_path / parent / wfn_name / f"rep_{i}"
        assert d.is_dir()

# tests for write wfn py 

def test_write_wfn_py_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = "parent_dir"
    wfn_name = "cisd"
    h2_geom = [['H', ['0.0000000000', '0.0000000000', '0.7500000000']], 
               ['H', ['0.0000000000', '0.0000000000', '0']]]
    basis = 'sto3g'
    # create parent directory to run calculation in 
    parent_dir = tmp_path / parent
    parent_dir.mkdir(parents=True)

    run_calc.write_wfn_py(parent, wfn_name, h2_geom, basis, objective='projected', solver='least_squares', pspace_exc=[1,2])

    out_file = parent_dir / "calculate.py"
    assert out_file.exists()
    content = out_file.read_text()
    assert "CISD" in content # this should be in generated file
    assert "sto3g" in content
    assert "ProjectedSchrodinger"