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

    run_calc.write_wfn_py(parent, wfn_name, h2_geom, basis, objective='projected', solver='least_squares')

    out_file = parent_dir / "calculate.py"
    assert out_file.exists()
    content = out_file.read_text()
    assert "CISD" in content # this should be in generated file
    assert "sto3g" in content
    assert "ProjectedSchrodinger"

def test_write_wfn_py_kwargs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = "parent_dir"
    wfn_name = "cisd"
    h2_geom = [['H', ['0.0000000000', '0.0000000000', '0.7500000000']], 
               ['H', ['0.0000000000', '0.0000000000', '0']]]
    basis = 'sto3g'
    # create parent directory to run calculation in 
    parent_dir = tmp_path / parent
    parent_dir.mkdir(parents=True)
    run_calc.write_wfn_py(parent, wfn_name, h2_geom, basis, objective='projected', solver='least_squares', wfn_noise=1e-5, ham_noise=1e-6, pspace_exc=[1,2], memory='2gb')
    out_file = parent_dir / "calculate.py"
    assert out_file.exists()
    content = out_file.read_text()
    assert "wfn.params + 1e-05" in content
    assert "ham.params + 1e-06" in content
    subprocess.run(["python", str(out_file)], check=True)
    

def test_run_calcs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # create database folder structure. This is necessary for run_calcs to work
    db_dir = os.path.join(tmp_path, "database")
    os.makedirs(db_dir, exist_ok=True)
    basis = "sto-3g"
    basis_dir = os.path.join(db_dir, "H2", basis)
    os.makedirs(basis_dir, exist_ok=True)

    # create wfn dirs
    run_calc.make_wfn_dirs(basis_dir, "cisd", num_runs=3)
    wfn_dir = os.path.join(basis_dir, "cisd")
    
    # create a dummy calculate.py that just writes to output.txt
    calc_file =  os.path.join(wfn_dir, "calculate.py")
    with open(calc_file, 'w') as f:
        f.write("with open('output.txt', 'w') as f: f.write('Calculation complete')")
    
    # check if file exists
    run_calc.run_calcs(calc_file, calc_range=[0,2])
    output_file = os.path.join(wfn_dir, "_0", "output.txt")
    assert os.path.isfile(output_file)
    # check if content is correct
    with open(output_file, 'r') as f:
        content = f.read()
    assert content == "Calculation complete"

    # run calcs without calc_range:
    # create wfn dirs
    run_calc.make_wfn_dirs(basis_dir, "ccsd", num_runs=1)
    wfn_dir = os.path.join(basis_dir, "ccsd")
    calc_file =  os.path.join(wfn_dir, "calculate.py")
    with open(calc_file, 'w') as f:
        f.write("with open('output.txt', 'w') as f: f.write('Calculation complete')")
    run_calc.run_calcs(calc_file)
    output_file = os.path.join(wfn_dir, "_0", "output.txt")
    assert os.path.isfile(output_file)
    # check if content is correct
    with open(output_file, 'r') as f:
        content = f.read()
    assert content == "Calculation complete"

def test_run_calcs_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # create a dummy calculate.py that just writes to output.txt
    calc_file =  os.path.join(tmp_path, "calculate.py")
    with open(calc_file, 'w') as f:
        f.write("with open('output.txt', 'w') as f: f.write('Calculation complete')")
    
    # need to provide both time and memory
    with pytest.raises(ValueError) as excinfo:
        run_calc.run_calcs(calc_file, memory='2gb')
    assert str(excinfo.value) == "You cannot provide only one of the time and memory."

    with pytest.raises(ValueError) as excinfo:
        run_calc.run_calcs(calc_file, time='1h')
    assert str(excinfo.value) == "You cannot provide only one of the time and memory."

    # wrong time unit
    with pytest.raises(ValueError) as excinfo:
        run_calc.run_calcs(calc_file, time='1p', memory='2gb')
    assert str(excinfo.value) == "Time must be given in minutes, hours, or days (e.g. 1440m, 24h, 1d)."

    # wrong memory unit 
    with pytest.raises(ValueError) as excinfo:
        run_calc.run_calcs(calc_file, time='1h', memory='4KB')
    assert str(excinfo.value) == "Memory must be given as a MB or GB (e.g. 1024MB, 1GB)"