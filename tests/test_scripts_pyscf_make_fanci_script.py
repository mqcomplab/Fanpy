"""Test fanpy.script.gaussian.make_fanci_script."""
import subprocess
import pytest
import numpy as np

from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.scripts.pyscf.make_fanci_script import make_script

from utils import find_datafile


h2_geom = [['H', ['0.0000000000', '0.0000000000', '0.7500000000']], 
        ['H', ['0.0000000000', '0.0000000000', '0']]]

basis = "sto6g"

@pytest.mark.parametrize("wfn", [
    "cisd", 
    "fci",
    "doci",
    "ap1rog",
    "apig"
])
def test_make_script_wfns(tmp_path, wfn):
    """Test fanpy.scripts.utils.make_script with different wavefunctions.
    """
    script_path = str(tmp_path / "script.py")


    make_script(h2_geom, wfn, basis, filename=script_path)
    subprocess.check_output(["python", script_path])

    make_script(h2_geom, wfn, basis, filename=script_path, wfn_kwargs="")
    subprocess.check_output(["python", script_path])

@pytest.mark.parametrize("objective", ["projected"])
def test_make_script_objectives(tmp_path, objective):
    """Test fanpy.scripts.utils.make_script with different objectives."""
    script_path = str(tmp_path / "script.py")
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective=objective,
        solver="least_squares",
        filename=script_path,
    )
    subprocess.check_output(["python", script_path])


def test_make_script_fanpt(tmp_path):
    """Test fanpy.scripts.utils.make_script with fanpt."""
    script_path = str(tmp_path / "script.py")
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="fanpt",
        filename=script_path,
    )
    subprocess.check_output(["python", script_path])

    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="fanpt",
        filename=script_path,
        fanpt_kwargs="energy_active=False, final_order=4, steps=100"
    )
    subprocess.check_output(["python", script_path])

def test_make_script_projected_solvers(tmp_path):
    """Test fanpy.scripts.utils.make_script with projected objective and different solvers."""
    script_path = str(tmp_path / "script.py")
    # least squares
    make_script(
        h2_geom,
        "apig",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])

    # root default solver kwargs
    h4_geom = [
        ['H', ['0.0000000000', '0.0000000000', '1.5000000000']],
        ['H', ['0.0000000000', '1.5000000000', '0.0000000000']],
        ['H', ['1.5000000000', '0.0000000000', '0.0000000000']],
        ['H', ['0.0000000000', '0.0000000000', '0.0000000000']]]
    make_script(
        h4_geom, "cisd", basis, hf_units="B", wfn_kwargs = "spin=0", pspace_exc=[1, 2, 3, 4], objective="projected", solver="root", filename=script_path
    ) # spin needs to be 0 so that system is not underdetermined
    subprocess.check_output(["python", script_path])

    # root with empty solver kwargs
    make_script(
        h4_geom,
        "cisd",
        basis,
        hf_units="B",
        wfn_kwargs="spin=0",
        objective="projected",
        pspace_exc=[1, 2, 3, 4],
        solver="root",
        filename=script_path,
        solver_kwargs="",
    ) # spin needs to be 0 so that system is not underdetermined
    subprocess.check_output(["python", script_path])


def test_make_script_projected_settings(tmp_path):
    """Test fanpy.scripts.utils.make_script with projected objective and different noise, memory and orbital optimization."""
    script_path = str(tmp_path / "script.py")

    # limited memory
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        memory="2gb",
    )
    subprocess.check_output(["python", script_path])

    # wfn noise
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        wfn_noise=0.2,
    )
    subprocess.check_output(["python", script_path])

    # ham noise
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        ham_noise=0.2,
    )
    subprocess.check_output(["python", script_path])

    # pspace excitations
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        pspace_exc=[1, 2],
    )
    subprocess.check_output(["python", script_path])

    # orbital optimization; not supported for projected
    # make_script(
    #     2,
    #     oneint,
    #     twoint,
    #     "ap1rog",
    #     objective="projected",
    #     solver="least_squares",
    #     filename=script_path,
    #     optimize_orbs=True,
    # )
    # subprocess.check_output(["python", script_path])


def test_make_script_checkpoint(tmp_path):
    """Test that make_script works with checkpointing options."""

    script_path = str(tmp_path / "script.py")
    # minimize with checkpointing
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        save_chk=str(tmp_path / "checkpoint.npy"),
    )
    subprocess.check_output(["python", script_path])

    # minimize with loading from checkpoint
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        load_wfn=str(tmp_path / "checkpoint_AP1roG.npy"),
    )
    subprocess.check_output(["python", script_path])


def test_make_script_filename_option(tmp_path):
    """Test that make_script works the same when filename is given and not given."""

    script_path = str(tmp_path / "script.py")
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
    )
    script = make_script(
        h2_geom, "ap1rog", basis, objective="projected", solver="least_squares", filename=-1
    )
    with open(script_path, "r") as f:
        assert f.read() == script

# # test if errors are raised when they should be
def test_make_script_errors(tmp_path):
    """Test that make_script raises errors when it should."""
    script_path = str(tmp_path / "script.py")
    
    with pytest.raises(ValueError):
        make_script(
            h2_geom,
            "ap1rog",
            basis,
            objective="projected",
            solver="least_squares",
            filename=script_path,
            optimize_orbs=True,  # orbital optimization not supported currently
        )
    with pytest.raises(ValueError):
        make_script(
            h2_geom,
            "ap1rog",
            basis,
            objective="variational", # this is not supported at the moment
            solver="cma",
            filename=script_path,
        )

def test_load_ham(tmp_path):
    """Test that make_script works with load_ham option."""

    script_path = str(tmp_path / "script.py")
    ham_path = str(tmp_path / "ham.npy")
    oneint = find_datafile("data/data_h2_hf_sto6g_oneint.npy")
    twoint = find_datafile("data/data_h2_hf_sto6g_twoint.npy")
    oneint_loaded = np.load(oneint)
    twoint_loaded = np.load(twoint)
    ham = RestrictedMolecularHamiltonian(oneint_loaded, twoint_loaded)
    ham.save_params(ham_path)

    # load hamiltonian
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        load_ham=ham_path,
    )
    subprocess.check_output(["python", script_path])

def test_energy_constraint(tmp_path):
    """Test that make_script works with energy constraint."""
    script_path = str(tmp_path / "script.py")

    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="projected",
        solver="least_squares",
        filename=script_path,
        constraint="energy",
    )
    subprocess.check_output(["python", script_path])

def test_make_fanci_script_cli(tmp_path):
    script_path = str(tmp_path / "script.py")
    subprocess.check_output([
        "fanpy_make_fanci_pyscf_script",
        "--geom", '[["H",["0.0","0.0","0.75"]],["H",["0.0","0.0","0.0"]]]',
        "--wfn_type", "cisd",
        "--basis", basis,
        "--filename", script_path,
    ])
    subprocess.check_output(["python", script_path])