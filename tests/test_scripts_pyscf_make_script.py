"""Test fanpy.script.make_script."""
import subprocess
import pytest

from fanpy.scripts.pyscf.make_script import make_script

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

@pytest.mark.parametrize("objective", ["least_squares", "variational", "one_energy"])
def test_make_script_objectives(tmp_path, objective):
    """Test fanpy.scripts.utils.make_script with different objectives."""
    script_path = str(tmp_path / "script.py")
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective=objective,
        solver="minimize",
        filename=script_path,
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
        h4_geom, "cisd", basis, hf_units="B", pspace_exc=[1, 2, 3, 4], objective="projected", solver="root", filename=script_path
    )
    subprocess.check_output(["python", script_path])

    # root with empty solver kwargs
    make_script(
        h4_geom,
        "cisd",
        basis,
        hf_units="B",
        objective="projected",
        pspace_exc=[1, 2, 3, 4],
        solver="root",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])


def test_make_script_one_energy_trustregion(tmp_path):
    """Test fanpy.scripts.utils.make_script with one_energy objective and trustregion solver."""

    script_path = str(tmp_path / "script.py")
   
    make_script(
        h2_geom,
        "apig",
        basis,
        objective="one_energy",
        solver="trustregion",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])


def test_make_script_variational_solvers(tmp_path):
    """Test fanpy.scripts.utils.make_script with variational objective and different solvers."""
    script_path = str(tmp_path / "script.py")

    # CMA default solver kwargs
    make_script(
        h2_geom, "apig", basis, objective="variational", solver="cma", filename=script_path
    )
    subprocess.check_output(["python", script_path])

    # CMA with empty solver kwargs
    make_script(
        h2_geom,
        "apig",
        basis,
        objective="variational",
        solver="cma",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])

    # diag
    make_script(
        h2_geom, "doci", basis, objective="variational", solver="diag", filename=script_path
    )
    subprocess.check_output(["python", script_path])

    # minimize empty solver kwargs
    make_script(
        h2_geom,
        "ap1rog",
        basis, 
        objective="variational",
        solver="minimize",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])

    # minimize with limited memory
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        memory="2gb",
    )
    subprocess.check_output(["python", script_path])

    # minimize with wfn noise
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        wfn_noise=0.2,
    )
    subprocess.check_output(["python", script_path])

    # minimize with ham noise
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        ham_noise=0.2,
    )
    subprocess.check_output(["python", script_path])

    # minimize with specified pspace excitations
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        pspace_exc=[1, 2],
    )
    subprocess.check_output(["python", script_path])

    # minimize with orbital optimization
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        optimize_orbs=True,
    )
    subprocess.check_output(["python", script_path])


def test_make_script_checkpoint(tmp_path):
    """Test that make_script works with checkpointing options."""

    script_path = str(tmp_path / "script.py")
    # minimize with checkpointing
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        save_chk=str(tmp_path / "checkpoint.npy"),
        optimize_orbs=True,
    )
    subprocess.check_output(["python", script_path])

    # minimize with loading from checkpoint
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        load_wfn=str(tmp_path / "checkpoint_AP1roG.npy"),
    )
    subprocess.check_output(["python", script_path])

    # minimize with loading ham from checkpoint
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        load_ham=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian.npy"),
    )
    subprocess.check_output(["python", script_path])

    # minimize with loading ham and ham um (unitary H) from checkpoint
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
        load_ham=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian.npy"),
        load_ham_um=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian_um.npy"),
    )
    subprocess.check_output(["python", script_path])

def test_make_script_filename_option(tmp_path):
    """Test that make_script works the same when filename is given and not given."""

    script_path = str(tmp_path / "script.py")
    make_script(
        h2_geom,
        "ap1rog",
        basis,
        objective="variational",
        solver="minimize",
        filename=script_path,
    )
    script = make_script(
        h2_geom, "ap1rog", basis, objective="variational", solver="minimize", filename=-1
    )
    with open(script_path, "r") as f:
        assert f.read() == script
