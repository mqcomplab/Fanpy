"""Test fanpy.script.gaussian.make_fanci_script."""
import subprocess
import pytest

from fanpy.scripts.gaussian.make_fanci_script import make_script

from utils import find_datafile


oneint = find_datafile("data/data_h2_hf_sto6g_oneint.npy")
twoint = find_datafile("data/data_h2_hf_sto6g_twoint.npy")

@pytest.mark.parametrize("wfn", [
    "ci_pairs",
    "cisd", 
    "fci",
    "doci",
    "mps",
    "determinant-ratio",
    "ap1rog",
    "apr2g",
    "apig",
    "apsetg",
    "apg",
])
def test_make_script_wfns(tmp_path, wfn):
    """Test fanpy.scripts.utils.make_script with different wavefunctions.
    """
    script_path = str(tmp_path / "script.py")


    make_script(2, oneint, twoint, wfn, nproj=4, filename=script_path)
    subprocess.check_output(["python", script_path])

    make_script(2, oneint, twoint, wfn, nproj=4, filename=script_path, wfn_kwargs="")
    subprocess.check_output(["python", script_path])
make_script(2, oneint, twoint, "apr2g", nproj=4, filename="example.py")
@pytest.mark.parametrize("objective", ["projected"])
def test_make_script_objectives(tmp_path, objective):
    """Test fanpy.scripts.utils.make_script with different objectives."""
    script_path = str(tmp_path / "script.py")
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective=objective,
        solver="least_squares",
        filename=script_path,
    )
    subprocess.check_output(["python", script_path])


def test_make_script_projected_solvers(tmp_path):
    """Test fanpy.scripts.utils.make_script with projected objective and different solvers."""
    script_path = str(tmp_path / "script.py")
    # least squares
    make_script(
        2,
        oneint,
        twoint,
        "apig",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])

    # root default solver kwargs
    oneint_root = find_datafile("data/data_h4_square_hf_sto6g_oneint.npy")
    twoint_root = find_datafile("data/data_h4_square_hf_sto6g_twoint.npy")
    make_script(
        4, oneint_root, twoint_root, "cisd", wfn_kwargs = "spin=0", pspace_exc=[1, 2, 3, 4], objective="projected", solver="root", filename=script_path
    ) # spin needs to be 0 so that system is not underdetermined
    subprocess.check_output(["python", script_path])

    # root with empty solver kwargs
    make_script(
        4,
        oneint_root,
        twoint_root,
        "cisd",
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
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        memory="2gb",
    )
    subprocess.check_output(["python", script_path])

    # wfn noise
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        wfn_noise=0.2,
    )
    subprocess.check_output(["python", script_path])

    # ham noise
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        ham_noise=0.2,
    )
    subprocess.check_output(["python", script_path])

    # pspace excitations
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
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
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        save_chk=str(tmp_path / "checkpoint.npy"),
    )
    subprocess.check_output(["python", script_path])

    # minimize with loading from checkpoint
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        load_wfn=str(tmp_path / "checkpoint_AP1roG.npy"),
    )
    subprocess.check_output(["python", script_path])

    # minimize with loading ham from checkpoint
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        load_ham=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian.npy"),
    )
    subprocess.check_output(["python", script_path])

    # minimize with loading ham and ham um (unitary H) from checkpoint
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
        load_ham=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian.npy"),
        load_ham_um=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian_um.npy"),
    )
    subprocess.check_output(["python", script_path])

def test_make_script_filename_option(tmp_path):
    """Test that make_script works the same when filename is given and not given."""

    script_path = str(tmp_path / "script.py")
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="projected",
        solver="least_squares",
        filename=script_path,
    )
    script = make_script(
        2, oneint, twoint, "ap1rog", objective="projected", solver="least_squares", filename=-1
    )
    with open(script_path, "r") as f:
        assert f.read() == script
