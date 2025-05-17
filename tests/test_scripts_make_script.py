"""Test fanpy.script.make_script."""
import subprocess
import pytest

from fanpy.scripts.make_script import make_script

from utils import find_datafile

@pytest.mark.skip(reason="This test fails and is being worked on (PR 34).")
def test_make_script(tmp_path):
    """Test fanpy.scripts.utils.make_script."""
    oneint = find_datafile("data/h2_hf_sto6g_oneint.npy")
    twoint = find_datafile("data/h2_hf_sto6g_twoint.npy")
    script_path = str(tmp_path / "script.py")

    wfn_list = [
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
    ]
    for wfn in wfn_list:
        make_script(2, oneint, twoint, wfn, filename=script_path)
        subprocess.check_output(["python", script_path])
        make_script(2, oneint, twoint, wfn, filename=script_path, wfn_kwargs="")
        subprocess.check_output(["python", script_path])

    for objective in ["least_squares", "variational", "one_energy"]:
        make_script(
            2,
            oneint,
            twoint,
            "ap1rog",
            objective=objective,
            solver="minimize",
            filename=script_path,
        )
        subprocess.check_output(["python", script_path])

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

    make_script(
        2, oneint, twoint, "apig", objective="variational", solver="cma", filename=script_path
    )
    subprocess.check_output(["python", script_path])
    make_script(
        2,
        oneint,
        twoint,
        "apig",
        objective="variational",
        solver="cma",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2, oneint, twoint, "apig", objective="projected", solver="root", filename=script_path
    )
    subprocess.check_output(["python", script_path])
    make_script(
        2,
        oneint,
        twoint,
        "apig",
        objective="projected",
        solver="root",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2, oneint, twoint, "doci", objective="variational", solver="diag", filename=script_path
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        solver_kwargs="",
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        memory="2gb",
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        wfn_noise=0.2,
    )
    subprocess.check_output(["python", script_path])
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        ham_noise=0.2,
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        pspace_exc=[1, 2],
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        optimize_orbs=True,
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        save_chk=str(tmp_path / "checkpoint.npy"),
        optimize_orbs=True,
    )
    subprocess.check_output(["python", script_path])
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        load_wfn=str(tmp_path / "checkpoint_AP1roG.npy"),
    )
    subprocess.check_output(["python", script_path])
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        load_ham=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian.npy"),
    )
    subprocess.check_output(["python", script_path])
    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
        load_ham=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian.npy"),
        load_ham_um=str(tmp_path / "checkpoint_RestrictedMolecularHamiltonian_um.npy"),
    )
    subprocess.check_output(["python", script_path])

    make_script(
        2,
        oneint,
        twoint,
        "ap1rog",
        objective="variational",
        solver="minimize",
        filename=script_path,
    )
    script = make_script(
        2, oneint, twoint, "ap1rog", objective="variational", solver="minimize", filename=-1
    )
    with open(script_path, "r") as f:
        assert f.read() == script
