"""Test fanpy.tools.wrapper.horton."""
import os
import sys
from subprocess import call

from fanpy.tools.wrapper.python_wrapper import generate_hartreefock_results
from interface_utils import check_data_h2_rhf_sto6g, check_data_h2_uhf_sto6g, check_dependency

import numpy as np

import pytest

from utils import find_datafile

def test_generate_hartreefock_results_error():
    """Test python_wrapper.generate_hartreefock_results.

    Check if it raises correct error.

    """
    with pytest.raises(ValueError):
        generate_hartreefock_results("sdf")
    os.environ["HORTONPYTHON"] = "Asfasdfsadf"
    with pytest.raises(FileNotFoundError):
        generate_hartreefock_results("horton_hartreefock.py")


@pytest.mark.skipif(
    not check_dependency("horton"), reason="HORTON is not available or HORTONPATH is not set."
)
def test_horton_hartreefock_h2_rhf_sto6g():
    """Test HORTON"s hartreefock against H2 HF STO-6G data from Gaussian."""
    hf_data = generate_hartreefock_results(
        "horton_hartreefock.py",
        energies_name="energies.npy",
        oneint_name="oneint.npy",
        twoint_name="twoint.npy",
        remove_npyfiles=True,
        fn=find_datafile("data/data_h2.xyz"),
        basis="sto-6g",
        nelec=2,
    )
    check_data_h2_rhf_sto6g(*hf_data)


@pytest.mark.skipif(
    not check_dependency("horton"), reason="HORTON is not available or HORTONPATH is not set."
)
def test_horton_gaussian_fchk_h2_rhf_sto6g():
    """Test HORTON"s gaussian_fchk against H2 HF STO-6G data from Gaussian."""
    fchk_data = generate_hartreefock_results(
        "horton_gaussian_fchk.py",
        energies_name="energies.npy",
        oneint_name="oneint.npy",
        twoint_name="twoint.npy",
        remove_npyfiles=True,
        fchk_file=find_datafile("data/data_h2_hf_sto6g.fchk"),
        horton_internal=False,
    )
    check_data_h2_rhf_sto6g(*fchk_data)


@pytest.mark.skipif(
    not check_dependency("horton"), reason="HORTON is not available or HORTONPATH is not set."
)
def test_gaussian_fchk_h2_uhf_sto6g():
    """Test HORTON"s gaussian_fchk against H2 UHF STO-6G data from Gaussian."""
    fchk_data = generate_hartreefock_results(
        "horton_gaussian_fchk.py",
        energies_name="energies.npy",
        oneint_name="oneint.npy",
        twoint_name="twoint.npy",
        remove_npyfiles=True,
        fchk_file=find_datafile("data/data_h2_uhf_sto6g.fchk"),
        horton_internal=False,
    )
    check_data_h2_uhf_sto6g(*fchk_data)
