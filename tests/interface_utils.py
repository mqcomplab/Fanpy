"""Test fanpy.tools.wrapper.horton."""
import os
import sys
from subprocess import call

import numpy as np

from fanpy.eqn.projected import BaseSchrodinger
from fanpy.wfn.cc.standard_cc import StandardCC
from fanpy.wfn.base import BaseWavefunction
from fanpy.ham.base import BaseHamiltonian

#################### FAKE CLASSES ######################################

class FakeSchrodinger(BaseSchrodinger):
    """fake fanpy objective for testing purposes"""
    def __init__(self, wfn, ham):
        super().__init__(wfn, ham)
    def objective(self, params):
        return 3.08

class FakeCC(StandardCC):
    """fake CC wavefunction for testing purposes
    This is used to test the double derivative of the overlap.
    """
    def __init__(self, nelec, nspin):
        super().__init__(nelec, nspin)

    def get_overlap_double_derivative(self, sd):
        double_deriv = np.ones((self.nparams, self.nparams))
        return double_deriv

class FakeWavefunction(BaseWavefunction):
    """ A fake wavefunction for testing purposes.
    """

    def __init__(self, nelec, nspin, params):
        """ Initialize FakeWavefunction.

        Args:
            nelec (int): Number of electrons.
            nspin (int): Number of spin orbitals.
        """
        self.assign_params(params)
        super().__init__(nelec, nspin)

    def get_overlap(self, sd, deriv=None):
        """ Get overlap with a Slater determinant.

        Args:
            sd (int): Slater determinant in integer representation. -> not used for fake implementation.
            deriv (np.ndarray, optional): If provided, compute the derivative of the overlap.

        Returns:
            float: Overlap value.
        """
        if deriv is not None:
            return np.zeros(len(deriv))
        else:
            return 1.0 
        
    def assign_params(self, params):
        """ Assign parameters to the wavefunction.

        Args:
            params (np.ndarray): Parameters to assign.
        """
        self.params = params

class FakeHamiltonian(BaseHamiltonian):
    """ A fake Hamiltonian for testing purposes.
    """
    def __init__(self, one_int, two_int):
        """ Initialize FakeHamiltonian.

        Args:
            one_int (np.ndarray): One-electron integrals.
            two_int (np.ndarray): Two-electron integrals.
        """
        self.one_int = one_int
        self.two_int = two_int
        self._nspin = one_int.shape[0] * 2 # assuming one_int is square and its size corresponds to the orbital space
    @property
    def nspin(self):
        """ Return the number of spin orbitals.

        Returns:
            int: Number of spin orbitals.
        """
        return self._nspin
    
    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        """ Integrate the Hamiltonian with against two Slater determinants.

        Args:
            sd1 (int): First Slater determinant in integer representation. -> not used for fake implementation.
            sd2 (int): Second Slater determinant in integer representation. -> not used for fake implementation.
            deriv (np.ndarray, optional): If provided, compute the derivative of the integral.

        Returns:
            float: Integral value.
        """
        if deriv is not None:
            return np.zeros(len(deriv))
        else:
            return 1.0


###################### HF DATA CHECKS ##################################

def check_data_h2_rhf_sto6g(el_energy, nuc_nuc_energy, one_int, two_int):
    """Check data for h2 rhf sto6g calculation."""
    assert np.allclose(el_energy, -1.838434259892)
    assert np.allclose(nuc_nuc_energy, 0.713176830593)
    assert np.allclose(
        one_int, np.array([[-1.25637540e00, 0.0000000000000], [0.0000000000000, -4.80588203e-01]])
    )
    assert np.allclose(
        two_int,
        np.array(
            [
                [
                    [[6.74316543e-01, 0.000000000000], [0.000000000000, 1.81610048e-01]],
                    [[0.000000000000, 6.64035234e-01], [1.81610048e-01, 0.000000000000]],
                ],
                [
                    [[0.000000000000, 1.81610048e-01], [6.64035234e-01, 0.000000000000]],
                    [[1.81610048e-01, 0.000000000000], [0.000000000000, 6.98855952e-01]],
                ],
            ]
        ),
    )


def check_data_h2_uhf_sto6g(el_energy, nuc_nuc_energy, one_int, two_ints):
    """Check data for h2 uhf sto6g calculation."""
    assert np.allclose(el_energy, -1.838434259892)
    assert np.allclose(nuc_nuc_energy, 0.713176830593)
    assert np.allclose(
        one_int[0],
        np.array([[-1.25637540e00, 0.0000000000000], [0.0000000000000, -4.80588203e-01]]),
    )
    assert np.allclose(
        one_int[1],
        np.array([[-1.25637540e00, 0.0000000000000], [0.0000000000000, -4.80588203e-01]]),
    )
    assert len(two_ints) == 3
    for two_int in two_ints:
        assert np.allclose(
            two_int,
            np.array(
                [
                    [
                        [[6.74316543e-01, 0.000000000000], [0.000000000000, 1.81610048e-01]],
                        [[0.000000000000, 6.64035234e-01], [1.81610048e-01, 0.000000000000]],
                    ],
                    [
                        [[0.000000000000, 1.81610048e-01], [6.64035234e-01, 0.000000000000]],
                        [[1.81610048e-01, 0.000000000000], [0.000000000000, 6.98855952e-01]],
                    ],
                ]
            ),
        )


# NOTE: the integrals from PySCF have different sign from HORTON's for some reason
# def check_data_h2_rhf_631gdp(el_energy, nuc_nuc_energy, one_int, two_int):
#     """Check data for LiH rhf sto6g calculation."""
#     assert np.allclose(el_energy, -1.84444667027)
#     assert np.allclose(nuc_nuc_energy, 0.7131768310)

#     # check types of the integrals
#     assert np.allclose(one_int, np.load(find_datafile('data_h2_hf_631gdp_oneint.npy')))
#     assert np.allclose(two_int, np.load(find_datafile('data_h2_hf_631gdp_twoint.npy')))


def check_data_lih_rhf_sto6g(*data):
    """Check data for LiH rhf sto6g calculation."""
    el_energy, nuc_nuc_energy, one_int, two_int = data

    assert np.allclose(el_energy, -7.95197153880 - 0.9953176337)
    assert np.allclose(nuc_nuc_energy, 0.9953176337)

    # check types of the integrals
    assert isinstance(one_int, np.ndarray)
    assert isinstance(two_int, np.ndarray)
    assert np.all(np.array(one_int.shape) == one_int[0].shape[0])
    assert np.all(np.array(two_int.shape) == one_int[0].shape[0])


def check_dependency(dependency):
    """Check if the dependency is available.

    Parameters
    ----------
    dependency : {'horton', 'pyscf'}
        Dependency for the functions `generate_hartreefock_results` and `generate_fci_results`

    Returns
    -------
    is_avail : bool
        If given dependency is available.

    """
    dependency = dependency.lower()
    try:
        if dependency == "horton":
            python_name = os.environ["HORTONPYTHON"]
        elif dependency == "pyscf":
            python_name = os.environ["PYSCFPYTHON"]
    except KeyError:
        python_name = sys.executable
    if not os.path.isfile(python_name):
        return False
    # FIXME: I can't think of a way to make sure that the python_name is a python interpreter.

    # NOTE: this is a possible security risk since we don't check that python_name is actually a
    # python interpreter. However, it is up to the user to make sure that their environment variable
    # is set up properly. Ideally, we shouldn't even have to call a python outside the current
    # python, but here we are.
    exit_code = call([python_name, "-c", "import {0}".format(dependency)])
    return exit_code == 0