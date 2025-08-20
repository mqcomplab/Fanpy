"""Unit test for fanpy.ham.utils"""

import numpy as np
import pytest 

from fanpy.ham.utils import ham_factory
from fanpy.tools import slater

def dummy_integrate_sd_sd_decomposed(sd1, sd2, integrals):
    """ Test function for the hamiltonian factory.

    Parameters
    ----------
    sd1 : int
        Slater Determinant against which the Hamiltonian is integrated.
    sd2 : int
        Slater Determinant against which the Hamiltonian is integrated.

    Returns
    -------
    dummy integral : list[float]
        Constant integral that makes testing the ham_factory easier. 

    """

    return [42.0, 0.0, 0.0]

def dummy_integrate_sd_wfn(self, sd, wfn, wfn_deriv=None):
    """ Test function for the hamiltonian factory. 
    Parameters
    ----------
    sd : int
        Slater Determinant against which the Hamiltonian is integrated.
    wfn : Wavefunction
        Wavefunction against which the Hamiltonian is integrated.
        Needs to have the following in `__dict__`: `get_overlap`.
    wfn_deriv : np.ndarray
        Indices of the wavefunction parameter against which the integral is derivatized.
        Default is no derivatization.
    
    Returns
    -------
    integral : float
        Constant answer to make testing easier. 
    """

    return 1.2

class DummyWfn:
    """ DummyWfn is a placeholder wavefunction class used for testing purposes.

    Methods
    -------
    __init__():
        Initializes the DummyWfn instance.
    get_overlap(sd, deriv=None):
        Returns a constant overlap value (1) for any input Slater determinant (sd).
        Parameters
        ----------
        sd : int or object
            Slater determinant or identifier for which overlap is calculated.
        deriv : optional
            Derivative specification (not used in this dummy implementation).
        Returns
        -------
        int
            Constant overlap value (1).

    """

    def __init__(self):
        pass

    def get_overlap(self, sd, deriv=None):
        """ Calculates the overlap for a given Slater determinant (sd).

        Parameters
        ----------
        sd : object
            The Slater determinant or representation for which the overlap is to be calculated.
        deriv : optional
            If specified, computes the derivative of the overlap with respect to the given indices.

        Returns
        -------
        float 
            Default overlap value (1). 
        """

        return 1.0

# declare necessary variables for all tests
# Note: integrals are only used in custom code the user defines
# test cases do not use any of the integrals values by desing. 
integrals = np.random.rand(10, 10)
dummy_wfn = DummyWfn()

def test_ham_factory_minimal_input():
    # create hamiltonian with dummy function
    # only declaring necessary parameters for ham_factory
    gen_ham = ham_factory(dummy_integrate_sd_sd_decomposed, 
                            integrals, 
                            nspin=10)
    
    # check whether the returned results match 
    assert gen_ham.integrate_sd_sd(slater.ground(5, 10), slater.ground(5, 10)) == 42.0

    assert np.allclose(gen_ham.integrate_sd_sd_decomposed(slater.ground(5, 10), 
                                                          slater.ground(5, 10)),
                                                          [42, 0, 0])
    
    assert gen_ham.integrate_sd_wfn(0b1101101, dummy_wfn) == 42 * ( 1 + 25 + 100 )

def test_ham_factory_errors():
    # check whether errors are raised
    # Note: some of the documented errors will not be raised by construction or custom usage of ham_factory
    gen_ham = ham_factory(dummy_integrate_sd_sd_decomposed, 
                            integrals, 
                            nspin=10)
    with pytest.raises(TypeError): # Slater det is not an integer
        gen_ham.integrate_sd_wfn(12.4, dummy_wfn)

def test_ham_factory_custom_integrate_sd_wfn():
    # create and test hamiltonian with custom integrate_sd_wfn
    gen_ham_w_integrate_sd_wfn = ham_factory(dummy_integrate_sd_sd_decomposed,
                            integrals,
                            nspin=10,
                            integrate_sd_wfn=dummy_integrate_sd_wfn)
    assert gen_ham_w_integrate_sd_wfn.integrate_sd_wfn(12, dummy_wfn) == 1.2 

def test_ham_factory_custom_orders():
    # create hamiltonian with custom orders
    gen_ham_custom_order = ham_factory(
        dummy_integrate_sd_sd_decomposed,
        integrals,
        nspin=10,
        orders=(1, 2, 3)
    )
    assert gen_ham_custom_order.integrate_sd_wfn(0b1101101, dummy_wfn) == 42 * (1 + 25 + 100 + 100)



