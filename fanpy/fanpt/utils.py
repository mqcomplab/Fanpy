"""FANPT wrapper"""

import numpy as np
import pyci


def linear_comb_ham(ham1, ham0, a1, a0):
    r"""Return a linear combination of two PyCI Hamiltonians as a Fanpy Hamiltonian.

    Arguments
    ---------
    ham1 : pyci.hamiltonian
        PyCI Hamiltonian of the real system.
    ham0 : pyci.hamiltonian
        PyCI Hamiltonian of the ideal system.
    a1 : float
        Coefficient of the real Hamiltonian.
    a0 : float
        Coefficient of the ideal Hamiltonian.

    Returns
    -------
    pyci.hamiltonian
    """
    ecore = a1 * ham1.ecore + a0 * ham0.ecore
    one_mo = a1 * ham1.one_mo + a0 * ham0.one_mo
    two_mo = a1 * ham1.two_mo + a0 * ham0.two_mo

    return pyci.hamiltonian(ecore, one_mo, two_mo)


def reduce_to_fock(two_int, lambda_val=0):
    """Reduce given two electron integrals to that of the correspoding Fock operator.

    Artguments
    ----------
    two_int : np.ndarray(K, K, K, K)
        Two electron integrals of restricted orbitals.

    """
    fock_two_int = two_int * lambda_val
    nspatial = two_int.shape[0]
    indices = np.arange(nspatial)

    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]] = (
        two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]]
    )

    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]] = (
        two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]]
    )

    return fock_two_int
