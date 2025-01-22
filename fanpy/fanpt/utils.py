""" FANPT wrapper"""

from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian

import numpy as np
import pyci


def update_fanci_objective(new_ham, fanci_objective, norm_det=None):

    # Get the class of the fanci_objective
    fanpy_objective_class = fanci_objective.fanpy_objective.__class__

    if isinstance(new_ham, pyci.hamiltonian):
        energy_nuc = new_ham.ecore
        new_ham = RestrictedMolecularHamiltonian(new_ham.one_mo, new_ham.two_mo)

    # Create new Fanpy objective
    new_fanpy_objective = fanpy_objective_class(
        fanci_objective.fanpy_wfn,
        new_ham,
        param_selection=fanci_objective.param_selection,
        optimize_orbitals=fanci_objective.fanpy_objective.optimize_orbitals,
        step_print=fanci_objective.step_print,
        step_save=fanci_objective.step_save,
        tmpfile=fanci_objective.tmpfile,
        pspace=fanci_objective.fanpy_objective.pspace,
        refwfn=fanci_objective.fanpy_objective.refwfn,
        eqn_weights=fanci_objective.fanpy_objective.eqn_weights,
        energy_type=fanci_objective.fanpy_objective.energy_type,
        energy=fanci_objective.fanpy_objective.energy.params,
        constraints=fanci_objective.fanpy_objective.constraints,
    )

    # Build FanCI objective as PyCI interface
    from fanpy.interface.pyci import PYCI

    fanci_interface = PYCI(
        new_fanpy_objective,
        energy_nuc,
        norm_det=norm_det,
        max_memory=fanci_objective.max_memory,
        legacy=fanci_objective.legacy_fanci,
    )

    return fanci_interface.objective


def linear_comb_ham(ham1, ham0, a1, a0):
    r"""Return a linear combination of two PyCI Hamiltonians as a Fanpy Hamiltonian.

    Parameters
    ----------
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

    Parameters
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
