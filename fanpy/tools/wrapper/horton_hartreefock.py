"""Script for generating one and two electron integrals using HORTON.

Functions
---------
hartreefock(xyz_file, basis, is_unrestricted=False)
    Runs HF in HORTON.

"""
# pylint: disable=W0212,C0103,W0621
import sys

from horton import (
    AufbauOccModel,
    DenseLinalgFactory,
    EDIIS2SCFSolver,
    IOData,
    PlainSCFSolver,
    RDirectTerm,
    REffHam,
    RExchangeTerm,
    RTwoIndexTerm,
    compute_nucnuc,
    get_gobasis,
    guess_core_hamiltonian,
    transform_integrals,
)

import numpy as np


def hartreefock(
    fn=None,
    basis=None,
    nelec=None,
    solver=EDIIS2SCFSolver,
    tol=1.0e-12,
    horton_internal=False,
    **kwargs
):
    """Run a HF calculation using HORTON.

    Parameters
    ----------
    fn : str
        File name.
        Supports XYZ file (in Angstrom).
    basis : str
        Basis set name.
    nelec : int
        Number of electrons.
    solver : {PlainSCFSolver, ODASSCFSolver, CDIISSCFSolver, EDIISSCFSolver,  EDIIS2SCFSolver}
        HORTON's SCF solver.
    tol : float
        The convergence threshold for the wavefunction.
    horton_internal : bool
        Flag for returning HORTON's internal data.
    kwargs
        Other optional arguments for the SCF Solver.

    Returns
    -------
    result : dict
        "el_energy"
            Electronic energy.
        "nuc_nuc_energy"
            Nuclear repulsion energy.
        "one_int"
            Tuple of the one-electron Hamiltonian.
        "two_int"
            Tuple of the two-electron Hamiltonian.

    Raises
    ------
    ValueError
        If solver is not callable.
    NotImplemntedError
        If `horton_internal` is True.

    Notes
    -----
    While HORTON does support unrestricted calculations, the wrapper does not support it.

    """
    # Initialize molecule and basis set from specified file
    if isinstance(fn, IOData):
        mol = fn
    else:
        mol = IOData.from_file(fn)
    obasis = get_gobasis(mol.coordinates, mol.numbers, basis)
    npair = nelec // 2

    # Fill in orbital expansion and overlap
    occ_model = AufbauOccModel(npair)
    lf = DenseLinalgFactory(obasis.nbasis)
    orb = lf.create_expansion(obasis.nbasis)
    olp = obasis.compute_overlap(lf)

    # Construct Hamiltonian and density matrix
    kin = obasis.compute_kinetic(lf)
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
    two = obasis.compute_electron_repulsion(lf)
    external = {"nn": compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        RTwoIndexTerm(kin, "kin"),
        RDirectTerm(two, "hartree"),
        RExchangeTerm(two, "x_hf"),
        RTwoIndexTerm(na, "ne"),
    ]
    ham = REffHam(terms, external)
    guess_core_hamiltonian(olp, kin, na, orb)

    # Do Hartree-Fock SCF
    if solver is PlainSCFSolver:
        PlainSCFSolver(tol)(ham, lf, olp, occ_model, orb, **kwargs)
    elif hasattr(solver, "__call__"):
        occ_model.assign(orb)
        dm = orb.to_dm()
        solver(tol)(ham, lf, olp, occ_model, dm, **kwargs)
        # transform back to orbitals
        fock_alpha = lf.create_two_index()
        ham.compute_fock(fock_alpha)
        orb.from_fock_and_dm(fock_alpha, dm, olp)
    else:
        raise ValueError("Given solver, {0}, is not callable".format(solver))
    energy = ham.cache["energy"]

    # Transform one- and two- electron integrals into MO basis
    one = kin
    one.iadd(na)
    one_mo, two_mo = transform_integrals(one, two, "tensordot", orb)
    one_mo = tuple(i._array for i in one_mo)
    two_mo = tuple(i._array for i in two_mo)

    output = {
        # Collect HF energy and integrals
        "el_energy": energy - external["nn"],
        "nuc_nuc_energy": external["nn"],
        "one_int": one_mo,
        "two_int": two_mo,
    }

    if horton_internal:
        output["horton_internal"] = {
            "lf": lf,
            "occ_model": occ_model,
            "one": one,
            "two": two,
            "orb": [orb],
            "olp": olp,
        }
        raise NotImplementedError(
            "horton_internal storage is unsupported until it become Python " "3.6 compatible."
        )

    return output


if __name__ == "__main__":
    # extract keyword from command line
    kwargs = {key: val for key, val in zip(sys.argv[4::2], sys.argv[5::2])}  # pylint: disable=R1721
    # change data types
    if "nelec" in kwargs:
        kwargs["nelec"] = int(kwargs["nelec"])
    if "tol" in kwargs:
        kwargs["tol"] = float(kwargs["tol"])
    if "horton_internal" in kwargs:
        kwargs["horton_internal"] = bool(kwargs["horton_internal"])
    if "solver" in kwargs:
        kwargs["solver"] = locals()[kwargs["solver"]]

    data = hartreefock(**kwargs)
    np.save(sys.argv[1], [data["el_energy"], data["nuc_nuc_energy"]])
    if len(data["one_int"]) == 1:
        np.save(sys.argv[2], data["one_int"][0])
    else:
        np.save(sys.argv[2], data["one_int"])
    if len(data["two_int"]) == 1:
        np.save(sys.argv[3], data["two_int"][0])
    else:
        np.save(sys.argv[3], data["two_int"])
