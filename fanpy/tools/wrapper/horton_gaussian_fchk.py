"""Script for generating one- and two-electron integrals from Gaussian using HORTON.

Functions
---------
gaussian_fchk(fchk_file, horton_internal=False, compute_nuc=True)
    Extract appropriate information from a Gaussian FCHK file.

"""
# pylint: disable=W0212,C0103
import sys

from horton import IOData, compute_nucnuc

import numpy as np


# FIXME: remove horton_internal
def gaussian_fchk(fchk_file, horton_internal=False):
    """Extract the appropriate data from Gaussian fchk file (using HORTON).

    Parameters
    ----------
    fchk_file : str
        Formatted chk file.
    horton_internal : bool
        Flag to return horton_internal variables.

    Returns
    -------
    result : dict
        "el_energy"
            Electronic energy.
        "nuc_nuc_energy"
            Nuclear nuclear repulsion energy.
        "one_int"
            Tuple of the one-electron Hamiltonian.
        "two_int"
            Tuple of the two-electron Hamiltonian.

    Raises
    ------
    NotImplementedError
        If `horton_internal` is True.

    """
    mol = IOData.from_file(fchk_file)

    # for spin orbitals
    exps = [mol.exp_alpha]
    if hasattr(mol, "exp_beta"):
        exps.append(mol.exp_beta)

    obasis = mol.obasis
    kin = obasis.compute_kinetic(mol.lf)._array
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, mol.lf)._array

    one_ab = kin
    one_ab += na
    two_ab = obasis.compute_electron_repulsion(mol.lf)._array

    # compute nuclear nuclear repulsion
    nuc_nuc = compute_nucnuc(mol.coordinates, mol.pseudo_numbers)

    # for spin orbitals
    one_mo = []
    two_mo = []
    for i, exp_i in enumerate(exps):
        for j, exp_j in enumerate(exps[i:]):
            j += i
            temp = np.einsum("sd,pqrs->pqrd", exp_j.coeffs, two_ab, casting="no", order="C")
            temp = np.einsum("rc,pqrd->pqcd", exp_i.coeffs, temp, casting="no", order="C")
            temp = np.einsum("qb,pqcd->pbcd", exp_j.coeffs, temp, casting="no", order="C")
            temp = np.einsum("pa,pbcd->abcd", exp_i.coeffs, temp, casting="no", order="C")
            two_mo.append(temp)
        one_mo.append(exp_i.coeffs.T.dot(one_ab).dot(exp_i.coeffs))

    output = {
        "el_energy": mol.energy - nuc_nuc,
        "nuc_nuc_energy": nuc_nuc,
        "one_int": tuple(one_mo),
        "two_int": tuple(two_mo),
        "t_ab_mo": tuple(i.coeffs for i in exps),
    }
    if horton_internal:
        raise NotImplementedError(
            "horton_internal storage is unsupported until it become Python " "3.6 compatible."
        )
        # output["horton_internal"] = {
        #     "mol": mol,
        #     "lf": mol.lf,
        #     "one": one_mo,
        #     "two": two_mo,
        #     "orb": exps,
        # }
    return output


if __name__ == "__main__":
    # extract keyword from command line
    kwargs = {key: val for key, val in zip(sys.argv[5::2], sys.argv[6::2])}  # pylint: disable=R1721
    # change data types
    if "horton_internal" in kwargs:
        kwargs["horton_internal"] = kwargs["horton_internal"] == "True"

    data = gaussian_fchk(**kwargs)
    np.save(sys.argv[1], [data["el_energy"], data["nuc_nuc_energy"]])
    if len(data["one_int"]) == 1:
        np.save(sys.argv[2], data["one_int"][0])
    else:
        np.save(sys.argv[2], data["one_int"])
    if len(data["two_int"]) == 1:
        np.save(sys.argv[3], data["two_int"][0])
    else:
        np.save(sys.argv[3], data["two_int"])
    if len(data["t_ab_mo"]) == 1:
        np.save(sys.argv[4], data["t_ab_mo"][0])
    else:
        np.save(sys.argv[4], data["t_ab_mo"])
