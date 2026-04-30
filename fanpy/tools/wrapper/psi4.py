"""Script for utilizing Psi4."""
import numpy as np

import psi4


def hartreefock(xyz_file, basis, is_unrestricted=False):  # pylint: disable=W0613
    """Run HF using PySCF.

    Parameters
    ----------
    xyz_file : str
        XYZ file location.
        Units are in Angstrom.
    basis : str
        Basis set available in PySCF.
    is_unrestricted : bool
        Flag to run unrestricted HF.
        Default is restricted HF.

    Returns
    -------
    result : dict
        "hf_energy"
            The electronic energy.
        "nuc_nuc"
            The nuclear repulsion energy.
        "one_int"
            The tuple of the one-electron interal.
        "two_int"
            The tuple of the two-electron integral in Physicist's notation.

    Raises
    ------
    ValueError
        If given xyz file does not exist.
    NotImplementedError
        If calculation is unrestricted or generalized.

    """
    # pylint: disable=E1101,C0103
    # get coordinates
    with open(xyz_file, "r") as fh:
        lines = [i.strip() for i in fh.readlines()[2:]]
        geometry = "\n".join(lines + ["symmetry c1"])

    mol = psi4.geometry(geometry)
    psi4.set_options({"basis": basis, "reference": "rhf", "scf_type": "direct"})
    hf_energy, hf_wfn = psi4.energy("scf", return_wfn=True)
    C = hf_wfn.Ca()  # noqa: N806
    npC = C.to_array()  # noqa: N806

    mints = psi4.core.MintsHelper(hf_wfn.basisset())
    one_int = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    one_int = np.einsum("uj,vi,uv", npC, npC, one_int)
    two_int = np.asarray(mints.mo_eri(C, C, C, C))
    two_int = two_int.swapaxes(1, 2)

    result = {
        "hf_energy": hf_energy,
        "nuc_nuc": mol.nuclear_repulsion_energy(),
        "one_int": one_int,
        "two_int": two_int,
    }
    return result
