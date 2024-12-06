r"""Routines to create and manage Natural Orbitals for a CIWavefunction.

Natural orbitals and their occupations are derived from the **one-particle reduced density matrix (1-RDM)**, defined as:

.. math::
    \gamma_{\mu \nu} = \langle \Psi | \hat{a}^\dagger_\nu \hat{a}_\mu | \Psi \rangle

where:
- :math:`\gamma_{\mu \nu}` is the 1-RDM.
- :math:`\Psi` is the CI wavefunction.
- :math:`\hat{a}^\dagger_\nu` and :math:`\hat{a}_\mu` are the creation and annihilation operators for spin orbitals.

The 1-RDM can be decomposed as:

.. math::
    \gamma_{\mu \nu} = \sum_i n_i \phi_i^*(\mu) \phi_i(\nu)

where:
- :math:`\phi_i` are the **natural orbitals**.
- :math:`n_i` are the corresponding **occupation numbers**.

The natural orbitals are obtained by solving the eigenvalue problem for the 1-RDM:

.. math::
    \gamma \mathbf{\phi}_\mu = n_\mu \mathbf{\phi}_\mu

Here, :math:`n_\mu` represents the eigenvalues (natural occupation numbers), and :math:`\mathbf{\phi}_\mu` are the eigenvectors (natural orbitals).

This module provides routines to compute, manipulate, and analyze natural orbitals and their occupation numbers for correlated CI wavefunctions.
See: [DOI:10.1016/S0065-3276(08)60547-X]

Functions
---------
make_natural_orbitals(wfn, ham) : {CIWavefunction, BaseHamiltonian}
    Create Natural Orbitals from a given CIWavefunction.
"""

import numpy as np
import scipy

def compute_natural_orbitals(wfn, mo_coeff=None):
    """Compute natural orbitals (NOs) and occupation numbers from the 1-RDM in the molecular orbital (MO) basis.
    If the molecular orbitals in atomic orbital (AO) basis is provided, compute NOs in AO basis.

    Parameters
    ----------
    wfn : CIWavefunction
        Reference CI wavefunction containing methods to compute the 1-RDM.
    mo_coeff : np.ndarray
        Array containing molecular orbitals in the atomic orbital (AO) basis.

    Returns
    -------
    occ_numbers : np.ndarray
        Occupation numbers of natural orbitals.
    natural_orbitals : np.ndarray
        Natural orbitals for beta spin.

    """
    # Compute 1-RDMs for the given CI wavefunction
    rdm1 = wfn.compute_1rdm()

    # Splitting 1-RDMs in \gamma(\alpha, \alpha) and \gamma(\beta, \beta)
    rdm1_alpha = rdm1[:wfn.nspatial, :wfn.nspatial]
    rdm1_beta = rdm1[wfn.nspatial:, wfn.nspatial:]

    # Perform eigendecomposition of the 1-RDM
    occ_alpha, natural_orbitals_alpha = scipy.linalg.eigh(rdm1_alpha)
    occ_beta, natural_orbitals_beta = scipy.linalg.eigh(rdm1_beta)

    # Sort natural orbitals and occupation numbers
    idx_alpha = np.argsort(occ_alpha)[::-1]
    occ_alpha = occ_alpha[idx_alpha]
    natural_orbitals_alpha = natural_orbitals_alpha[:, idx_alpha]

    idx_beta = np.argsort(occ_beta)[::-1]
    occ_beta = occ_beta[idx_beta]
    natural_orbitals_beta = natural_orbitals_beta[:, idx_beta]

    # Transform NOs from MO basis to AO basis
    if isinstance(mo_coeff, np.ndarray):
        natural_orbitals_alpha = np.dot(mo_coeff, natural_orbitals_alpha)
        natural_orbitals_beta = np.dot(mo_coeff, natural_orbitals_beta)

    # Concatenate natural orbitals occupation numbers
    occ_numbers = np.concatenate((occ_alpha, occ_beta))

    # Concatenate spin natural orbitals
    natural_orbitals = np.zeros((wfn.nspatial, wfn.nspin))
    natural_orbitals[:, :wfn.nspatial] = natural_orbitals_alpha
    natural_orbitals[:, wfn.spatial:] = natural_orbitals_beta

    return occ_numbers, natural_orbitals

def save_natural_orbitals(filename, wfn, mol, mo_coeff):
    """Save natural orbitals (NOs) at the atomic orbital (AO) basis in the Molden file format.

    Parameters
    ----------
    filename : str
        Name of the Molden file to be created. If the provided filename does not include the
        '.molden' extension, it will be added automatically.
    wfn : CIWavefunction
        Reference CI wavefunction containing methods to compute the 1-RDM.
    mol : pyscf.gto.mole.Mole
        PySCF object containing basis sets and other required data to build the Molden file.
    mo_coeff : np.ndarray
        Array containing molecular orbitals in the atomic orbital (AO) basis.

    """
    try:
        from pyscf.tools import molden
    except ImportError:
        print("# ERROR: PySCF package not found.")

    occ_numbers, natural_orbitals = compute_natural_orbitals(wfn, mo_coeff=mo_coeff)

    if not filename.endswith('.molden'):
        filename += '.molden'

    with open(filename, 'w') as file:
        molden.header(mol, file)
        molden.orbital_coeff(mol, file, natural_orbitals, occ=occ_numbers, ene=occ_numbers)
        print("Molden file {:} sucessfully exported.".format(filename))