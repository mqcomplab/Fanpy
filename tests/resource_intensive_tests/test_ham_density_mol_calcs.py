"""Test fanpy.ci.density."""
from fanpy.ham.density import density_matrix

import numpy as np

from utils import find_datafile

# Tests using examples
def test_density_matrix_restricted_h2_fci_sto6g():
    """Test density.density_matrix using H2 FCI/STO-6G.

    Uses numbers obtained from Gaussian and PySCF
        Gaussian's one electron density matrix is used to compare
        PySCF's SD coefficient is used to construct density matrix
        Electronic energy of FCI from Gaussian and PySCF were the same

    Coefficients for the Slater determinants (from PySCF) are [0.993594152, 0.0, 0.0, -0.113007352]
    FCI Electronic energy is -1.85908985 Hartree

    """
    sd_coeffs = np.array([0.993594152, 0.0, 0.0, -0.113007352])
    one_density, two_density = density_matrix(
        sd_coeffs,
        [0b0101, 0b1001, 0b0110, 0b1010],
        2,
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="restricted",
    )
    # Count number of electrons
    assert abs(np.einsum("ii", one_density[0]) - 2) < 1e-8

    # Compare to reference
    ref_one_density = np.array([[0.197446e01, -0.163909e-14], [-0.163909e-14, 0.255413e-01]])
    assert np.allclose(one_density[0], ref_one_density)

    # Reconstruct FCI energy
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"][0]
    # two_int = hf_dict["two_int"][0]
    one_int = np.load(find_datafile("../data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h2_hf_sto6g_twoint.npy"))
    # physicist notation
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,ijkl", two_int, two_density[0])
            )
            - (-1.85908985)
        )
        < 1e-8
    )
    # chemist notation
    one_density, two_density = density_matrix(
        sd_coeffs,
        [0b0101, 0b1001, 0b0110, 0b1010],
        2,
        is_chemist_notation=True,
        val_threshold=0,
        orbtype="restricted",
    )
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,iklj", two_int, two_density[0])
            )
            - (-1.85908985)
        )
        < 1e-8
    )


def test_density_matrix_restricted_h2_631gdp():
    """Test density.density_matrix using H2 system (FCI/6-31G**).

    Uses numbers obtained from PySCF
        PySCF's SD coefficient is used to construct density matrix
        Electronic energy of FCI

    FCI Electronic energy is -1.87832559 Hartree

    """
    nelec = 2

    # read in gaussian fchk file and generate one and two electron integrals (using horton)
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"][0]
    # two_int = hf_dict["two_int"][0]
    one_int = np.load(find_datafile("../data/data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h2_hf_631gdp_twoint.npy"))

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load(find_datafile("../data/data_h2_hf_631gdp_cimatrix.npy"))
    civec = np.load(find_datafile("../data/data_h2_hf_631gdp_civec.npy"))
    civec = [int(i) for i in civec]
    sd_coeffs = np.linalg.eigh(ci_matrix)[1][:, 0]
    energy = np.linalg.eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="restricted",
    )

    # Count number of electrons
    assert abs(np.einsum("ii", one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,ijkl", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )
    # chemist notation
    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=True,
        val_threshold=0,
        orbtype="restricted",
    )
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,iklj", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )


def test_density_matrix_restricted_lih_sto6g():
    """Test density.density_matrix using LiH system (FCI/STO6G) with restricted orbitals.

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI

    """
    nelec = 4

    # read in gaussian fchk file and generate one and two electron integrals (using horton)
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"][0]
    # two_int = hf_dict["two_int"][0]
    one_int = np.load(find_datafile("../data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_sto6g_twoint.npy"))

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load(find_datafile("../data/data_lih_hf_sto6g_cimatrix.npy"))
    civec = np.load(find_datafile("../data/data_lih_hf_sto6g_civec.npy"))
    civec = [int(i) for i in civec]
    sd_coeffs = np.linalg.eigh(ci_matrix)[1][:, 0]
    energy = np.linalg.eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="restricted",
    )

    # Count number of electrons
    assert abs(np.einsum("ii", one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,ijkl", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )
    # chemist notation
    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=True,
        val_threshold=0,
        orbtype="restricted",
    )
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,iklj", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )


def test_density_matrix_restricted_lih_631g_slow():
    """Test density.density_matrix using LiH system (FCI/6-31G) with restricted orbitals.

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI

    """
    nelec = 4

    # read in gaussian fchk file and generate one and two electron integrals (using horton)
    # hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
    # one_int = hf_dict["one_int"][0]
    # two_int = hf_dict["two_int"][0]
    one_int = np.load(find_datafile("../data/data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_631g_twoint.npy"))

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load(find_datafile("../data/data_lih_hf_631g_cimatrix.npy"))
    civec = np.load(find_datafile("../data/data_lih_hf_631g_civec.npy"))
    civec = [int(i) for i in civec]
    sd_coeffs = np.linalg.eigh(ci_matrix)[1][:, 0]
    energy = np.linalg.eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="restricted",
    )

    # Count number of electrons
    assert abs(np.einsum("ii", one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,ijkl", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )
    # chemist notation
    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=True,
        val_threshold=0,
        orbtype="restricted",
    )
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,iklj", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )


def test_density_matrix_unrestricted_lih_sto6g():
    """Test density.density_matrix using LiH system (FCI/STO6G) with unrestricted orbitals.

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI

    """
    nelec = 4

    # read in gaussian fchk file and generate one and two electron integrals (using horton)
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"][0]
    # two_int = hf_dict["two_int"][0]
    one_int = np.load(find_datafile("../data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_sto6g_twoint.npy"))

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load(find_datafile("../data/data_lih_hf_sto6g_cimatrix.npy"))
    civec = np.load(find_datafile("../data/data_lih_hf_sto6g_civec.npy"))
    civec = [int(i) for i in civec]
    sd_coeffs = np.linalg.eigh(ci_matrix)[1][:, 0]
    energy = np.linalg.eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="unrestricted",
    )

    # Count number of electrons
    assert abs(sum(np.einsum("ii", i) for i in one_density) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + np.einsum("ij,ij", one_int, one_density[1])
                + 0.5 * np.einsum("ijkl,ijkl", two_int, two_density[0])
                + np.einsum("ijkl,ijkl", two_int, two_density[1])
                + 0.5 * np.einsum("ijkl,ijkl", two_int, two_density[2])
            )
            - (energy)
        )
        < 1e-8
    )
    # chemist notation
    one_density, two_density = density_matrix(
        sd_coeffs,
        civec,
        one_int.shape[0],
        is_chemist_notation=True,
        val_threshold=0,
        orbtype="unrestricted",
    )
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + np.einsum("ij,ij", one_int, one_density[1])
                + 0.5 * np.einsum("ijkl,iklj", two_int, two_density[0])
                + np.einsum("ijkl,iklj", two_int, two_density[1])
                + 0.5 * np.einsum("ijkl,iklj", two_int, two_density[2])
            )
            - (energy)
        )
        < 1e-8
    )


def test_density_matrix_generalized_lih_sto6g():
    """Test density.density_matrix using LiH system (FCI/STO6G) with generalized orbitals.

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI

    """
    nelec = 4

    # read in gaussian fchk file and generate one and two electron integrals (using horton)
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"][0]
    # two_int = hf_dict["two_int"][0]
    one_int = np.load(find_datafile("../data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_sto6g_twoint.npy"))

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load(find_datafile("../data/data_lih_hf_sto6g_cimatrix.npy"))
    civec = np.load(find_datafile("../data/data_lih_hf_sto6g_civec.npy"))
    civec = [int(i) for i in civec]
    sd_coeffs = np.linalg.eigh(ci_matrix)[1][:, 0]
    energy = np.linalg.eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(
        sd_coeffs, civec, 6, is_chemist_notation=False, val_threshold=0, orbtype="generalized"
    )

    # Count number of electrons
    assert abs(np.einsum("ii", one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    one_int = np.vstack(
        (np.hstack((one_int, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), one_int)))
    )
    # alpha alpha alpha alpha, beta alpha alpha alpha
    # alpha beta alpha alpha, beta beta alpha alpha
    temp1 = np.concatenate(
        (
            np.concatenate((two_int, np.zeros((6, 6, 6, 6))), axis=0),
            np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))), axis=0),
        ),
        axis=1,
    )
    # alpha alpha beta alpha, beta alpha beta alpha
    # alpha beta beta alpha, beta beta beta alpha
    temp2 = np.concatenate(
        (
            np.concatenate((np.zeros((6, 6, 6, 6)), two_int), axis=0),
            np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))), axis=0),
        ),
        axis=1,
    )
    # alpha alpha alpha beta, beta alpha alpha beta
    # alpha beta alpha beta, beta beta alpha beta
    temp3 = np.concatenate(
        (
            np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))), axis=0),
            np.concatenate((two_int, np.zeros((6, 6, 6, 6))), axis=0),
        ),
        axis=1,
    )
    # alpha alpha beta beta, beta alpha beta beta
    # alpha beta beta beta, beta beta beta beta
    temp4 = np.concatenate(
        (
            np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))), axis=0),
            np.concatenate((np.zeros((6, 6, 6, 6)), two_int), axis=0),
        ),
        axis=1,
    )
    two_int = np.concatenate(
        (np.concatenate((temp1, temp2), axis=2), np.concatenate((temp3, temp4), axis=2)), axis=3
    )

    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,ijkl", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )
    # chemist notation
    one_density, two_density = density_matrix(
        sd_coeffs, civec, 6, is_chemist_notation=True, val_threshold=0, orbtype="generalized"
    )
    assert (
        abs(
            (
                np.einsum("ij,ij", one_int, one_density[0])
                + 0.5 * np.einsum("ijkl,iklj", two_int, two_density[0])
            )
            - (energy)
        )
        < 1e-8
    )