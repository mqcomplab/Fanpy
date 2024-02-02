import fanpy.tools.slater as slater
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
import itertools as it
import numpy as np

from fanpy.upgrades.cext_sign import sign_excite_one, sign_excite_two, sign_excite_two_ab



def _integrate_sd_sds_one_alpha(self, occ_alpha, occ_beta, vir_alpha):
    """Return the integrals of the given Slater determinant with its first order excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(3, M)
        Integrals of the given Slater determinant with its first order excitations involving the
        alpha spin orbitals.
        First index corresponds to the one-electron (first element), coulomb (second element),
        and exchange (third element) integrals.
        Second index corresponds to the first order excitations of the given Slater determinant.
        The excitations are ordered by the occupied orbital then the virtual orbital. For
        example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
        excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
        excitations of the given Slater determinants.

    """
    shared_alpha = slater.shared_indices_remove_one_index(occ_alpha)

    sign_a = np.array(sign_excite_one(occ_alpha, vir_alpha))

    one_electron_a = self.one_int[occ_alpha[:, np.newaxis], vir_alpha[np.newaxis, :]].ravel()

    coulomb_a = np.sum(
        self.two_int[
            shared_alpha[:, :, np.newaxis],
            occ_alpha[:, np.newaxis, np.newaxis],
            shared_alpha[:, :, np.newaxis],
            vir_alpha[np.newaxis, np.newaxis, :],
        ],
        axis=1,
    ).ravel()
    coulomb_a += np.sum(
        self.two_int[
            occ_alpha[:, np.newaxis, np.newaxis],
            occ_beta[np.newaxis, :, np.newaxis],
            vir_alpha[np.newaxis, np.newaxis, :],
            occ_beta[np.newaxis, :, np.newaxis],
        ],
        axis=1,
    ).ravel()

    exchange_a = -np.sum(
        self.two_int[
            shared_alpha[:, :, np.newaxis],
            occ_alpha[:, np.newaxis, np.newaxis],
            vir_alpha[np.newaxis, np.newaxis, :],
            shared_alpha[:, :, np.newaxis],
        ],
        axis=1,
    ).ravel()

    return sign_a[None, :] * np.array([one_electron_a, coulomb_a, exchange_a])


def _integrate_sd_sds_one_beta(self, occ_alpha, occ_beta, vir_beta):
    """Return the integrals of the given Slater determinant with its first order excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(3, M)
        Integrals of the given Slater determinant with its first order excitations involving the
        beta spin orbitals.
        First index corresponds to the one-electron (first element), coulomb (second element),
        and exchange (third element) integrals.
        Second index corresponds to the first order excitations of the given Slater determinant.
        The excitations are ordered by the occupied orbital then the virtual orbital. For
        example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
        excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
        excitations of the given Slater determinants.

    """
    shared_beta = slater.shared_indices_remove_one_index(occ_beta)

    sign_b = np.array(sign_excite_one(occ_beta, vir_beta))

    one_electron_b = self.one_int[occ_beta[:, np.newaxis], vir_beta[np.newaxis, :]].ravel()

    coulomb_b = np.sum(
        self.two_int[
            shared_beta[:, :, np.newaxis],
            occ_beta[:, np.newaxis, np.newaxis],
            shared_beta[:, :, np.newaxis],
            vir_beta[np.newaxis, np.newaxis, :],
        ],
        axis=1,
    ).ravel()
    coulomb_b += np.sum(
        self.two_int[
            occ_alpha[np.newaxis, :, np.newaxis],
            occ_beta[:, np.newaxis, np.newaxis],
            occ_alpha[np.newaxis, :, np.newaxis],
            vir_beta[np.newaxis, np.newaxis, :],
        ],
        axis=1,
    ).ravel()

    exchange_b = -np.sum(
        self.two_int[
            shared_beta[:, :, np.newaxis],
            occ_beta[:, np.newaxis, np.newaxis],
            vir_beta[np.newaxis, np.newaxis, :],
            shared_beta[:, :, np.newaxis],
        ],
        axis=1,
    ).ravel()

    return sign_b[None, :] * np.array([one_electron_b, coulomb_b, exchange_b])


def _integrate_sd_sds_two_aa(self, occ_alpha, occ_beta, vir_alpha):
    """Return the integrals of a Slater determinant with its second order (alpha) excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(2, M)
        Integrals of the given Slater determinant with its second order excitations involving
        the alpha spin orbitals.
        First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=C0103

    annihilators = np.array(list(it.combinations(occ_alpha, 2)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.combinations(vir_alpha, 2)))
    c = creators[:, 0]
    d = creators[:, 1]

    sign = np.array(sign_excite_two(occ_alpha, vir_alpha))

    coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
    exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

    return sign[None, :] * np.array([coulomb, exchange])


def _integrate_sd_sds_two_ab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
    """Return the integrals of a SD with its second order (alpha and beta) excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(M,)
        Coulomb integrals of the given Slater determinant with its second order excitations
        involving both alpha and beta orbitals
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=C0103

    annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.product(vir_alpha, vir_beta)))
    c = creators[:, 0]
    d = creators[:, 1]

    sign = np.array(sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta))
    coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()

    return sign * coulomb


def _integrate_sd_sds_two_bb(self, occ_alpha, occ_beta, vir_beta):
    """Return the integrals of a Slater determinant with its second order (beta) excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(2, M)
        Integrals of the given Slater determinant with its second order excitations involving
        the beta spin orbitals.
        First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=C0103

    annihilators = np.array(list(it.combinations(occ_beta, 2)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.combinations(vir_beta, 2)))
    c = creators[:, 0]
    d = creators[:, 1]

    sign = np.array(sign_excite_two(occ_beta, vir_beta))
    coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
    exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

    return sign[None, :] * np.array([coulomb, exchange])


def _integrate_sd_sds_deriv_one_aa(self, occ_alpha, occ_beta, vir_alpha):
    """Return (alpha) derivative of integrals an SD with its (alpha) single excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(3, N_params, M)
        Derivatives of the integrals of the given Slater determinant with its first order
        excitations of alpha orbitals with respect to parameters associated with alpha orbitals.
        First index of the numpy array corresponds to the one-electron (first element), coulomb
        (second elements), and exchange (third element) integrals.
        Second index of the numpy array corresponds to index of the parameter with respect to
        which the integral is derivatived.
        Third index of the numpy array corresponds to the first order excitations of the given
        Slater determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
        ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
        of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915
    nspatial = self.nspin // 2
    all_alpha = np.arange(nspatial)

    sign_a = np.array(sign_excite_one(occ_alpha, vir_alpha))

    occ_alpha_array_indices = np.arange(occ_alpha.size)
    vir_alpha_array_indices = np.arange(vir_alpha.size)

    shared_alpha = slater.shared_indices_remove_one_index(occ_alpha)

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation
    # the third index corresponds to the occupied orbital that will be annihilated in the
    # excitation
    # the fourth index corresponds to the occupied orbital that will be created in the
    # excitation
    # FIXME: hardcoded parameter shape
    one_electron_a = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))
    coulomb_a = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))
    exchange_a = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))

    # ONE-ELECTRON INTEGRALS
    # x == a
    one_electron_a[
        occ_alpha[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= self.one_int[
        all_alpha[None, :, None], vir_alpha[None, None, :]  # y, b
    ]
    # x == b
    one_electron_a[
        vir_alpha[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= self.one_int[
        occ_alpha[:, None, None], all_alpha[None, :, None]  # a, y
    ]
    # y == a
    one_electron_a[
        all_alpha[:, None, None],  # x
        occ_alpha[None, :, None],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += self.one_int[
        all_alpha[:, None, None], vir_alpha[None, None, :]  # x, b
    ]
    # y == b
    one_electron_a[
        all_alpha[:, None, None],  # x
        vir_alpha[None, None, :],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += self.one_int[
        occ_alpha[None, :, None], all_alpha[:, None, None]  # a, x
    ]

    # COULOMB INTEGRALS
    # x == a
    coulomb_a[
        occ_alpha[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            all_alpha[None, None, :, None],  # y
            shared_alpha[:, :, None, None],  # shared alpha
            vir_alpha[None, None, None, :],  # b
            shared_alpha[:, :, None, None],  # shared alpha
        ],
        axis=1,
    )
    coulomb_a[
        occ_alpha[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            all_alpha[None, None, :, None],  # y
            occ_beta[None, :, None, None],  # shared beta
            vir_alpha[None, None, None, :],  # b
            occ_beta[None, :, None, None],  # shared beta
        ],
        axis=1,
    )
    # x == b
    coulomb_a[
        vir_alpha[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            occ_alpha[:, None, None, None],  # a
            shared_alpha[:, :, None, None],  # shared alpha
            all_alpha[None, None, :, None],  # y
            shared_alpha[:, :, None, None],  # shared alpha
        ],
        axis=1,
    )
    coulomb_a[
        vir_alpha[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            occ_alpha[:, None, None, None],  # a
            occ_beta[None, :, None, None],  # shared beta
            all_alpha[None, None, :, None],  # y
            occ_beta[None, :, None, None],  # shared beta
        ],
        axis=1,
    )
    # x in shared
    coulomb_a[
        shared_alpha[:, :, None, None],  # x
        all_alpha[None, None, :, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        shared_alpha[:, :, None, None],  # x
        occ_alpha[:, None, None, None],  # a (occupied index)
        all_alpha[None, None, :, None],  # y
        vir_alpha[None, None, None, :],  # b (virtual index)
    ]
    coulomb_a[
        shared_alpha[:, :, None, None],  # x
        all_alpha[None, None, :, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        shared_alpha[:, :, None, None],  # x
        vir_alpha[None, None, None, :],  # b (virtual index)
        all_alpha[None, None, :, None],  # y
        occ_alpha[:, None, None, None],  # a (occupied index)
    ]
    # y == a
    coulomb_a[
        all_alpha[:, None, None],  # x
        occ_alpha[None, :, None],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            all_alpha[:, None, None, None],  # x
            shared_alpha.T[None, :, :, None],  # shared alpha
            vir_alpha[None, None, None, :],  # b
            shared_alpha.T[None, :, :, None],  # shared alpha
        ],
        axis=1,
    )
    coulomb_a[
        all_alpha[:, None, None],  # x
        occ_alpha[None, :, None],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            all_alpha[:, None, None, None],  # x
            occ_beta[None, :, None, None],  # shared beta
            vir_alpha[None, None, None, :],  # b
            occ_beta[None, :, None, None],  # shared beta
        ],
        axis=1,
    )
    # y == b
    coulomb_a[
        all_alpha[:, None, None],  # x
        vir_alpha[None, None, :],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            occ_alpha[None, None, :, None],  # a
            shared_alpha.T[None, :, :, None],  # shared alpha
            all_alpha[:, None, None, None],  # x
            shared_alpha.T[None, :, :, None],  # shared alpha
        ],
        axis=1,
    )
    coulomb_a[
        all_alpha[:, None, None],  # x
        vir_alpha[None, None, :],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            occ_alpha[None, None, :, None],  # a
            occ_beta[None, :, None, None],  # shared beta
            all_alpha[:, None, None, None],  # x
            occ_beta[None, :, None, None],  # shared beta
        ],
        axis=1,
    )
    # y in shared
    coulomb_a[
        all_alpha[None, None, :, None],  # x
        shared_alpha[:, :, None, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        all_alpha[None, None, :, None],  # x
        occ_alpha[:, None, None, None],  # a (occupied index)
        shared_alpha[:, :, None, None],  # y
        vir_alpha[None, None, None, :],  # b (virtual index)
    ]
    coulomb_a[
        all_alpha[None, None, :, None],  # x
        shared_alpha[:, :, None, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        all_alpha[None, None, :, None],  # x
        vir_alpha[None, None, None, :],  # b (virtual index)
        shared_alpha[:, :, None, None],  # y
        occ_alpha[:, None, None, None],  # a (occupied index)
    ]

    # EXCHANGE INTEGRALS
    # x == a
    exchange_a[
        occ_alpha[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            all_alpha[None, None, :, None],  # y
            shared_alpha[:, :, None, None],  # shared alpha
            shared_alpha[:, :, None, None],  # shared alpha
            vir_alpha[None, None, None, :],  # b
        ],
        axis=1,
    )
    # x == b
    exchange_a[
        vir_alpha[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_alpha_array_indices[:, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            occ_alpha[:, None, None, None],  # a
            shared_alpha[:, :, None, None],  # shared alpha
            shared_alpha[:, :, None, None],  # shared alpha
            all_alpha[None, None, :, None],  # y
        ],
        axis=1,
    )
    # x in shared
    exchange_a[
        shared_alpha[:, :, None, None],  # x
        all_alpha[None, None, :, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        shared_alpha[:, :, None, None],  # x
        occ_alpha[:, None, None, None],  # a (occupied index)
        vir_alpha[None, None, None, :],  # b (virtual index)
        all_alpha[None, None, :, None],  # y
    ]
    exchange_a[
        shared_alpha[:, :, None, None],  # x
        all_alpha[None, None, :, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        shared_alpha[:, :, None, None],  # x
        vir_alpha[None, None, None, :],  # b (virtual index)
        occ_alpha[:, None, None, None],  # a (occupied index)
        all_alpha[None, None, :, None],  # y
    ]
    # y == a
    exchange_a[
        all_alpha[:, None, None],  # x
        occ_alpha[None, :, None],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            all_alpha[:, None, None, None],  # x
            shared_alpha.T[None, :, :, None],  # shared alpha
            shared_alpha.T[None, :, :, None],  # shared alpha
            vir_alpha[None, None, None, :],  # b
        ],
        axis=1,
    )
    # y == b
    exchange_a[
        all_alpha[:, None, None],  # x
        vir_alpha[None, None, :],  # y
        occ_alpha_array_indices[None, :, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            occ_alpha[None, None, :, None],  # a
            shared_alpha.T[None, :, :, None],  # shared alpha
            shared_alpha.T[None, :, :, None],  # shared alpha
            all_alpha[:, None, None, None],  # x
        ],
        axis=1,
    )
    # y in shared
    exchange_a[
        all_alpha[None, None, :, None],  # x
        shared_alpha[:, :, None, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        all_alpha[None, None, :, None],  # x
        occ_alpha[:, None, None, None],  # a (occupied index)
        vir_alpha[None, None, None, :],  # b (virtual index)
        shared_alpha[:, :, None, None],  # y
    ]
    exchange_a[
        all_alpha[None, None, :, None],  # x
        shared_alpha[:, :, None, None],  # y
        occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        all_alpha[None, None, :, None],  # x
        vir_alpha[None, None, None, :],  # b (virtual index)
        occ_alpha[:, None, None, None],  # a (occupied index)
        shared_alpha[:, :, None, None],  # y
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
    return sign_a[None, None, :] * np.array(
        [
            one_electron_a[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            coulomb_a[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            exchange_a[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
        ]
    )


def _integrate_sd_sds_deriv_one_ba(self, occ_alpha, occ_beta, vir_alpha):
    """Return (beta) derivative of integrals of the a SD with its (alpha) single excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(N_params, M)
        Derivatives of the coulomb integrals of the given Slater determinant with its first
        order excitations of alpha orbitals with respect to parameters associated with beta
        orbitals.
        Second index of the numpy array corresponds to index of the parameter with respect to
        which the integral is derivatived.
        Third index of the numpy array corresponds to the first order excitations of the given
        Slater determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
        ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
        of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915
    nspatial = self.nspin // 2
    all_beta = np.arange(nspatial)
    occ_alpha_array_indices = np.arange(occ_alpha.size)
    vir_alpha_array_indices = np.arange(vir_alpha.size)

    sign_a = np.array(sign_excite_one(occ_alpha, vir_alpha))

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation
    # the third index corresponds to the occupied orbital that will be annihilated in the
    # excitation
    # the fourth index corresponds to the occupied orbital that will be created in the
    # excitation
    # FIXME: hardcoded parameter shape
    coulomb_ba = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))

    #
    coulomb_ba[
        occ_beta[:, None, None, None],  # x
        all_beta[None, None, None, :, None],  # y
        occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        occ_alpha[None, None, :, None, None],  # a (occupied index)
        occ_beta[:, None, None, None],  # x
        vir_alpha[None, None, None, None, :],  # b (virtual index)
        all_beta[None, None, None, :, None],  # y
    ]
    coulomb_ba[
        occ_beta[:, None, None, None],  # x
        all_beta[None, None, None, :, None],  # y
        occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        vir_alpha[None, None, None, None, :],  # b (virtual index)
        occ_beta[:, None, None, None],  # x
        occ_alpha[None, None, :, None, None],  # a (occupied index)
        all_beta[None, None, None, :, None],  # y
    ]
    #
    coulomb_ba[
        all_beta[None, None, None, :, None],  # x
        occ_beta[:, None, None, None],  # y
        occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        occ_alpha[None, None, :, None, None],  # a (occupied index)
        all_beta[None, None, None, :, None],  # x
        vir_alpha[None, None, None, None, :],  # b (virtual index)
        occ_beta[:, None, None, None],  # y
    ]
    coulomb_ba[
        all_beta[None, None, None, :, None],  # x
        occ_beta[:, None, None, None],  # y
        occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
        vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        vir_alpha[None, None, None, None, :],  # b (virtual index)
        all_beta[None, None, None, :, None],  # x
        occ_alpha[None, None, :, None, None],  # a (occupied index)
        occ_beta[:, None, None, None],  # y
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
    return sign_a[None, :] * coulomb_ba[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)


def _integrate_sd_sds_deriv_one_ab(self, occ_alpha, occ_beta, vir_beta):
    """Return (alpha) derivative of integrals of the a SD with its (beta) single excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(N_params, M)
        Derivatives of the coulomb integrals of the given Slater determinant with its first
        order excitations of beta orbitals with respect to parameters associated with alpha
        orbitals.
        Second index of the numpy array corresponds to index of the parameter with respect to
        which the integral is derivatived.
        Third index of the numpy array corresponds to the first order excitations of the given
        Slater determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
        ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
        of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915
    nspatial = self.nspin // 2
    all_alpha = np.arange(nspatial)

    occ_beta_array_indices = np.arange(occ_beta.size)
    vir_beta_array_indices = np.arange(vir_beta.size)

    sign_b = np.array(sign_excite_one(occ_beta, vir_beta))

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation
    # the third index corresponds to the occupied orbital that will be annihilated in the
    # excitation
    # the fourth index corresponds to the occupied orbital that will be created in the
    # excitation
    # FIXME: hardcoded parameter shape
    coulomb_ab = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))

    #
    coulomb_ab[
        occ_alpha[:, None, None, None],  # x
        all_alpha[None, None, :, None],  # y
        occ_beta_array_indices[None, :, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        occ_alpha[:, None, None, None],  # x
        occ_beta[None, :, None, None],  # a (occupied index)
        all_alpha[None, None, :, None],  # y
        vir_beta[None, None, None, :],  # b (virtual index)
    ]
    coulomb_ab[
        occ_alpha[:, None, None, None],  # x
        all_alpha[None, None, :, None],  # y
        occ_beta_array_indices[None, :, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        occ_alpha[:, None, None, None],  # x
        vir_beta[None, None, None, :],  # b (virtual index)
        all_alpha[None, None, :, None],  # y
        occ_beta[None, :, None, None],  # a (occupied index)
    ]
    #
    coulomb_ab[
        all_alpha[None, None, None, :, None],  # x
        occ_alpha[:, None, None, None],  # y
        occ_beta_array_indices[None, :, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        all_alpha[None, None, :, None],  # x
        occ_beta[None, :, None, None],  # a (occupied index)
        occ_alpha[:, None, None, None],  # y
        vir_beta[None, None, None, :],  # b (virtual index)
    ]
    coulomb_ab[
        all_alpha[None, None, :, None],  # x
        occ_alpha[:, None, None, None],  # y
        occ_beta_array_indices[None, :, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        all_alpha[None, None, :, None],  # x
        vir_beta[None, None, None, :],  # b (virtual index)
        occ_alpha[:, None, None, None],  # y
        occ_beta[None, :, None, None],  # a (occupied index)
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)

    return sign_b[None, :] * coulomb_ab[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)


def _integrate_sd_sds_deriv_one_bb(self, occ_alpha, occ_beta, vir_beta):
    """Return (beta) derivative of integrals an SD with its single excitations of beta orbitals.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(3, N_params, M)
        Derivatives of the integrals of the given Slater determinant with its first order
        excitations of beta orbitals with respect to parameters associated with beta orbitals.
        First index of the numpy array corresponds to the one-electron (first element), coulomb
        (second elements), and exchange (third element) integrals.
        Second index of the numpy array corresponds to index of the parameter with respect to
        which the integral is derivatived.
        Third index of the numpy array corresponds to the first order excitations of the given
        Slater determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
        ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
        of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915
    nspatial = self.nspin // 2
    all_beta = np.arange(nspatial)

    occ_beta_array_indices = np.arange(occ_beta.size)
    vir_beta_array_indices = np.arange(vir_beta.size)

    shared_beta = slater.shared_indices_remove_one_index(occ_beta)

    sign_b = np.array(sign_excite_one(occ_beta, vir_beta))

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation
    # the third index corresponds to the occupied orbital that will be annihilated in the
    # excitation
    # the fourth index corresponds to the occupied orbital that will be created in the
    # excitation
    # FIXME: hardcoded parameter shape
    one_electron_b = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))
    coulomb_b = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))
    exchange_b = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))

    # ONE-ELECTRON INTEGRALS
    # x == a
    one_electron_b[
        occ_beta[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= self.one_int[
        all_beta[None, :, None], vir_beta[None, None, :]  # y, b
    ]
    # x == b
    one_electron_b[
        vir_beta[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= self.one_int[
        occ_beta[:, None, None], all_beta[None, :, None]  # a, y
    ]
    # y == a
    one_electron_b[
        all_beta[:, None, None],  # x
        occ_beta[None, :, None],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += self.one_int[
        all_beta[:, None, None], vir_beta[None, None, :]  # x, b
    ]
    # y == b
    one_electron_b[
        all_beta[:, None, None],  # x
        vir_beta[None, None, :],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += self.one_int[
        occ_beta[None, :, None], all_beta[:, None, None]  # a, x
    ]

    # COULOMB INTEGRALS
    # x == a
    coulomb_b[
        occ_beta[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            all_beta[None, None, :, None],  # y
            shared_beta[:, :, None, None],  # shared beta
            vir_beta[None, None, None, :],  # b
            shared_beta[:, :, None, None],  # shared beta
        ],
        axis=1,
    )
    coulomb_b[
        occ_beta[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            occ_alpha[None, :, None, None],  # shared alpha
            all_beta[None, None, :, None],  # y
            occ_alpha[None, :, None, None],  # shared alpha
            vir_beta[None, None, None, :],  # b
        ],
        axis=1,
    )
    # x == b
    coulomb_b[
        vir_beta[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            occ_beta[:, None, None, None],  # a
            shared_beta[:, :, None, None],  # shared beta
            all_beta[None, None, :, None],  # y
            shared_beta[:, :, None, None],  # shared beta
        ],
        axis=1,
    )
    coulomb_b[
        vir_beta[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            occ_alpha[None, :, None, None],  # shared alpha
            occ_beta[:, None, None, None],  # a
            occ_alpha[None, :, None, None],  # shared alpha
            all_beta[None, None, :, None],  # y
        ],
        axis=1,
    )
    # x in shared
    coulomb_b[
        shared_beta[:, :, None, None],  # x
        all_beta[None, None, :, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        shared_beta[:, :, None, None],  # x
        occ_beta[:, None, None, None],  # a (occupied index)
        all_beta[None, None, :, None],  # y
        vir_beta[None, None, None, :],  # b (virtual index)
    ]
    coulomb_b[
        shared_beta[:, :, None, None],  # x
        all_beta[None, None, :, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        shared_beta[:, :, None, None],  # x
        vir_beta[None, None, None, :],  # b (virtual index)
        all_beta[None, None, :, None],  # y
        occ_beta[:, None, None, None],  # a (occupied index)
    ]
    #
    # y == a
    coulomb_b[
        all_beta[:, None, None],  # x
        occ_beta[None, :, None],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            all_beta[:, None, None, None],  # x
            shared_beta.T[None, :, :, None],  # shared beta
            vir_beta[None, None, None, :],  # b
            shared_beta.T[None, :, :, None],  # shared beta
        ],
        axis=1,
    )
    coulomb_b[
        all_beta[:, None, None],  # x
        occ_beta[None, :, None],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            occ_alpha[None, :, None, None],  # shared alpha
            all_beta[:, None, None, None],  # x
            occ_alpha[None, :, None, None],  # shared alpha
            vir_beta[None, None, None, :],  # b
        ],
        axis=1,
    )
    # y == b
    coulomb_b[
        all_beta[:, None, None],  # x
        vir_beta[None, None, :],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            occ_beta[None, None, :, None],  # a
            shared_beta.T[None, :, :, None],  # shared beta
            all_beta[:, None, None, None],  # x
            shared_beta.T[None, :, :, None],  # shared beta
        ],
        axis=1,
    )
    coulomb_b[
        all_beta[:, None, None],  # x
        vir_beta[None, None, :],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            occ_alpha[None, :, None, None],  # shared alpha
            occ_beta[None, None, :, None],  # a
            occ_alpha[None, :, None, None],  # shared alpha
            all_beta[:, None, None, None],  # x
        ],
        axis=1,
    )
    # y in shared
    coulomb_b[
        all_beta[None, None, :, None],  # x
        shared_beta[:, :, None, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        all_beta[None, None, :, None],  # x
        occ_beta[:, None, None, None],  # a (occupied index)
        shared_beta[:, :, None, None],  # y
        vir_beta[None, None, None, :],  # b (virtual index)
    ]
    coulomb_b[
        all_beta[None, None, :, None],  # x
        shared_beta[:, :, None, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        all_beta[None, None, :, None],  # x
        vir_beta[None, None, None, :],  # b (virtual index)
        shared_beta[:, :, None, None],  # y
        occ_beta[:, None, None, None],  # a (occupied index)
    ]
    #

    # EXCHANGE INTEGRALS
    # x == a
    exchange_b[
        occ_beta[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            all_beta[None, None, :, None],  # y
            shared_beta[:, :, None, None],  # shared beta
            shared_beta[:, :, None, None],  # shared beta
            vir_beta[None, None, None, :],  # b
        ],
        axis=1,
    )
    # x == b
    exchange_b[
        vir_beta[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_beta_array_indices[:, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] += np.sum(
        self.two_int[
            occ_beta[:, None, None, None],  # a
            shared_beta[:, :, None, None],  # shared beta
            shared_beta[:, :, None, None],  # shared beta
            all_beta[None, None, :, None],  # y
        ],
        axis=1,
    )
    # x in shared
    exchange_b[
        shared_beta[:, :, None, None],  # x
        all_beta[None, None, :, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        shared_beta[:, :, None, None],  # x
        occ_beta[:, None, None, None],  # a (occupied index)
        vir_beta[None, None, None, :],  # b (virtual index)
        all_beta[None, None, :, None],  # y
    ]
    exchange_b[
        shared_beta[:, :, None, None],  # x
        all_beta[None, None, :, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] += self.two_int[
        shared_beta[:, :, None, None],  # x
        vir_beta[None, None, None, :],  # b (virtual index)
        occ_beta[:, None, None, None],  # a (occupied index)
        all_beta[None, None, :, None],  # y
    ]
    # y == a
    exchange_b[
        all_beta[:, None, None],  # x
        occ_beta[None, :, None],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            all_beta[:, None, None, None],  # x
            shared_beta.T[None, :, :, None],  # shared beta
            shared_beta.T[None, :, :, None],  # shared beta
            vir_beta[None, None, None, :],  # b
        ],
        axis=1,
    )
    # y == b
    exchange_b[
        all_beta[:, None, None],  # x
        vir_beta[None, None, :],  # y
        occ_beta_array_indices[None, :, None],  # a (occupied index)
        vir_beta_array_indices[None, None, :],  # b (virtual index)
    ] -= np.sum(
        self.two_int[
            occ_beta[None, None, :, None],  # a
            shared_beta.T[None, :, :, None],  # shared beta
            shared_beta.T[None, :, :, None],  # shared beta
            all_beta[:, None, None, None],  # x
        ],
        axis=1,
    )
    # y in shared
    exchange_b[
        all_beta[None, None, :, None],  # x
        shared_beta[:, :, None, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        all_beta[None, None, :, None],  # x
        occ_beta[:, None, None, None],  # a (occupied index)
        vir_beta[None, None, None, :],  # b (virtual index)
        shared_beta[:, :, None, None],  # y
    ]
    exchange_b[
        all_beta[None, None, :, None],  # x
        shared_beta[:, :, None, None],  # y
        occ_beta_array_indices[:, None, None, None],  # a (occupied index)
        vir_beta_array_indices[None, None, None, :],  # b (virtual index)
    ] -= self.two_int[
        all_beta[None, None, :, None],  # x
        vir_beta[None, None, None, :],  # b (virtual index)
        occ_beta[:, None, None, None],  # a (occupied index)
        shared_beta[:, :, None, None],  # y
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
    return sign_b[None, None, :] * np.array(
        [
            one_electron_b[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            coulomb_b[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            exchange_b[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
        ]
    )


def _integrate_sd_sds_deriv_two_aaa(self, occ_alpha, occ_beta, vir_alpha):
    """Return (alpha) derivatives of integrals of an SD and its (alpha) double excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(2, N_params, M)
        Derivatives of the coulomb and exchange integrals (with respect to parameters associated
        with alpha orbitals) of the given Slater determinant with its second order excitations
        involving only alpha orbitals.
        First index of the numpy array corresponds to the one-electron (first element), coulomb
        (second elements), and exchange (third element) integrals.
        Second index corresponds to index of the parameter with respect to which the integral is
        derivatived.
        Third index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915,C0103
    nspatial = self.nspin // 2
    all_alpha = np.arange(nspatial)

    annihilators = np.array(list(it.combinations(occ_alpha, 2)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.combinations(vir_alpha, 2)))
    c = creators[:, 0]
    d = creators[:, 1]
    occ_array_indices = np.arange(a.size)
    vir_array_indices = np.arange(c.size)

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation
    # the third index corresponds to the occupied orbital that will be annihilated in the
    # excitation
    # the fourth index corresponds to the occupied orbital that will be created in the
    # excitation
    coulomb_aa = np.zeros((nspatial, nspatial, a.size, c.size))
    exchange_aa = np.zeros((nspatial, nspatial, a.size, c.size))

    # FIXME: may need to make a cythonized function for this
    sign_aa = np.array(sign_excite_two(occ_alpha, vir_alpha))

    # x == a
    coulomb_aa[
        a[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        all_alpha[None, :, None],  # y
        b[:, None, None],  # b
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_aa[
        a[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        all_alpha[None, :, None],  # y
        b[:, None, None],  # b
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # x == b
    coulomb_aa[
        b[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        all_alpha[None, :, None],  # y
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_aa[
        b[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        all_alpha[None, :, None],  # y
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # x == c
    coulomb_aa[
        c[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_alpha[None, :, None],  # y
        d[None, None, :],  # d
    ]
    exchange_aa[
        c[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        d[None, None, :],  # d
        all_alpha[None, :, None],  # y
    ]
    # x == d
    coulomb_aa[
        d[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        c[None, None, :],  # c
        all_alpha[None, :, None],  # y
    ]
    exchange_aa[
        d[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_alpha[None, :, None],  # y
        c[None, None, :],  # c
    ]

    # y == a
    coulomb_aa[
        all_alpha[None, :, None],  # x
        a[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        all_alpha[None, :, None],  # x
        b[:, None, None],  # b
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_aa[
        all_alpha[None, :, None],  # x
        a[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        all_alpha[None, :, None],  # x
        b[:, None, None],  # b
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # y == b
    coulomb_aa[
        all_alpha[None, :, None],  # x
        b[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        all_alpha[None, :, None],  # x
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_aa[
        all_alpha[None, :, None],  # x
        b[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        all_alpha[None, :, None],  # x
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # y == c
    coulomb_aa[
        all_alpha[None, :, None],  # x
        c[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_alpha[None, :, None],  # x
        d[None, None, :],  # d
    ]
    exchange_aa[
        all_alpha[None, :, None],  # x
        c[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        d[None, None, :],  # d
        all_alpha[None, :, None],  # x
    ]
    # y == d
    coulomb_aa[
        all_alpha[None, :, None],  # x
        d[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        c[None, None, :],  # c
        all_alpha[None, :, None],  # x
    ]
    exchange_aa[
        all_alpha[None, :, None],  # x
        d[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_alpha[None, :, None],  # x
        c[None, None, :],  # c
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
    return sign_aa[None, None, :] * np.array(
        [
            coulomb_aa[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            exchange_aa[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
        ]
    )


def _integrate_sd_sds_deriv_two_aab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
    """Return (alpha) derivatives of integrals of an SD and its (alpha beta) double excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(N_params, M)
        Derivatives of the coulomb integrals (with respect to parameters associated with alpha
        orbitals) of the given Slater determinant with its second order excitations involving
        alpha and beta orbitals.
        First index corresponds to index of the parameter with respect to which the integral is
        derivatived.
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915,C0103
    nspatial = self.nspin // 2
    all_alpha = np.arange(nspatial)

    annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.product(vir_alpha, vir_beta)))
    c = creators[:, 0]
    d = creators[:, 1]
    occ_array_indices = np.arange(a.size)
    vir_array_indices = np.arange(c.size)

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation
    # the third index corresponds to the occupied orbital that will be annihilated in the
    # excitation
    # the fourth index corresponds to the occupied orbital that will be created in the
    # excitation
    coulomb_ab = np.zeros((nspatial, nspatial, a.size, c.size))

    sign_ab = np.array(sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta))

    # x == a
    coulomb_ab[
        a[:, None, None],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        all_alpha[None, :, None],  # y
        b[:, None, None],  # b
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    # x == c
    coulomb_ab[
        c[None, None, :],  # x
        all_alpha[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_alpha[None, :, None],  # y
        d[None, None, :],  # d
    ]

    # y == a
    coulomb_ab[
        all_alpha[None, :, None],  # x
        a[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        all_alpha[None, :, None],  # x
        b[:, None, None],  # b
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    # y == c
    coulomb_ab[
        all_alpha[None, :, None],  # x
        c[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_alpha[None, :, None],  # x
        d[None, None, :],  # d
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
    return sign_ab[None, :] * coulomb_ab[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)


def _integrate_sd_sds_deriv_two_bab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
    """Return (beta) derivatives of integrals of an SD and its (alpha, beta) double excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(N_params, M)
        Derivatives of the coulomb integrals (with respect to parameters associated with beta
        orbitals) of the given Slater determinant with its second order excitations involving
        alpha and beta orbitals.
        First index corresponds to index of the parameter with respect to which the integral is
        derivatived.
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915,C0103
    nspatial = self.nspin // 2
    all_beta = np.arange(nspatial)
    occ_indices = np.hstack(
        [
            slater.spatial_to_spin_indices(occ_alpha, nspatial, to_beta=False),
            slater.spatial_to_spin_indices(occ_beta, nspatial, to_beta=True),
        ]
    )

    annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.product(vir_alpha, vir_beta)))
    c = creators[:, 0]
    d = creators[:, 1]
    occ_array_indices = np.arange(a.size)
    vir_array_indices = np.arange(c.size)

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation
    # the third index corresponds to the occupied orbital that will be annihilated in the
    # excitation
    # the fourth index corresponds to the virtual orbital that will be created in the excitation
    coulomb_ba = np.zeros((nspatial, nspatial, b.size, d.size))

    sign_ab = np.array(sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta))

    # x == b
    coulomb_ba[
        b[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        all_beta[None, :, None],  # y
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    # x == d
    coulomb_ba[
        d[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        c[None, None, :],  # c
        all_beta[None, :, None],  # y
    ]

    # y == b
    coulomb_ba[
        all_beta[None, :, None],  # x
        b[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        all_beta[None, :, None],  # x
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    # y == d
    coulomb_ba[
        all_beta[None, :, None],  # x
        d[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        c[None, None, :],  # c
        all_beta[None, :, None],  # x
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
    return sign_ab[None, :] * coulomb_ba[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)


def _integrate_sd_sds_deriv_two_bbb(self, occ_alpha, occ_beta, vir_beta):
    """Return (beta) derivatives of integrals of an SD and its (beta) double excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(2, N_params, M)
        Derivatives of the coulomb and exchange integrals (with respect to parameters associated
        with beta orbitals) of the given Slater determinant with its second order excitations
        involving only beta orbitals.
        First index of the numpy array corresponds to the one-electron (first element), coulomb
        (second elements), and exchange (third element) integrals.
        Second index corresponds to index of the parameter with respect to which the integral is
        derivatived.
        Third index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=R0915,C0103
    nspatial = self.nspin // 2
    all_beta = np.arange(nspatial)

    annihilators = np.array(list(it.combinations(occ_beta, 2)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.combinations(vir_beta, 2)))
    c = creators[:, 0]
    d = creators[:, 1]
    occ_array_indices = np.arange(a.size)
    vir_array_indices = np.arange(c.size)

    sign_bb = np.array(sign_excite_two(occ_beta, vir_beta))

    # NOTE: here, we use the following convention for indices:
    # the first index corresponds to the row index of the antihermitian matrix for orbital
    # rotation of the beta orbitals
    # the second index corresponds to the column index of the antihermitian matrix for orbital
    # rotation of the beta orbitals
    # the third index corresponds to the occupied orbitals that will be annihilated in the
    # excitation
    # the fourth index corresponds to the virtual orbitals that will be created in the
    # excitation
    coulomb_bb = np.zeros((nspatial, nspatial, a.size, c.size))
    exchange_bb = np.zeros((nspatial, nspatial, a.size, c.size))

    # x == a
    coulomb_bb[
        a[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        all_beta[None, :, None],  # y
        b[:, None, None],  # b
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_bb[
        a[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        all_beta[None, :, None],  # y
        b[:, None, None],  # b
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # x == b
    coulomb_bb[
        b[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        all_beta[None, :, None],  # y
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_bb[
        b[:, None, None],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        all_beta[None, :, None],  # y
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # x == c
    coulomb_bb[
        c[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_beta[None, :, None],  # y
        d[None, None, :],  # d
    ]
    exchange_bb[
        c[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        d[None, None, :],  # d
        all_beta[None, :, None],  # y
    ]
    # x == d
    coulomb_bb[
        d[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        c[None, None, :],  # c
        all_beta[None, :, None],  # y
    ]
    exchange_bb[
        d[None, None, :],  # x
        all_beta[None, :, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_beta[None, :, None],  # y
        c[None, None, :],  # c
    ]

    # y == a
    coulomb_bb[
        all_beta[None, :, None],  # x
        a[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        all_beta[None, :, None],  # x
        b[:, None, None],  # b
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_bb[
        all_beta[None, :, None],  # x
        a[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        all_beta[None, :, None],  # x
        b[:, None, None],  # b
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # y == b
    coulomb_bb[
        all_beta[None, :, None],  # x
        b[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        all_beta[None, :, None],  # x
        c[None, None, :],  # c
        d[None, None, :],  # d
    ]
    exchange_bb[
        all_beta[None, :, None],  # x
        b[:, None, None],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        all_beta[None, :, None],  # x
        d[None, None, :],  # d
        c[None, None, :],  # c
    ]
    # y == c
    coulomb_bb[
        all_beta[None, :, None],  # x
        c[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_beta[None, :, None],  # x
        d[None, None, :],  # d
    ]
    exchange_bb[
        all_beta[None, :, None],  # x
        c[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        d[None, None, :],  # d
        all_beta[None, :, None],  # x
    ]
    # y == d
    coulomb_bb[
        all_beta[None, :, None],  # x
        d[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] += self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        c[None, None, :],  # c
        all_beta[None, :, None],  # x
    ]
    exchange_bb[
        all_beta[None, :, None],  # x
        d[None, None, :],  # y
        occ_array_indices[:, None, None],  # a, b (occupied index)
        vir_array_indices[None, None, :],  # c, d (virtual index)
    ] -= self.two_int[
        a[:, None, None],  # a
        b[:, None, None],  # b
        all_beta[None, :, None],  # x
        c[None, None, :],  # c
    ]

    triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
    return sign_bb[None, None, :] * np.array(
        [
            coulomb_bb[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            exchange_bb[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
        ]
    )


RestrictedMolecularHamiltonian._integrate_sd_sds_one_alpha = _integrate_sd_sds_one_alpha
RestrictedMolecularHamiltonian._integrate_sd_sds_one_beta = _integrate_sd_sds_one_beta
RestrictedMolecularHamiltonian._integrate_sd_sds_two_aa = _integrate_sd_sds_two_aa
RestrictedMolecularHamiltonian._integrate_sd_sds_two_ab = _integrate_sd_sds_two_ab
RestrictedMolecularHamiltonian._integrate_sd_sds_two_bb = _integrate_sd_sds_two_bb
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_aa = _integrate_sd_sds_deriv_one_aa
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_bb = _integrate_sd_sds_deriv_one_bb
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_ba = _integrate_sd_sds_deriv_one_ba
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_ab = _integrate_sd_sds_deriv_one_ab
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_aaa = _integrate_sd_sds_deriv_two_aaa
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_aab = _integrate_sd_sds_deriv_two_aab
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_bab = _integrate_sd_sds_deriv_two_bab
RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_bbb = _integrate_sd_sds_deriv_two_bbb
