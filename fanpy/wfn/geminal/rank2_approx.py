"""Rank-2 approximation to geminal wavefunction."""

from fanpy.tools import math_tools, slater
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.geminal.base import BaseGeminal

import numpy as np


# pylint: disable=E1101
# FIXME: parent was removed to allow easier multiple inheritance. Maybe multiple inheritance should
#        be replaced with a wrapper instead? especially since ordering is necessary w/o splitting
#        the BaseGeminal in two.
class RankTwoApprox:
    r"""Rank-2 approximation to the geminal wavefunction.

    Geminal wavefunction where the geminal coefficient is parameterized as a rank-2 Cauchy matrix.

    .. math::

        C_{pi} = \frac{\zeta_i}{\epsilon_i - \lambda_p}

    Then, the evaluation of the permanent is equilalent to product of determinants.

    .. math::

        |C|^+ = \frac{|C \circ C|^-}{|C|^-}

    Properties
    ----------
    lambdas : np.ndarray(ngem)
        The :math:`lambda` part of the parameters.
    epsilons : np.ndarray(nspatial)
        The :math:`epsilon` part of the parameters.
    zetas : np.ndarray(nspatial)
        The :math:`zeta` part of the parameters.
    fullrank_params : np.ndarray(ngem, nspatial)
        Geminal coefficient matrix.

    Methods
    -------
    assign_params(self, params=None, add_noise=False)
        Assign the parameters of the geminal wavefunction.
    compute_permanent(self, col_inds, deriv=None) : float
        Compute the permanent of the matrix that corresponds to the given orbital pairs.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    """

    @property
    def lambdas(self):
        r"""Return the :math:`\lambda` part of the parameters.

        Returns
        -------
        lambdas : np.ndarray(ngem)
            The :math:`lambda` part of the parameters.

        """
        return self.params[: self.ngem]

    @property
    def epsilons(self):
        r"""Return the :math:`\epsilons` part of the parameters.

        Returns
        -------
        epsilons : np.ndarray(nspatial)
            The :math:`epsilon` part of the parameters.

        """
        return self.params[self.ngem : self.ngem + self.norbpair]

    @property
    def zetas(self):
        r"""Return the :math:`\zetas` part of the parameters.

        Returns
        -------
        zetas : np.ndarray(nspatial)
            The :math:`zeta` part of the parameters.

        """
        return self.params[self.ngem + self.norbpair :]

    @property
    def fullrank_params(self):
        """Return corresponding full rank parameters.

        Returns
        -------
        fullrank_params : np.ndarray(ngem, nspatial)
            Geminal coefficient matrix.

        """
        return self.zetas / (self.lambdas[:, np.newaxis] - self.epsilons)

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
            Default corresponds to the ground state HF wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        Raises
        ------
        NotImplementedError
            If BaseGeminal is given as a parameter.
        ValueError
            If there are any lambdas that are equal to epsilons (i.e. zero in denominator).

        """
        if params is None:
            # FIXME: add constraints to parameters
            #        zetas should be less than 1
            #        lambda - epsilon should be greater than 1
            #        lambda should be around 1 (epsilons should be less than 0)

            template = np.zeros((self.ngem, self.norbpair))
            for i in range(self.ngem):
                col_ind = self.get_col_ind((i, i + self.nspatial))
                template[i, col_ind] += 1
            template += 0.0001 * np.random.rand(*template.shape)  # pylint: disable=E1133
            # FIXME: fails a lot
            params = full_to_rank2(template, rmsd=0.01)

        if isinstance(params, BaseGeminal):
            raise NotImplementedError("Rank 2 Wavefunction cannot assign parameters using a BaseGeminal instance.")

        self.params = params
        if __debug__:
            # check for zeros in denominator
            if np.any(np.abs(self.lambdas[:, np.newaxis] - self.epsilons) < 1e-9):
                raise ValueError("Corresponding geminal coefficient matrix has a division by zero")
        BaseWavefunction.assign_params(self, params=params, add_noise=add_noise)
        self.clear_cache()

    # FIXME: too many branches
    def compute_permanent(self, col_inds, deriv=None):
        """Compute the permanent of the matrix that corresponds to the given orbital pairs.

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of geminal coefficient matrices that will be used.
        deriv : {int, None}
            Indices of the element (in parameters) with respect to which the permanent is
            derivatized.
            Default is no derivatization.

        Returns
        -------
        permanent : float
            Permanent of the matrix that corresponds to the given column indices.

        Raises
        ------
        TypeError
            If deriv is not an integer.
        ValueError
            If deriv is less than zero or greater than or equal to the number of parameters.

        """
        # pylint: disable=R0912
        row_inds = np.arange(self.ngem)
        col_inds = np.array(col_inds)

        if deriv is None:
            return math_tools.permanent_borchardt(self.lambdas[row_inds], self.epsilons[col_inds], self.zetas[col_inds])
        # if differentiating along row (lambda)
        if __debug__:
            if not (isinstance(deriv, int) or np.issubdtype(type(deriv), np.integer)):
                raise TypeError("deriv must be an integer.")
            if not 0 <= deriv < self.nparams:
                raise ValueError(
                    "deriv must be greater than or equal to zero and less than the number of " "parameters"
                )
        # FIXME: not the best way of evaluating
        if 0 <= deriv < self.npair:
            row_to_remove = deriv
            row_inds_trunc = row_inds[row_inds != row_to_remove]
            val = 0.0
            for col_to_remove in col_inds:
                col_inds_trunc = col_inds[col_inds != col_to_remove]
                # this will never happen (but just in case)
                if row_inds_trunc.size == row_inds.size or col_inds_trunc.size == col_inds.size:  # pragma: no cover
                    continue
                # derivative of matrix element c_ij wrt lambda_i
                der_cij_rowi = (
                    -self.zetas[col_to_remove] / (self.lambdas[row_to_remove] - self.epsilons[col_to_remove]) ** 2
                )
                if row_inds_trunc.size == col_inds_trunc.size == 0:
                    val += der_cij_rowi
                else:
                    val += der_cij_rowi * math_tools.permanent_borchardt(
                        self.lambdas[row_inds_trunc],
                        self.epsilons[col_inds_trunc],
                        self.zetas[col_inds_trunc],
                    )
            return val
        # if differentiating along column (epsilon or zeta)
        # else:  # if self.ngem <= deriv < self.ngem + 2 * self.norbpair:
        col_to_remove = (deriv - self.ngem) % self.norbpair
        col_inds_trunc = col_inds[col_inds != col_to_remove]
        if col_inds_trunc.size == col_inds.size:
            return 0.0

        val = 0.0
        for row_to_remove in row_inds:
            row_inds_trunc = row_inds[row_inds != row_to_remove]
            # differentiating wrt column
            if self.ngem <= deriv < self.ngem + self.norbpair:
                # derivative of matrix element c_ij wrt epsilon_j
                der_cij_colj = (
                    self.zetas[col_to_remove] / (self.lambdas[row_to_remove] - self.epsilons[col_to_remove]) ** 2
                )
            else:
                # derivative of matrix element c_ij wrt zeta_j
                der_cij_colj = 1.0 / (self.lambdas[row_to_remove] - self.epsilons[col_to_remove])

            if row_inds_trunc.size == col_inds_trunc.size == 0:
                val += der_cij_colj
            else:
                val += der_cij_colj * math_tools.permanent_borchardt(
                    self.lambdas[row_inds_trunc],
                    self.epsilons[col_inds_trunc],
                    self.zetas[col_inds_trunc],
                )
        return val

    def _olp_deriv(self, sd):  # pylint: disable=C0103
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.

        Returns
        -------
        olp_deriv : np.ndarray
            Derivatives of the overlap with respect to each parameter.

        """
        # NOTE: Need to recreate occ_indices, row_removed, col_removed
        occ_indices = slater.occ_indices(sd)

        orbpair_generator = self.generate_possible_orbpairs(occ_indices)

        output = np.zeros(self.nparams)
        for orbpairs, sign in orbpair_generator:
            # ASSUMES: permanent evaluation is much more expensive than the lookup
            if len(orbpairs) == 0:
                continue
            col_inds = np.array([self.get_col_ind(orbp) for orbp in orbpairs], dtype=int)
            # FIXME: converting all orbpairs is slow for some reason
            # col_inds = self.get_col_inds(np.array(orbpairs))
            for row_ind in range(self.ngem):
                output[row_ind] += sign * self.compute_permanent(col_inds, deriv=row_ind)
            for col_ind in col_inds:
                i = col_ind + self.ngem
                output[i] += sign * self.compute_permanent(col_inds, deriv=i)
                i = col_ind + self.ngem + self.norbpair
                output[i] += sign * self.compute_permanent(col_inds, deriv=i)
        return output


def full_to_rank2(params, rmsd=0.1, method="least squares"):
    r"""Convert full rank parameters to the rank-2 parameters.

    Using least squares, the full rank geminal coefficients are converted to the rank-2 variant,
    i.e. find the coefficients :math:`\{\lambda_j\}`, :math:`\{\epsilon_i\}`, and
    :math:`\{\zeta_i\}` such that following equation is best satisfied:

    .. math::
        C_{ij} &= \frac{\zeta_i}{\epsilon_i + \lambda_j}\\
        0 &= \zeta_i - C_{ij} \epsilon_i - C_{ij} \lambda_j\\

    The least square has the form of :math:`Ax=b`. Given that the :math:`b=0` and the unknowns
    are

    .. math::
        x = \begin{bmatrix}
        \lambda_1 \\ \vdots\\ \lambda_K\\
        \zeta_1 \\ \vdots\\ \zeta_P\\
        \epsilon_1 \\ \vdots\\ \epsilon_P\\
        \end{bmatrix},

    Then, A must be

    .. math::
        A =
        \left[
        \begin{array}{@{}*{12}{c}@{}}
            -C_{11} & 0 & \dots & 0 & -C_{11} & 0 & \dots & 0 & 1 & 0 & \dots & 0\\
            -C_{12} & 0 & \dots & 0 & 0 & -C_{12} & \dots & 0 & 0 & 1 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots & \vdots\\
            -C_{1K} & 0 & \dots & 0 & 0 & 0 & \dots & -C_{1K} & 0 & 0 & \dots & 1\\
            0 & -C_{21} & \dots & 0 & -C_{21} & 0 & \dots & 0 & 1 & 0 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots & \vdots\\
            0 & -C_{2K} & \dots & 0 & 0 & 0 & \dots & -C_{2K} & 0 & 0 & \dots & 1\\
            0 & 0 & \dots & -C_{PK} & -C_{P1} & 0 & \dots & 0 & 1 & 0 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots & \vdots\\
            0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & 1\\
        \end{array}
        \right]

    Parameters
    ----------
    params : np.ndarray(P, K)
        Geminal coefficient matrix.
        Number of rows is the number of geminals.
        Number of columns is the number of orbital pairs.
    rmsd : float
        Root mean square deviation allowed for the generated rank-2 coefficient matrix (compared
        to the full rank coefficient matrix).
        Default is `0.1`.
    method : {'least squares', 'svd'}
        Method by which the APr2G parameters are obtained.
        Default is 'least squares'.

    Returns
    -------
    rank2_params : np.ndarray
        Rank-2 parameters that "best" corresponds to the given full rank parameters.

    Raises
    ------
    ValueError
        If rank-2 coefficient matrix has a root mean square deviation with the full rank
        coefficient matrix that is greater than the threshold value.

    Notes
    -----
    This does not always work. You will likely need to tinker with some of the parameters inside
    this function.

    Examples
    --------
    Assuming we have a system with 2 electron pairs and 4 spatial orbitals, we have

    .. math::
        C = \begin{bmatrix}
        C_{11} & \dots & C_{1K}\\
        C_{21} & \dots & C_{2K}
        \end{bmatrix}

    .. math::
        A = \begin{bmatrix}
        C_{11} & 0 & -C_{11} & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        C_{12} & 0 & 0 & -C_{12} & 0 & 0 & 0 & 1 & 0 & 0\\
        C_{13} & 0 & 0 & 0 & -C_{13} & 0 & 0 & 0 & 1 & 0\\
        C_{14} & 0 & 0 & 0 & 0 & -C_{14} & 0 & 0 & 0 & 1\\
        0 & C_{21} & -C_{21} & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & C_{22} & 0 & -C_{22} & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & C_{23} & 0 & 0 & -C_{23} & 0 & 0 & 0 & 1 & 0\\
        0 & C_{24} & 0 & 0 & 0 & -C_{24} & 0 & 0 & 0 & 1\\
        \end{bmatrix}

    .. math::
        x = \begin{bmatrix}
        \lambda_ 1& \lambda_2
        \epsilon_1 \\ \epsilon_2\ \\ \epsilon_3 \\ \epsilon_4\\
        \zeta_1 \\ \zeta_2\\
        \end{bmatrix}

    """
    ngem, norbpair = params.shape
    # assign least squares matrix by reference
    matrix = np.zeros((params.size, ngem + 2 * norbpair), dtype=params.dtype)
    # set up submatrices that references a specific part of the matrix
    # NOTE: these values are broadcasted to `matrix`
    lambdas = matrix[:, :ngem]
    epsilons = matrix[:, ngem : ngem + norbpair]
    zetas = matrix[:, ngem + norbpair : ngem + 2 * norbpair]
    for i in range(ngem):
        lambdas[i * norbpair : (i + 1) * norbpair, i] = -params[i, :]
        epsilons[i * norbpair : (i + 1) * norbpair, :] = np.diag(params[i, :])
        zetas[i * norbpair : (i + 1) * norbpair, :] = np.identity(norbpair)

    # solve by Least Squares
    if method == "least squares":
        # Turn system of equations heterogeneous
        indices = np.zeros(ngem + 2 * norbpair, dtype=bool)
        vals = np.array([])

        # assign lambdas
        # indices[:ngem] = True
        # vals = np.hstack((vals, [1]*ngem))

        # assign epsilons
        # indices[ngem] = True
        # indices[ngem+norbpair-1] = True
        # vals = np.hstack((vals, [-10, -1]))

        # assign zetas
        indices[ngem + norbpair] = True
        vals = np.hstack((vals, [1]))
        # indices[ngem + norbpair:ngem + 2*norbpair] = True
        # vals = np.hstack((vals, np.ones(norbpair)))

        ordinate = -matrix[:, indices].dot(vals)

        # Solve the least-squares system
        rank2_params = np.zeros(indices.size)
        rank2_params[indices] = vals
        not_indices = np.logical_not(indices)
        rank2_params[not_indices] = np.linalg.lstsq(matrix[:, not_indices], ordinate, rcond=None)[0]
    # solve by SVD
    else:  # if method == "svd":
        _, s, vh = np.linalg.svd(matrix, full_matrices=False)  # pylint: disable=C0103
        # find null vectors
        indices = np.abs(s) < 1
        # guess solution
        b = np.vstack(  # pylint: disable=C0103
            [
                np.random.rand(ngem, 1),
                np.sort(np.random.rand(norbpair, 1)) - 1,
                np.ones((norbpair, 1)),
            ]
        )
        # linearly combine right null vectors
        lin_comb = np.linalg.lstsq(vh[indices].T, b, rcond=None)[0]
        rank2_params = vh[indices].T.dot(lin_comb).flatten()

    # Check
    lambdas = rank2_params[:ngem, np.newaxis]
    epsilons = rank2_params[ngem : ngem + norbpair]
    zetas = rank2_params[ngem + norbpair :]
    rank2_coeffs = zetas / (lambdas - epsilons)
    deviation = (np.sum((params - rank2_coeffs) ** 2) / params.size) ** (0.5)
    if np.isnan(deviation) or deviation > rmsd:
        raise ValueError(
            "Rank-2 coefficient matrix has RMSD of {0} with the full-rank coefficient " "matrix".format(deviation)
        )

    return rank2_params
