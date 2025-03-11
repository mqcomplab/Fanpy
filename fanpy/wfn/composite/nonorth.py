"""Wavefunction with nonorthonormal orbitals."""

import itertools as it

from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.base_one import BaseCompositeOneWavefunction

import numpy as np


# FIXME: needs refactoring
class NonorthWavefunction(BaseCompositeOneWavefunction):
    r"""Wavefunction with nonorthonormal orbitals expressed with respect to orthonormal orbitals.

    A parameterized multideterminantal wavefunction can be written as

    .. math::
        \left| \Psi \right> = \sum_{\mathbf{m}} f(\mathbf{m}) \left| \mathbf{m} \right>

    where :math:`\left| \mathbf{m} \right>` is a Slater determinant. If the Slater determinants are
    constructed from nonorthonormal orbitals, then each Slater determinant can be expressed as a
    linear combination of Slater determinants constructed from orthonormal orbitals.

    .. math::
        \left| \Psi \right>
        &= \sum_{\mathbf{n}} f(\mathbf{n}) \left| \mathbf{n} \right>\\
        &= \sum_{\mathbf{n}} f(\mathbf{n}) \sum_{\mathbf{m}}
        |C(\mathbf{n}, \mathbf{m})|^- \left| \mathbf{m} \right>\\
        &= \sum_{\mathbf{n}} \sum_{\mathbf{m}}
        f(\mathbf{n}) |C(\mathbf{n}, \mathbf{m})|^- \left| \mathbf{m} \right>

    where :math:`\left| \mathbf{m} \right>` and :math:`\left| \mathbf{n} \right>` are Slater
    determinants constructed from orthonormal and nonorthonormal orbitals. The nonorthonormal
    orbitals are constructed by linearly transforming the orbitals of
    :math:`\left| \mathbf{m} \right>` with :math:`C`. The :math:`C(\mathbf{n}, \mathbf{m})` is a
    submatrix of :math:`C` where rows are selected according to :math:`\left| \mathbf{n} \right>`
    and columns to :math:`\left| \mathbf{m} \right>`.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    memory : float
        Memory available for the wavefunction.
    params : tuple of np.ndarray
        Orbital transformation matrices.
        If one transformation matrix is given, then the transformation coresponds to those of
        restricted orbitals, where the spatial orbitals are transformed or to those of generalized
        orbitals, where spin orbitals are transformed.
        If two transformation matrices are given, then the transformation corresponds to those of
        unrestricted orbitals, where the spatial orbitals are transformed.
    wfn : BaseWavefunction
        Wavefunction whose orbitals are rotated.
    jacobi_indices : 2-tuple of ints
        Orbitals that are rotated.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    dtype
        Data type of the wavefunction.
    params_shape : tuple of int
        Shape of the parameters.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Orbital type.

    Methods
    -------
    __init__(self, nelec, nspin, wfn, memory=None, params=None, orbtype=None, jacobi_indices=None):
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign the orbital transformation matrix.
    enable_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    """

    @property
    def spin(self):
        """Return the spin of the wavefunction.

        If the orbitals are restricted or unrestricted, the spin should be same as the original.
        Otherwise, the orbitals may mix regardless of the spin, the spin of the wavefunction is hard
        to determine.

        Returns
        -------
        spin : float
            Spin of the (composite) wavefunction if the orbitals are restricted or unrestricted.
            None if all spins are allowed.

        """
        if self.orbtype in ["restricted", "unrestricted"]:
            return self.wfn.spin
        return None

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        If the orbitals are restricted or unrestricted, the seniority should be same as the
        original. Otherwise, the orbitals may mix regardless of the seniority, the seniority of the
        wavefunction is hard to determine.

        Returns
        -------
        seniority : int
            Seniority of the (composite) wavefunction if the orbitals are restricted or
            unrestricted.
            None if all seniority are allowed.

        """
        if self.orbtype in ["restricted", "unrestricted"]:
            return self.wfn.seniority
        return None

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : tuple of int
            Number of elements in each transformation matrix.

        """
        return sum(i.size for i in self.params)

    # FIXME: bad name
    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of 2-tuple of ints
            Shape of each transformation matrix.

        """
        return tuple(i.shape for i in self.params)

    @property
    def orbtype(self):  # pylint: disable=R1710
        """Return the orbital type.

        Returns
        -------
        orbtype : str
            'restricted' if only one transformation matrix is given and the number of rows
            corresponds to the number of spatial orbitals.
            'unrestricted' if two transformation matrices are given and the number of rows
            corresponds to the number of spatial orbitals.
            'restricted' if only one transformation matrix is given and the number of rows
            corresponds to the number of spin orbitals.

        """
        # pylint: disable=R1705
        if len(self.params) == 1 and self.params[0].shape[0] == self.nspatial:
            return "restricted"
        elif (
            len(self.params) == 2
            and self.params[0].shape[0] == self.nspatial
            and self.params[1].shape[0] == self.nspatial
        ):
            return "unrestricted"
        elif len(self.params) == 1 and self.params[0].shape[0] == self.nspin:  # pragma: no branch
            return "generalized"

    def assign_params(self, params=None, add_noise=False):
        """Assign the orbital transformation matrix.

        Parameters
        ----------
        params : {np.ndarray, tuple of np.ndarray}
            Transformation matrices.
            If one transformation matrix is given, then the transformation coresponds to those of
            restricted orbitals, where the spatial orbitals are transformed or to those of
            generalized orbitals, where spin orbitals are transformed.
            If two transformation matrices are given, then the transformation corresponds to those
            of unrestricted orbitals, where the spatial orbitals are transformed.
            Default is the no orbital transformation (unitary matrix).
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        Raises
        ------
        TypeError
            If transformation matrix is not a numpy array or 1- or 2-tuple/list of numpy arrays.
            If transformation matrix is not a two dimension numpy array.
            If transformation matrix is not a two dimension numpy array.
        ValueError
            If transformation matrix does not have the right shape.

        """
        if params is None:
            params = (np.eye(self.nspatial, self.wfn.nspatial),)

        if isinstance(params, np.ndarray):
            params = (params,)

        if __debug__:
            if not (isinstance(params, (tuple, list)) and len(params) in [1, 2]):
                raise TypeError(
                    "Transformation matrix must be a two dimensional numpy array or a "
                    "1- or 2-tuple/list of two dimensional numpy arrays. Only one numpy "
                    "arrays indicate that the orbitals are restricted or generalized and "
                    "two numpy arrays indicate that the orbitals are unrestricted."
                )
            for i in params:
                if not (isinstance(i, np.ndarray) and len(i.shape) == 2):
                    raise TypeError("Transformation matrix must be a two-dimensional numpy array.")

                if len(params) == 1 and not (
                    (i.shape[0] == self.nspatial and i.shape[1] == self.wfn.nspatial)
                    or (i.shape[0] == self.nspin and i.shape[1] == self.wfn.nspin)
                ):
                    raise ValueError(
                        "Given the type of transformation, the numpy matrix has the "
                        "wrong shape. If only one numpy array is given, the "
                        "orbitals are transformed either from orthonormal spatial "
                        "orbitals to nonorthonormal spatial orbitals or from "
                        "orthonormal spin orbitals to nonorthonormal spin orbitals."
                    )

                if len(params) == 2 and not (i.shape[0] == self.nspatial and i.shape[1] == self.wfn.nspatial):
                    raise ValueError(
                        "Given the type of transformation, the numpy matrix has the "
                        "wrong shape. If two numpy arrays are given, the orbitals are "
                        "transformed from orthonormal spatial orbitals to nonorthonormal "
                        "spatial orbitals."
                    )

        # add random noise
        if add_noise:
            # set scale
            scale = 0.2 / self.nparams
            new_params = []
            for old_params in params:
                old_params += scale * (np.random.rand(*old_params.shape) - 0.5)
                if old_params.dtype in [complex, np.complex128]:
                    old_params += 0.01j * scale * (np.random.rand(*old_params.shape).astype(complex) - 0.5)
                new_params.append(old_params)
            params = new_params

        self.params = tuple(params)
        self.clear_cache()

    def _olp(self, sd):  # pylint: disable=E0202
        """Calculate the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.

        Returns
        -------
        olp : float
            Overlap of the wavefunction with the Slater determinant.

        """
        output = 0.0
        if self.orbtype == "generalized":
            row_inds = slater.occ_indices(sd)
            all_col_inds = range(self.wfn.nspin)
            for col_inds in it.combinations(all_col_inds, self.nelec):
                nonorth_sd = slater.create(0, *col_inds)
                wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                # FIXME: use broadcasting
                nonorth_coeff = np.linalg.det(self.params[0][row_inds, :][:, col_inds])
                output += wfn_coeff * nonorth_coeff
        else:
            alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
            alpha_row_inds = slater.occ_indices(alpha_sd)
            beta_row_inds = slater.occ_indices(beta_sd)
            all_col_inds = range(self.wfn.nspatial)
            for alpha_col_inds in it.combinations(all_col_inds, len(alpha_row_inds)):
                if len(alpha_col_inds) == 0:
                    alpha_coeff = 1.0
                else:
                    # FIXME: use broadcasting
                    alpha_coeff = np.linalg.det(self.params[0][alpha_row_inds, :][:, alpha_col_inds])
                for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds)):
                    # FIXME: change i+nspatial to slater.to_beta
                    nonorth_sd = slater.create(0, *alpha_col_inds, *[i + self.nspatial for i in beta_col_inds])
                    wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                    if self.orbtype == "restricted":
                        i = 0
                    else:
                        i = 1

                    if len(beta_col_inds) == 0:
                        beta_coeff = 1.0
                    else:
                        # FIXME: use broadcasting
                        beta_coeff = np.linalg.det(self.params[i][beta_row_inds, :][:, beta_col_inds])
                    output += wfn_coeff * alpha_coeff * beta_coeff
        return output

    # FIXME: too many branches, too many statements
    def _olp_deriv(self, sd, deriv):  # pylint: disable=E0202
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.

        Returns
        -------
        olp_deriv : float
            Derivative of the overlap of the wavefunction with the Slater determinant.

        """
        # pylint: disable=R0912,R0915
        # number of parameters for alpha orbitals (this variable will have no effect for restricted
        # and generalized orbital types)
        nparams_alpha = self.params[0].size
        # lots of repetition b/c slight variations with different orbital types
        transform_ind = deriv // nparams_alpha
        row_removed = (deriv % nparams_alpha) // self.params_shape[transform_ind][1]
        col_removed = (deriv % nparams_alpha) % self.params_shape[transform_ind][1]

        output = 0.0
        if self.orbtype == "generalized":
            row_inds = slater.occ_indices(slater.annihilate(sd, row_removed))
            row_sign = (-1) ** np.sum(np.array(row_inds) < row_removed)
            all_col_inds = (i for i in range(self.wfn.nspin) if i != col_removed)
            for col_inds in it.combinations(all_col_inds, len(row_inds)):
                nonorth_sd = slater.create(0, col_removed, *col_inds)
                wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                col_sign = (-1) ** np.sum(np.array(col_inds) < col_removed)
                # FIXME: use broadcasting
                nonorth_coeff = np.linalg.det(self.params[0][row_inds, :][:, col_inds])
                output += wfn_coeff * row_sign * col_sign * nonorth_coeff
            return output

        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        if not (slater.occ(alpha_sd, row_removed) or slater.occ(beta_sd, row_removed)):
            return 0.0
        # FIXME/TODO: need to add signature for derivative
        if self.orbtype == "restricted" and slater.occ(alpha_sd, row_removed) and slater.occ(beta_sd, row_removed):
            # if both alpha and beta Slater determinants contain the orbital
            alpha_row_inds = slater.occ_indices(slater.annihilate(alpha_sd, row_removed))
            beta_row_inds = slater.occ_indices(slater.annihilate(beta_sd, row_removed))

            all_col_inds = [i for i in range(self.wfn.nspatial) if i != col_removed]

            # alpha block includes derivatized column
            for alpha_col_inds in it.combinations(all_col_inds, len(alpha_row_inds)):
                alpha_col_inds = np.array(alpha_col_inds, dtype=int)
                if len(alpha_col_inds) == 0:
                    der_alpha_coeff = 1.0
                else:
                    # FIXME: use broadcasting
                    der_alpha_coeff = np.linalg.det(self.params[0][alpha_row_inds, :][:, alpha_col_inds])
                # FIXME: use broadcasting
                alpha_coeff = np.linalg.det(
                    self.params[0][np.hstack([alpha_row_inds, row_removed]), :][
                        :, np.hstack([alpha_col_inds, col_removed])
                    ]
                )
                # beta block includes derivatized column
                # FIXME: change i+nspatial to slater.to_beta
                for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds)):
                    beta_col_inds = np.array(beta_col_inds, dtype=int)
                    nonorth_sd = slater.create(
                        0,
                        col_removed,
                        col_removed + self.nspatial,
                        *alpha_col_inds,
                        *(i + self.nspatial for i in beta_col_inds)
                    )
                    wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                    if len(beta_col_inds) == 0:
                        der_beta_coeff = 1.0
                    else:
                        # FIXME: use broadcasting
                        der_beta_coeff = np.linalg.det(self.params[0][beta_row_inds, :][:, beta_col_inds])
                    # FIXME: use broadcasting
                    beta_coeff = np.linalg.det(
                        self.params[0][np.hstack([beta_row_inds, row_removed]), :][
                            :, np.hstack([beta_col_inds, col_removed])
                        ]
                    )
                    output += wfn_coeff * (der_alpha_coeff * beta_coeff + alpha_coeff * der_beta_coeff)

                # beta block does not include derivatized column
                for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds) + 1):
                    nonorth_sd = slater.create(
                        0, col_removed, *alpha_col_inds, *(i + self.nspatial for i in beta_col_inds)
                    )
                    wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                    # FIXME: use broadcasting
                    beta_coeff = np.linalg.det(
                        self.params[0][np.hstack([beta_row_inds, row_removed]), :][:, beta_col_inds]
                    )
                    output += wfn_coeff * der_alpha_coeff * beta_coeff
            # alpha block does not include the derivatized column
            for alpha_col_inds in it.combinations(all_col_inds, len(alpha_row_inds) + 1):
                # FIXME: use broadcasting
                alpha_coeff = np.linalg.det(
                    self.params[0][np.hstack([alpha_row_inds, row_removed]), :][:, alpha_col_inds]
                )
                # beta block includes derivatized column
                for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds)):
                    nonorth_sd = slater.create(
                        0, col_removed + self.nspatial, *alpha_col_inds, *(i + self.nspatial for i in beta_col_inds)
                    )
                    wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                    if len(beta_col_inds) == 0:
                        der_beta_coeff = 1.0
                    else:
                        # FIXME: use broadcasting
                        der_beta_coeff = np.linalg.det(self.params[0][beta_row_inds, :][:, beta_col_inds])
                    output += wfn_coeff * alpha_coeff * der_beta_coeff

            return output

        # if only one of alpha and beta Slater determinants contains the orbital
        # elif slater.occ(alpha_sd, row_removed) != slater.occ(beta_sd, row_removed):
        if (self.orbtype == "restricted" and slater.occ(alpha_sd, row_removed)) or (
            self.orbtype == "unrestricted" and transform_ind == 0
        ):
            alpha_row_inds = slater.occ_indices(slater.annihilate(alpha_sd, row_removed))
            beta_row_inds = slater.occ_indices(beta_sd)
            row_sign = (-1) ** np.sum(np.array(alpha_row_inds) < row_removed)
            all_alpha_col_inds = (i for i in range(self.wfn.nspatial) if i != col_removed)
            all_beta_col_inds = range(self.wfn.nspatial)
        else:
            alpha_row_inds = slater.occ_indices(alpha_sd)
            beta_row_inds = slater.occ_indices(slater.annihilate(beta_sd, row_removed))
            row_sign = (-1) ** np.sum(np.array(beta_row_inds) < row_removed)
            all_alpha_col_inds = range(self.wfn.nspatial)
            all_beta_col_inds = (i for i in range(self.wfn.nspatial) if i != col_removed)

        for alpha_col_inds in it.combinations(all_alpha_col_inds, len(alpha_row_inds)):
            col_sign = 1
            if transform_ind == 0:
                col_sign *= (-1) ** np.sum(np.array(alpha_col_inds) < col_removed)
            if len(alpha_col_inds) == 0:
                alpha_coeff = 1.0
            else:
                alpha_coeff = np.linalg.det(self.params[0][alpha_row_inds, :][:, alpha_col_inds])
            for beta_col_inds in it.combinations(all_beta_col_inds, len(beta_row_inds)):
                # FIXME: change i+nspatial to slater.to_beta
                nonorth_sd = slater.create(0, *alpha_col_inds, *[i + self.nspatial for i in beta_col_inds])
                if (self.orbtype == "restricted" and slater.occ(alpha_sd, row_removed)) or (
                    self.orbtype == "unrestricted" and transform_ind == 0
                ):
                    nonorth_sd = slater.create(nonorth_sd, col_removed)
                else:
                    nonorth_sd = slater.create(nonorth_sd, col_removed + self.nspatial)

                wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                if transform_ind == 1:
                    col_sign *= (-1) ** np.sum(np.array(beta_col_inds) < col_removed)
                if len(beta_col_inds) == 0:
                    beta_coeff = 1.0
                else:
                    orb_ind = 0 if self.orbtype == "restricted" else 1
                    beta_coeff = np.linalg.det(self.params[orb_ind][beta_row_inds, :][:, beta_col_inds])
                output += wfn_coeff * row_sign * col_sign * alpha_coeff * beta_coeff
        return output

    # FIXME: incredibly slow/bad approach
    # TODO: instead of all possible combinations (itertools), have something that selects a smaller
    #       subset
    # FIXME: derivative wrt wavefunction parameters (not jacobi rotation)?
    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with an orthonormal Slater determinant.

        A wavefunction built using nonorthonormal Slater determinants, :math:`\mathbf{n}`, can be
        expressed with respect to orthonormal Slater determinants, :math:`\mathbf{m}`:

        .. math::
            \left| \Psi \right>
            &= \sum_{\mathbf{n}} f(\mathbf{n}) \sum_{\mathbf{m}} |U(\mathbf{m}, \mathbf{n})|^+
            \left| \mathbf{m} \right>\\
            &= \sum_{\mathbf{m}} \sum_{\mathbf{n}} f(\mathbf{n}) |U(\mathbf{m}, \mathbf{n})|^+
            \left| \mathbf{m} \right>

        Then, the overlap with an orthonormal Slater determinant is

        .. math::
            \left< \Phi_i \middle| \Psi \right>
            = \sum_{\mathbf{n}} f(\mathbf{n}) |U(\Phi_i, \mathbf{n})|^+

        where :math:`U(\Phi_i, \mathbf{n})` is the transformation matrix with rows and columns that
        correspond to the Slater determinants :math:`\Phi_i` and :math:`\mathbf{n}`, respectively.

        Parameters
        ----------
        sd : int
            Slater Determinant against which the overlap is taken.
        deriv : {2-tuple, None}
            Wavefunction and the indices of the parameters with respect to which the overlap is
            derivatized.
            First element of the tuple is the wavefunction. Second element of the tuple is the
            indices of the parameters of the corresponding wavefunction. The overlap will be
            derivatized with respect to the selected parameters of this wavefunction.
            Default returns the overlap without derivatization.

        Returns
        -------
        overlap : {float, np.ndarray}
            Overlap (or derivative of the overlap) of the wavefunction with the given Slater
            determinant.

        Raises
        ------
        TypeError
            If given Slater determinant is not an integer.
            If `deriv` is not a 2-tuple where the first element is a BaseWavefunction instance and
            the second element is a one-dimensional numpy array of integers.
        ValueError
            If first element of `deriv` is not the composite wavefunction or the underlying
            waefunction.
            If the provided indices is less than zero or greater than or equal to the number of the
            corresponding parameters.
        NotImplementedError
            If the overlap is derivatized with respect to the parameters of the underlying
            wavefunction.

        """
        if deriv is None:
            return self._olp(sd)

        # if derivatization
        if __debug__:
            if not (
                isinstance(deriv, tuple)
                and len(deriv) == 2
                and isinstance(deriv[0], BaseWavefunction)
                and isinstance(deriv[1], np.ndarray)
                and deriv[1].ndim == 1
                and np.issubdtype(deriv[1].dtype, np.integer)
            ):
                raise TypeError(
                    "Derivative indices must be given as a 2-tuple whose first element is the "
                    "wavefunction and the second elment is the one-dimensional numpy array of "
                    "integer indices."
                )
            if deriv[0] not in (self, self.wfn):
                raise ValueError(
                    "Selected wavefunction (for derivatization) is not one of the composite "
                    "wavefunction or its underlying wavefunction."
                )
            if deriv[0] == self and (np.any(deriv[1] < 0) or np.any(deriv[1] >= self.nparams)):
                raise ValueError(
                    "Provided indices must be greater than or equal to zero and less than the " "number of parameters."
                )

        wfn, indices = deriv
        if wfn == self:
            output = np.zeros(len(indices))
            for i in indices:
                # number of parameters for alpha orbitals (this variable will have no effect for
                # restricted and generalized orbital types)
                nparams_alpha = self.params[0].size
                # get index of the transformation (if unrestricted)
                transform_ind = i // nparams_alpha
                # convert parameter index to row and col index
                row_removed = (i % nparams_alpha) // self.params_shape[transform_ind][1]

                # if either of these orbitals are not present in the Slater determinant, skip
                # FIXME: change i+nspatial to slater.to_beta
                if self.orbtype == "restricted" and not (
                    slater.occ(sd, row_removed) or slater.occ(sd, row_removed + self.nspatial)
                ):
                    continue
                if self.orbtype == "unrestricted" and not slater.occ(sd, row_removed + transform_ind * self.nspatial):
                    continue
                if self.orbtype == "generalized" and not slater.occ(sd, row_removed):
                    continue
                output[i] = self._olp_deriv(sd, i)
            return output
        raise NotImplementedError(
            "To implement this, the derivative indices must be passed to the "
            "`wfn.get_overlap` in `_olp`. But that interferes with the caching system."
        )
