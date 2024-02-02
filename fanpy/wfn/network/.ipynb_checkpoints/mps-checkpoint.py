"""Hard-coded Matrix Product State wavefunction."""
import functools

from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction

import numpy as np


# TODO: make subclass of TensorNetworkState
# TODO: use TensorFlow or some other tensor library
class MatrixProductState(BaseWavefunction):
    r"""Matrix Product State wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.


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

    Methods
    -------
    __init__(self, nelec, nspin, memory=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign parameters of the wavefunction.
    enable_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.
    assign_dimension(self, dimension=None)
        Assign the dimension of the matrices.
    get_occupation_indices(self, sd) : np.ndarray
        Return the occupation vector of the Slater determinant in the format used in MPS.
    get_matrix_shape(self, index) : 3-tuple of int
        Get the shape of a matrix.
    get_matrix_indices(self, index) : 2-tuple of int
        Get the start and end indices of a matrix.
    get_matrix(self, index) : np.ndarray
        Get the matrix that correspond to the spatial orbital of the given index.
    decompose_index(self, param_index) : 4-tuple of int
        Return the indices of the spatial orbital, occupation, row and column indices.

    """

    def __init__(self, nelec, nspin, memory=None, params=None, dimension=None, enable_cache=True):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            If number is provided, it is the number of bytes.
            If string is provided, it should end iwth either "mb" or "gb" to specify the units.
            Default does not limit memory usage (i.e. infinite).
        dimension : int
            Number of rows and columns in the matrices that correspond to the non terminal orbitals.
        params : np.ndarray
            Parameters of the wavefunction.
        enable_cache : bool
            Option to cache the results of `_olp` and `_olp_deriv`.
            By default, `_olp` and `_olp_deriv` are cached.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_dimension(dimension)
        self.assign_params(params)
        if enable_cache:
            self.enable_cache()

    # TODO: all of the auxiliary indices are fixed to be equal. This may need to be flexible
    def assign_dimension(self, dimension=None):
        """Assign the dimension of the parameters of the wavefunction.

        Parameters
        ----------
        dimension : {int, None}
            Dimension of the auxiliary index.
            All of the auxiliary indices will have the same dimension.

        Raises
        ------
        TypeError
            If dimension is not an integer or None.
        ValueError
            If dimension is less than or equal to zero.

        Note
        ----
        Needs `nspin` attribute for the default behaviour.

        """
        # FIXME: need a better default for dimension
        if dimension is None:
            dimension = self.nspin

        if __debug__:
            if not isinstance(dimension, int):
                raise TypeError("Provided dimension must be an integer or None.")
            if dimension <= 0:
                raise ValueError("Dimension of the matrices must be a nonnegative integer.")

        self.dimension = dimension

    def get_occupation_indices(self, sd):
        """Return the occupation vector of the Slater determinant in the format used in MPS.

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.

        Returns
        -------
        indices : np.ndarray
            Indices that correspond to the occupation of the Slater determinant.
            Entry of 0 means that the spatial orbital is not occupied.
            Entry of 1 means that the spatial orbital is singly occupied (alpha).
            Entry of 2 means that the spatial orbital is singly occupied (beta).
            Entry of 3 means that the spatial orbital is doubly occupied.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.

        Note
        ----
        Requires `nspin` attribute.

        """
        # pylint: disable=C0103
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
        indices = np.zeros(self.nspatial, dtype=int)
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        alpha_occ = list(slater.occ_indices(alpha_sd))
        beta_occ = list(slater.occ_indices(beta_sd))
        indices[alpha_occ] += 1
        indices[beta_occ] += 2
        return indices

    def get_matrix_shape(self, index):
        """Get the shape of a matrix.

        Parameters
        ----------
        index : int
            Index of the spatial orbital that corresponds to the desired matrix.

        Returns
        -------
        shape : tuple of int
            Shape of the selected matrix.

        Raises
        ------
        TypeError
            If index is not an integer
        ValueError
            If index is not greater than or equal to 0 and less than number of spatial orbitals.

        Note
        ----
        Requires attributes `nspin` and `dimension`.

        """
        if __debug__:
            if not isinstance(index, (int, np.int64)):
                raise TypeError("Given index must be an integer.")
            if not 0 <= index < self.nspatial:
                raise ValueError(
                    "Given index must be greater than or equal to zero or less than the number of "
                    "spatial orbitals."
                )

        if index == 0:  # pylint: disable=R1705
            return (4, 1, self.dimension)
        elif index < self.nspatial - 1:
            return (4, self.dimension, self.dimension)
        else:
            return (4, self.dimension, 1)

    def get_matrix_indices(self, index):
        """Get the start and end indices of a matrix.

        Parameters
        ----------
        index : int
            Index of the spatial orbital that corresponds to the desired matrix.

        Returns
        -------
        start_index : int
            Index of the parameter that indicates the start (inclusive) of the matrix.
        end_index : int
            Index of the parameter that indicates the end (exclusive) of the matrix.

        """
        size_matrix = np.prod(self.get_matrix_shape(index))
        if index == 0:
            return (0, size_matrix)
        # NOTE: assumes that matrix of index 1 has the same shape as the subsequent ones (except
        #       last one)
        start = np.prod(self.get_matrix_shape(0)) + np.prod(self.get_matrix_shape(1)) * (index - 1)
        return (start, start + np.prod(self.get_matrix_shape(index)))

    def get_matrix(self, index):
        """Get the matrix that correspond to the spatial orbital of the given index.

        Parameters
        ----------
        index : int
            Index of the spatial orbital.

        Returns
        -------
        matrix : np.ndarray
            Matrix that corresponds to the given index.

        """
        start_ind, end_ind = self.get_matrix_indices(index)
        matrix_shape = self.get_matrix_shape(index)
        return self.params[start_ind:end_ind].reshape(matrix_shape)

    def decompose_index(self, param_index):
        """Return the indices of the spatial orbital, occupation, row and column indices.

        Parameters
        ----------
        param_index : int
            Index of the parameter.

        Returns
        -------
        ind_spatial : int
            Index of the spatial orbital that corresponds to the parameter.
        ind_occ : int
            Index of the occupation of the spatial orbital that corresponds to the parameter.
        ind_row : int
            Row index of the matrix that correspond to the given spatial orbital and occupation.
        ind_col : int
            Column index of the matrix that correspond to the given spatial orbital and occupation.

        """
        start_index, end_index = self.get_matrix_indices(0)
        if start_index <= param_index < end_index:
            ind_spatial = 0
        else:
            # NOTE: assumes that matrix of index 1 has the same shape as the subsequent ones (except
            #       last one)
            ind_spatial = (param_index - end_index) // np.prod(self.get_matrix_shape(1))
            ind_spatial += 1
            param_index -= self.get_matrix_indices(ind_spatial)[0]

        matrix_shape = self.get_matrix_shape(ind_spatial)
        ind_occ, ind_row, ind_col = np.unravel_index(  # pylint: disable=W0632
            [param_index], matrix_shape
        )
        return ind_spatial, ind_occ.item(), ind_row.item(), ind_col.item()

    # TODO: the parameters can probably be changed to something more elegant (like a tensor object
    #       or something)
    # TODO: the MPS structure may need to become compatible with the TNS in the future (and vice
    #       versa)
    # FIXME: move to the parent class
    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the MPS wavefunction.

        Parameters
        ----------
        params : {np.ndarray, MatrixProductState, None}
            Parameters of the MPS wavefunction.
            If MatrixProductState instance is given, then the parameters of this instance are used.
            Default corresponds to the ground state HF wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        """
        if params is None:
            # NOTE: assumes that matrix of index 1 has the same shape as the subsequent ones (except
            #       last one)
            #       assumes that there are K matrices
            matrices = [np.zeros(self.get_matrix_shape(0))]
            matrices += [np.zeros(self.get_matrix_shape(1)) for i in range(self.nspatial - 2)]
            matrices += [np.zeros(self.get_matrix_shape(self.nspatial - 1))]

            ground_sd = slater.ground(self.nelec, self.nspin)
            occ_indices = self.get_occupation_indices(ground_sd)

            # adjust scale so that the overlap is 1
            scale = self.dimension ** (-1 / occ_indices.size)
            # set the first matrix
            matrices[0][occ_indices[0], 0, :] = scale
            # set the middle matrices
            for i, ind in enumerate(occ_indices[1:-1]):
                # select the diagonal
                diag_indices = np.arange(self.dimension)
                matrices[i + 1][ind, diag_indices, diag_indices] = scale
            # set the last matrix
            matrices[-1][occ_indices[-1], :, 0] = scale
            # flatten and join together
            params = np.hstack([matrix.flatten() for matrix in matrices])

        if isinstance(params, MatrixProductState):
            params = params.params
        super().assign_params(params=params, add_noise=add_noise)
        self.clear_cache()

    def _olp(self, sd):  # pylint: disable=E0202
        """Calculate the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.

        Returns
        -------
        olp : {float, complex}
            Overlap of the current instance with the given Slater determinant.

        """
        occ_indices = self.get_occupation_indices(sd)

        temp_matrix = self.get_matrix(0)[occ_indices[0], :, :]
        for i in range(1, occ_indices.size):
            temp_matrix = temp_matrix.dot(self.get_matrix(i)[occ_indices[i], :, :])
        return temp_matrix.item()

    def _olp_deriv(self, sd, deriv):  # pylint: disable=E0202
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.
            Assumed to correspond to the matrix elements that correspond to the given Slater
            determinant.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        deriv_matrix, _, deriv_row, deriv_col = self.decompose_index(deriv)

        occ_indices = self.get_occupation_indices(sd)

        left_temp, right_temp = 1.0, 1.0

        # if selected matrix is not the right most matrix
        if deriv_matrix < occ_indices.size - 1:
            # index of the matrix to the right of the given matrix
            index_right = deriv_matrix + 1

            # multiply matrix towards the right (from the derivatized matrix)
            right_temp = self.get_matrix(index_right)[occ_indices[index_right], deriv_col, :]
            for i in range(index_right + 1, occ_indices.size):
                right_temp = right_temp.dot(self.get_matrix(i)[occ_indices[i], :, :])

        # if selected matrix is not the left most matrix
        if deriv_matrix > 0:
            # index of the matrix to the left of the given matrix
            index_left = deriv_matrix - 1

            # multiply matrix towards the left (from the derivatized matrix)
            left_temp = self.get_matrix(index_left)[occ_indices[index_left], :, deriv_row]
            for i in reversed(range(index_left)):
                left_temp = self.get_matrix(i)[occ_indices[i], :, :].dot(left_temp)

        return (left_temp * right_temp).item()

    def _olp_deriv_block(self, sd, ind_spatial):  # pylint: disable=C0103,E0202
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.
            Assumed to correspond to the matrix elements that correspond to the given Slater
            determinant.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        occ_indices = self.get_occupation_indices(sd)

        left_temp, right_temp = 1.0, 1.0

        # if selected matrix is not the right most matrix
        if ind_spatial < occ_indices.size - 1:
            # index of the matrix to the right of the given matrix
            index_right = ind_spatial + 1

            # multiply matrix towards the right (from the derivatized matrix)
            right_temp = self.get_matrix(index_right)[occ_indices[index_right], :, :]
            for i in range(index_right + 1, occ_indices.size):
                right_temp = right_temp.dot(self.get_matrix(i)[occ_indices[i], :, :])
            # right_temp is a column vector now, where each entry corresponds to the column index of
            # the chosen matrix with which the overlap is derivatized

        # if selected matrix is not the left most matrix
        if ind_spatial > 0:
            # index of the matrix to the left of the given matrix
            index_left = ind_spatial - 1

            # multiply matrix towards the left (from the derivatized matrix)
            left_temp = self.get_matrix(index_left)[occ_indices[index_left], :, :]
            for i in reversed(range(index_left)):
                left_temp = self.get_matrix(i)[occ_indices[i], :, :].dot(left_temp)
            # left_temp is a row vector now, where each entry corresponds to the row index of
            # the chosen matrix with which the overlap is derivatized

        return (left_temp * right_temp).T

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
            If deriv is not a one dimensional numpy array of integers.

        """
        # pylint: disable=C0103
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
            if deriv is not None and not (
                isinstance(deriv, np.ndarray) and deriv.ndim == 1 and deriv.dtype == int
            ):
                raise TypeError("deriv must be given as a one dimensional numpy array of integers.")

        # if no derivatization
        if deriv is None:
            return self._olp(sd)
        # if derivatization
        # return np.array([self._olp_deriv(sd, i) for i in deriv])
        occ_indices = self.get_occupation_indices(sd)
        output = np.zeros(self.nparams)

        D = self.dimension  # noqa: N806
        K = self.nspatial  # noqa: N806
        for k in range(self.nspatial):
            n = occ_indices[k]
            deriv_block = self._olp_deriv_block(sd, k)
            if k == 0:
                start_index = D * n
                end_index = start_index + D
            elif k < K - 1:
                start_index = 4 * D + 4 * D ** 2 * (k - 1) + D ** 2 * n
                end_index = start_index + D ** 2
            else:
                start_index = 4 * D + 4 * D ** 2 * (k - 1) + D * n
                end_index = start_index + D
            output[start_index:end_index] = np.ravel(deriv_block)
        return output[deriv]

    def enable_cache(self, include_derivative=True):
        """Enable cache for the `_olp` and possibly `_olp_deriv_block` functions.

        The methods `_olp` and `_olp_deriv_block` are cached instead of `get_overlap` because it is
        assumed that `_olp` and `_olp_deriv_block` do not compute "trivial" results which can be
        obtained very quickly. These results are computed in `get_overlap` and the more complicated
        results are obtained via `_olp` and `_olp_deriv_block`.

        Parameters
        ----------
        include_deriv_blockative : bool
            Option to cached `_olp_deriv_block`.
            By default, `_olp_deriv_block` is cached alongside `_olp`.

        Notes
        -----
        Needs to access `memory` and `params`.

        """
        # assign memory allocated to cache
        if self.memory == np.inf:
            maxsize = 2 ** 30
        elif include_derivative:
            maxsize = int(self.memory / 8 / (self.dimension ** 2 + 1))
        else:
            maxsize = int(self.memory / 8)

        # store the cached function
        self._cache_fns = {}
        self._olp = functools.lru_cache(maxsize)(self._olp)
        self._cache_fns["overlap"] = self._olp
        if include_derivative:
            self._olp_deriv_block = functools.lru_cache(maxsize)(self._olp_deriv_block)
            self._cache_fns["overlap derivative"] = self._olp_deriv_block
