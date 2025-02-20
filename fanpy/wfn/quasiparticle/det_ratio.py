"""Hard-coded determinant-ratio wavefunction."""

from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction

import numpy as np


class DeterminantRatio(BaseWavefunction):
    r"""Wavefunction whose overlap is a ratio of determinants.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    memory : float
        Memory available for the wavefunction.
    numerator_mask : np.ndarray
        Mask for selecting the matrices that correspond to the numerator.
    params : np.ndarray
        Parameters for each matrix have been flattened, and joined together.
        Each matrix is assumed to have shape, (nelec, nspin).

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
    matrix_shape : 2-tuple of int
        Shape of each matrix.
    matrix_size : int
        Size of each matrix.
    num_matrices : int
        Number of matrices.

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
        Loaad the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    """

    def __init__(self, nelec, nspin, memory=None, numerator_mask=None, params=None, enable_cache=True):
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
        numerator_mask : {np.ndarray, None}
            Mask for selecting the matrices that correspond to the numerator.
            Controls the number of matrices used in the wavefunction.
        params : {np.ndarray, None}
            Parameters for each matrix have been flattened, and joined together.
            Each matrix is assumed to have shape, (nelec, nspin).
        enable_cache : bool
            Option to cache the results of `_olp` and `_olp_deriv`.
            By default, `_olp` and `_olp_deriv` are cached.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_numerator_mask(numerator_mask)
        self.assign_params(params)
        if enable_cache:
            self.enable_cache()

    def assign_numerator_mask(self, numerator_mask=None):
        """Assign the mask for selecting matrices that correspond to the numerator.

        Parameters
        ----------
        numerator_mask : {np.ndarray, None}
            Boolean mask for selecting the matrices that correspond to the numerator.
            Default has two matrices, first for the numerator and second for the demominator.
            Controls the number of matrices used in the wavefunction.

        Raises
        ------
        TypeError
            If numerator_mask is not a boolean numpy array with one or more element.

        """
        if numerator_mask is None:
            numerator_mask = np.array([True, False])

        if __debug__ and not (
            isinstance(numerator_mask, np.ndarray) and numerator_mask.dtype == bool and numerator_mask.size > 0
        ):
            raise TypeError(
                "Mask for the numerator must be given as a boolean numpy array with one or more " "element."
            )

        self.numerator_mask = numerator_mask

    @property
    def matrix_shape(self):
        """Return the shape of each matrix.

        Returns
        -------
        matrix_shape : 2-tuple of int
            Shape of the matrix.
            Assumes each matrix has the same shape.

        """
        return (self.nelec, self.nspin)

    @property
    def matrix_size(self):
        """Return the size of each matrix.

        Returns
        -------
        matrix_size : int
            Size of the matrix.
            Assumes each matrix has the same shape.

        """
        nrow, ncol = self.matrix_shape
        return nrow * ncol

    @property
    def num_matrices(self):
        """Return the number of matrices.

        Returns
        -------
        num_matrices : int
            Number of matrices.

        """
        return self.numerator_mask.size

    # NOTE: all of the rows are assumed to be selected when the columns are selected.
    # TODO: select different columns depending on the matrix whose columns are selected
    def get_columns(self, sd, index):  # pylint: disable=C0103,W0613,R0201
        """Get the columns that correspond to the given Slater determinant for the given matrix.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
        index : int
            Index of the selected matrix.

        Returns
        -------
        col_indices : np.ndarray of int
            Indices of the columns that are associated with the given Slater determinant and the
            selected matrix.

        """
        return np.array(slater.occ_indices(sd))

    def get_matrix(self, index):
        """Get the matrix that correspond to the given index.

        Parameters
        ----------
        index : int
            Index of the selected matrix.

        Returns
        -------
        matrix : np.ndarray
            Selected matrix.

        Raises
        ------
        TypeError
            If given index is not an integer.
        ValueError
            If given index is not greater than or equal to zero and less than number of matrices.

        """
        if __debug__:
            if not isinstance(index, int):
                raise TypeError("index must be given as an integer.")
            if not 0 <= index < self.num_matrices:
                raise ValueError("index must be greater than or equal to zero and less than the number of " "matrices.")

        flat_matrix = self.params[index * self.matrix_size : (index + 1) * self.matrix_size]
        return flat_matrix.reshape(self.matrix_shape)

    def decompose_index(self, param_index):
        """Return the matrix, row and column indices that correspond to the given parameter index.

        Parameters
        ----------
        param_index : int
            Index of the parameter.

        Returns
        -------
        ind_matrix : int
            Index of the matrix that corresponds to the parameter.
        ind_row : int
            Row index of the matrix.
        ind_col : int
            Column index of the matrix.

        """
        _, ncol = self.matrix_shape
        ind_matrix = param_index // self.matrix_size
        param_index = param_index % self.matrix_size
        ind_row = param_index // ncol
        ind_col = param_index % ncol
        return ind_matrix, ind_row, ind_col

    # TODO: each matrix is assumed to have the same shape (nelec, nspin). It may need to be made
    #       more flexible (let them have different shape or let each matrix have different shapes
    #       from one another)
    # FIXME: can probably removed because it doesn't do much different from the parent assign_param
    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, DeterminantRatio, None}
            Parameters of the DeterminantRatio wavefunction.
            If DeterminantRatio instance is given, then the parameters of this instance are used.
            Default corresponds to the ground state HF wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        """
        if params is None:
            # NOTE: assume that same columns are selected for all matrices
            # NOTE: all of the rows are assumed to be selected when the columns are selected.

            # get ground slater determinant
            ground_sd = slater.ground(*self.matrix_shape)
            columns = self.get_columns(ground_sd, 0)

            # make matrix
            matrix = np.zeros(self.matrix_shape)
            matrix[np.arange(self.matrix_shape[0]), columns] = 1

            # make denominator
            denominator = np.copy(matrix)
            indices = [i for i in range(self.matrix_shape[1]) if i not in columns]
            denominator[:, indices] = np.random.rand(self.matrix_shape[0], len(indices))
            denominator /= np.linalg.norm(denominator, axis=0)

            # flatten and join together
            matrices = np.array([matrix] * self.num_matrices)
            matrices[np.logical_not(self.numerator_mask)] = denominator
            params = matrices.flatten()
        if isinstance(params, DeterminantRatio):
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
        # NOTE: all of the rows are assumed to be selected when the columns are selected.
        determinants = np.array(
            [np.linalg.det(self.get_matrix(i)[:, self.get_columns(sd, i)]) for i in range(self.num_matrices)]
        )
        numerator = np.prod(determinants[self.numerator_mask])
        denominator = np.prod(determinants[np.logical_not(self.numerator_mask)])
        return numerator / denominator

    def _olp_deriv(self, sd):  # pylint: disable=E0202
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        output = np.zeros(self.nparams)
        for deriv_matrix in range(self.num_matrices):
            # compute determinants of matrices that are not being derivatized
            # NOTE: all of the rows are assumed to be selected when the columns are selected.
            # ASSUME: selected matrix is square (i.e. has same number of electrons)
            determinants = np.array(
                [
                    np.linalg.det(self.get_matrix(i)[:, self.get_columns(sd, i)])
                    for i in range(self.num_matrices)
                    if i != deriv_matrix
                ]
            )
            new_numerator_mask = np.delete(self.numerator_mask, deriv_matrix)
            numerator = np.prod(determinants[new_numerator_mask])
            denominator = np.prod(determinants[np.logical_not(new_numerator_mask)])

            # derivatize selected matrix
            # NOTE: all of the rows are assumed to be selected when the columns are selected.
            # ASSUME: deriv_row is in rows and deriv_col is in cols (i.e. selected index corresponds
            # to an occupied orbital of the given Slater determinant)
            matrix = self.get_matrix(deriv_matrix)

            old_determinant = np.linalg.det(matrix[:, self.get_columns(sd, deriv_matrix)])
            for deriv_row in range(self.matrix_shape[0]):
                for deriv_col in self.get_columns(sd, deriv_matrix):
                    rows = np.arange(matrix.shape[0])
                    cols = self.get_columns(sd, deriv_matrix)
                    # mask for finding the deriv_row and deriv_col from rows and cols, respectively
                    rows_mask = rows != deriv_row
                    cols_mask = cols != deriv_col
                    # find sign that corresponds to the derivative
                    sign_row = (-1) ** np.where(np.logical_not(rows_mask))[0][0]
                    sign_col = (-1) ** np.where(np.logical_not(cols_mask))[0][0]
                    # filter out the deriv_row and deriv_col
                    rows = rows[rows_mask]
                    cols = cols[cols_mask]

                    minor = np.linalg.det(matrix[rows[:, None], cols[None, :]])
                    deriv_determinant = sign_row * sign_col * minor

                    index = deriv_matrix * self.matrix_size + deriv_row * self.matrix_shape[1] + deriv_col
                    # if derivatized matrix is a numerator
                    if self.numerator_mask[deriv_matrix]:
                        output[index] = deriv_determinant * numerator / denominator
                    # if derivatized matrix is a denominator
                    else:
                        output[index] = numerator / denominator * (-1) * old_determinant ** (-2) * deriv_determinant
        return output

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Parameters
        ----------
        sd : int
            Slater Determinant against which the overlap is taken.
        deriv : {np.ndarray, None}
            Indices of the parameters with respect to which the overlap is derivatized.
            Default returns the overlap without derivatization.

        Returns
        -------
        overlap : {float, np.ndarray}
            Overlap (or derivative of the overlap) of the wavefunction with the given Slater
            determinant.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
            If deriv is not a one dimensional numpy array of integers.

        """
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
            if deriv is not None and not (isinstance(deriv, np.ndarray) and deriv.ndim == 1 and deriv.dtype == int):
                raise TypeError("deriv must be given as a one dimensional numpy array of integers.")

        if slater.total_occ(sd) != self.nelec:
            return 0.0

        # if no derivatization
        if deriv is None:
            return self._olp(sd)
        # if derivatization
        return self._olp_deriv(sd)[deriv]
