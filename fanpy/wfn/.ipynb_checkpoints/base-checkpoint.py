"""Parent class of the wavefunctions."""
import abc
import functools

import numpy as np


class BaseWavefunction:
    r"""Base wavefunction class.

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

    Abstract Methods
    ----------------
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    """

    def __init__(self, nelec, nspin, memory=None, params=None):  # pylint: disable=W0613
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
        params : np.ndarray
            Parameters of the wavefunction.

        """
        # pylint: disable=W0231
        self.assign_nelec(nelec)
        self.assign_nspin(nspin)
        self.assign_memory(memory)
        # assign_params not included because it depends on template_params, which may involve
        # more attributes than is given above
        # self.assign_params(params)
        self.probable_sds = {}
        self.olp_threshold = 42
        self.pspace_norm = set()

    @property
    def nspatial(self):
        """Return the number of spatial orbitals.

        Returns
        -------
        nspatial : int
            Number of spatial orbitals.

        """
        return self.nspin // 2

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.params.size

    @property
    def spin(self):
        r"""Return the spin of the wavefunction.

        .. math::

            \frac{1}{2}(N_\alpha - N_\beta)

        Returns
        -------
        spin : float
            Spin of the wavefunction.

        Notes
        -----
        `None` means that all possible spins are allowed.

        """
        return None

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        Seniority of a Slater determinant is its number of unpaired electrons. The seniority of the
        wavefunction is the expected number of unpaired electrons.

        Returns
        -------
        seniority : int
            Seniority of the wavefunction.

        Notes
        -----
        `None` means that all possible seniority are allowed.

        """
        return None

    @property
    def dtype(self):
        """Return the data type of the wavefunction.

        Returns
        -------
        dtype

        """
        return self.params.dtype

    def assign_nelec(self, nelec):
        """Assign the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.

        """
        if __debug__:
            if not isinstance(nelec, int):
                raise TypeError("Number of electrons must be an integer.")
            if nelec <= 0:
                raise ValueError("Number of electrons must be a positive integer.")
        self.nelec = nelec

    def assign_nspin(self, nspin):
        """Assign the number of spin orbitals.

        Parameters
        ----------
        nspin : int
            Number of spin orbitals

        Raises
        ------
        TypeError
            If number of spin orbitals is not an integer.
        ValueError
            If number of spin orbitals is not a positive number.
            If number of spin orbitals is odd.

        """
        if __debug__:
            if not isinstance(nspin, int):
                raise TypeError("Number of spin orbitals must be an integer.")
            if nspin <= 0:
                raise ValueError("Number of spin orbitals must be a positive integer.")
            if nspin % 2 == 1:
                raise ValueError("Number of spin orbitals must be even.")
        self.nspin = nspin

    def assign_memory(self, memory=None):
        """Assign memory available for the wavefunction.

        Parameters
        ----------
        memory : {int, str, None}
            Memory available for the wavefunction.
            If number is provided, it is the number of bytes.
            If string is provided, it should end iwth either "mb" or "gb" to specify the units.

        Raises
        ------
        ValueError
            If memory is given as a string and does not end with "mb" or "gb".
        TypeError
            If memory is not given as a None, int, float, or string.

        """
        if memory is None:
            memory = np.inf
        elif isinstance(memory, (int, float)):
            memory = float(memory)
        elif isinstance(memory, str):
            if "mb" in memory.lower():
                memory = 1e6 * float(memory.rstrip("mb ."))
            elif "gb" in memory.lower():
                memory = 1e9 * float(memory.rstrip("gb ."))

        if __debug__:
            if isinstance(memory, str):
                raise ValueError('Memory given as a string should end with either "mb" or "gb".')
            if not isinstance(memory, (int, float, str)):
                raise TypeError("Memory should be given as a `None`, int, float, or string.")

        self.memory = memory

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        Raises
        ------
        NotImplementedError
            If `params` is None.

        """
        if params is None:
            raise NotImplementedError("Default parameters have not been implemented.")

        self.params = params

        # add random noise
        if add_noise:
            # set scale
            scale = 0.2 / self.nparams
            self.params += scale * (np.random.rand(*self.params.shape) - 0.5)
            if self.dtype in [complex, np.complex128]:
                self.params += (
                    0.01j * scale * (np.random.rand(*self.params.shape).astype(complex) - 0.5)
                )

    def save_params(self, filename):
        """Save parameters associated with the wavefunction.

        Parameters
        ----------
        filename : str

        """
        np.save(filename, self.params)

    def enable_cache(self, include_derivative=True):
        """Enable cache for the `_olp` and possibly `_olp_deriv` functions.

        The methods `_olp` and `_olp_deriv` are cached instead of `get_overlap` because it is
        assumed that `_olp` and `_olp_deriv` do not compute "trivial" results which can be obtained
        very quickly. These results are computed in `get_overlap` and the more complicated results
        are obtained via `_olp` and `_olp_deriv`.

        Parameters
        ----------
        include_derivative : bool
            Option to cached `_olp_deriv`.
            By default, `_olp_deriv` is cached alongside `_olp`.

        Notes
        -----
        Needs to access `memory` and `params`.

        """
        # pylint: disable=C0103
        # assign memory allocated to cache
        if self.memory == np.inf:
            maxsize = 2 ** 30
        elif include_derivative:
            maxsize = int(self.memory / 8 / (self.nparams + 1))
        else:
            maxsize = int(self.memory / 8)

        # store the cached function
        self._cache_fns = {}
        self._olp = functools.lru_cache(maxsize)(self._olp)
        self._cache_fns["overlap"] = self._olp
        if include_derivative:
            self._olp_deriv = functools.lru_cache(maxsize)(self._olp_deriv)
            self._cache_fns["overlap derivative"] = self._olp_deriv

    def _olp(self, sd):  # pylint: disable=E0202
        """Calculate the nontrivial overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.

        Returns
        -------
        olp : {float, complex}
            Overlap of the current instance with the given Slater determinant.

        Notes
        -----
        Nontrivial overlap would be an overlap whose value must be computed rather than trivially
        obtained by some set of rules (for example, a seniority zero wavefunction will have a zero
        overlap with a nonseniority zero Slater determinant).

        """
        # pylint: disable=C0103
        raise NotImplementedError("Overlap (for caching) has not been implemented.")

    def _olp_deriv(self, sd):  # pylint: disable=E0202
        """Calculate the nontrivial derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.

        Returns
        -------
        olp_deriv : np.ndarray
            Derivatives of the overlap with respect to each parameter.

        Notes
        -----
        Nontrivial derivative of the overlap would be a derivative that must be computed rather than
        obtained by some set of rules (for example, derivative of the overlap with respect to a
        parameter that is not involved in the overlap would be zero).

        """
        # pylint: disable=C0103
        raise NotImplementedError("Derivative of overlap (for caching) has not been implemented.")

    def clear_cache(self, key=None):
        """Clear the cache.

        Parameters
        ----------
        key : str
            Key to the cached function in _cache_fns.
            Default clears all cached functions.

        """
        if hasattr(self, "_cache_fns"):
            if key is None:
                for func in self._cache_fns.values():
                    func.cache_clear()
            else:
                self._cache_fns[key].cache_clear()

    @abc.abstractmethod
    def get_overlap(self, sd, deriv=None):  # pylint: disable=C0103
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

        Notes
        -----
        This method is different from `_olp` and `_olp_deriv` because it would do the necessary
        checks to ensure that only the nontrivial arguments get passed to the `_olp` and
        `_olp_deriv`. Since the outputs of `_olp` and `_olp_deriv` is cached, we can minimize the
        number of values cached this way.

        """

    def normalize(self, refwfn=None):
        pass
