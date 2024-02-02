"""Base class for excitation Coupled Cluster wavefunctions."""
import numpy as np
import functools
import itertools
from itertools import combinations, permutations, product, repeat
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.tools.sd_list import sd_list


class Overlap_1(BaseWavefunction):
    r"""

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    ranks : list of ints
        Ranks of the excitation operators.
    exops : dictionary of tuple to int
        Excitation operators given as lists of ints. The first half of indices correspond to
        indices to be annihilated, the second half correspond to indices to be created.
    refwfn : {CIWavefunction, int}
        Reference wavefunction upon which the CC operator will act.
    exop_combinations : dict
        dictionary, the keys are tuples with the indices of annihilation and creation
        operators, and the values are the excitation operators that allow to excite from the
        annihilation to the creation operators.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    template_params : np.ndarray
        Default parameters of the wavefunction.
    nexops : int
        Number of excitation operators.
    nranks : int
        Number of allowed ranks.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, ranks=None, indices=None,
             refwfn=None, params=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_ranks(self, ranks=None)
        Assign the allowed excitation ranks.
    assign_exops(self, exops=None)
        Assign the allowed excitation operators.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign the parameters of the CC wavefunction.
    get_ind(self, exop) : int
        Return the parameter index that corresponds to a given excitation operator.
    get_exop(self, ind) : list of int
        Return the excitation operator that corresponds to a given parameter index.
    product_amplitudes(self, inds, deriv=None) : float
        Return the product of the CC amplitudes of the coefficients corresponding to
        the given indices.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.
    generate_possible_exops(self, a_inds, c_inds):
        Assign the excitation operators that can excite from the given indices to be annihilated
        to the given indices to be created.

    """
    def __init__(self, nelec, nspin, memory=None, sds=None, params=None,):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        ranks : {int, list of int, None}
            Ranks of the excitation operators (in increasing order).
            If int is provided, it takes all the ranks lower than that.
            Default is None, which is equivalent to taking ranks=self.nelec.
        indices : {list of list of ints, None}
            List of lists containing the indices ot the spin-orbitals to annihilate and create.
            The first sub-list contains indices of orbitals to annihilate.
            The second sub-list contains indices of orbitals to create.
            Default generates all possible indices according to the given ranks.
        refwfn : {CIWavefunction, int, None}
            Reference wavefunction upon which the CC operator will act.
        params : {np.ndarray, BaseCC, None}
            1-vector of CC amplitudes.
        exop_combinations : dict
            dictionary, the keys are tuples with the indices of annihilation and creation
            operators, and the values are the excitation operators that allow to excite from the
            annihilation to the creation operators.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_sds(sds=sds)
        self.dict_sd_index = {sd: i for i, sd in enumerate(self.sds)}
        self.assign_params(params=params)
        
    @property
    def nsd(self):
        """Return number of Slater determinants.

        Returns
        -------
        nsd : int
            Number of Slater determinants.

        """
        return len(self.sds)
    
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

        """
        if params is None:
            params = np.zeros(len(self.sds))
            params[0] = 1

        super().assign_params(params=params, add_noise=add_noise)

    def assign_sds(self, sds=None):
        """Assign the list of Slater determinants from which the CI wavefunction is constructed.

        Parameters
        ----------
        sds : iterable of int
            List of Slater determinants.

        Raises
        ------
        TypeError
            If sds is not iterable.
            If a Slater determinant is not an integer.
        ValueError
            If an empty iterator was provided.
            If a Slater determinant does not have the correct number of electrons.
            If a Slater determinant does not have the correct spin.
            If a Slater determinant does not have the correct seniority.

        Notes
        -----
        Must have `nelec`, `nspin`, `spin` and `seniority` defined for default behaviour and type
        checking.

        """
        # pylint: disable=C0103
        # FIXME: terrible memory usage
        if sds is None:
            sds = sd_list(
                self.nelec,
                self.nspin,
                num_limit=None,
                exc_orders=None,
                spin=self.spin,
                seniority=self.seniority,
            )

        if __debug__:
            # FIXME: no check for repeated entries
            if not hasattr(sds, "__iter__"):
                raise TypeError("Slater determinants must be given as an iterable")
            sds, temp = itertools.tee(sds, 2)
            sds_is_empty = True
            for sd in temp:
                sds_is_empty = False
                if not slater.is_sd_compatible(sd):
                    raise TypeError("Slater determinant must be given as an integer.")
                if slater.total_occ(sd) != self.nelec:
                    raise ValueError(
                        "Slater determinant, {0}, does not have the correct number of "
                        "electrons, {1}".format(bin(sd), self.nelec)
                    )
                if isinstance(self.spin, float) and slater.get_spin(sd, self.nspatial) != self.spin:
                    raise ValueError(
                        "Slater determinant, {0}, does not have the correct spin, {1}".format(
                            bin(sd), self.spin
                        )
                    )
                if (
                    isinstance(self.seniority, int)
                    and slater.get_seniority(sd, self.nspatial) != self.seniority
                ):
                    raise ValueError(
                        "Slater determinant, {0}, does not have the correct seniority, {1}".format(
                            bin(sd), self.seniority
                        )
                    )
            if sds_is_empty:
                raise ValueError("No Slater determinants were provided.")

        self.sds = tuple(sds)
        
        
    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the CI wavefunction with a Slater determinant.

            The overlap of the CI wavefunction with a Slater determinant is the coefficient of that
            Slater determinant in the wavefunction.

            .. math::

                 \left< \Phi_i \middle| \Psi \right> = c_i

            where

            .. math::

                 \left| \Psi \right> = \sum_i c_i \left| \Phi_i \right>

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

        """
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
        # pylint:disable=R1705
        if deriv is None:
            try:
                return self.params[self.dict_sd_index[sd]]
            except KeyError:
                # FIXME: product hack
                return self.default_val

        output = np.zeros(self.nparams)
        try:
            output[self.dict_sd_index[sd]] = 1.0
        except KeyError:
            pass
        return output[deriv]
