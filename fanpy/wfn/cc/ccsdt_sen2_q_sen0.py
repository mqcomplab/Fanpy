"""Coupled Cluster SD with seniority 1 Triples and seniority 0 Quadruples."""

from itertools import combinations
from fanpy.tools import slater
from fanpy.wfn.cc.ccsdtq_sen0 import CCSDTQsen0


class CCSDTsen2Qsen0(CCSDTQsen0):
    r"""Coupled Cluster SD with seniority 2 Triples and seniority 0 Quadruples.

    .. math::

        \left| {{\Psi }_{CCSDTsen2Qsen0}} \right\rangle =\prod\limits_{i}{\left( 1+\sum\nolimits_{a}
        {t_{i}^{a}a_{a}^{\dagger }a_{{\bar{a}}}^{\dagger }a_{b}^{\dagger }a_{{\bar{b}}}^{\dagger }
        {{a}_{{\bar{j}}}}{{a}_{j}}{{a}_{{\bar{i}}}}{{a}_{i}}} \right)}\prod\limits_{i}
        {\left( 1+\sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }a_{{\bar{a}}}^{\dagger }a_{b}^
        {\dagger }{{a}_{j}}{{a}_{{\bar{i}}}}{{a}_{i}}} \right)}\prod\limits_{i}
        {\left( 1+\sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }a_{b}^{\dagger }{{a}_{j}}{{a}_{i}}}
        \right)}\prod\limits_{i}{\left( 1+\sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }{{a}_{i}}}
        \right)}\left| {{\Phi }_{0}} \right\rangle

    In this case the reference wavefunction can only be a single Slater determinant with
    seniority 0.

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
    exops : list of list of int
        Excitation operators given as lists of ints. The first half of indices correspond to
        indices to be annihilated, the second half correspond to indices to be created.
    refwfn : int
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
    __init__(self, nelec, nspin, memory=None, ngem=None, orbpairs=None, params=None)
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

    def assign_exops(self, indices=None):
        """Assign the excitation operators that will be used to construct the CC operator.

        Parameters
        ----------
        indices : None
            The allowed excitation operators are solely defined by the occupied and virtual
            orbitals of the given reference Slater determinant.

        Raises
        ------
        TypeError
            If `indices` is not None.

        Notes
        -----
        The excitation operators are given as a list of lists of ints.
        Each sub-list corresponds to an excitation operator.
        In each sub-list, the first half of indices corresponds to the indices of the
        spin-orbitals to annihilate, and the second half corresponds to the indices of the
        spin-orbitals to create.
        [a1, a2, ..., aN, c1, c2, ..., cN]

        """
        if indices is not None:
            raise TypeError(
                "Only the excitation operators constructed by default from "
                "the given reference Slater determinant are allowed"
            )
        else:
            exops = {}
            counter = 0
            ex_from = slater.occ_indices(self.refwfn).tolist()
            ex_to = [i for i in range(self.nspin) if i not in ex_from]

            for rank in self.ranks[:-2]:
                for annihilators in combinations(ex_from, rank):
                    for creators in combinations(ex_to, rank):
                        exop = []
                        for annihilator in annihilators:
                            exop.append(annihilator)
                        for creator in creators:
                            exop.append(creator)
                        exops[tuple(exop)] = counter
                        counter += 1

            # Seniority 2 triples
            for occ_alpha1 in ex_from[: len(ex_from) // 2]:
                for occ2 in [oorb for oorb in ex_from if oorb not in [occ_alpha1, occ_alpha1 + self.nspatial]]:
                    for virt_alpha1 in ex_to[: len(ex_to) // 2]:
                        for virt2 in [vorb for vorb in ex_to if vorb not in [virt_alpha1, virt_alpha1 + self.nspatial]]:
                            exop = [
                                occ_alpha1,
                                occ_alpha1 + self.nspatial,
                                occ2,
                                virt_alpha1,
                                virt_alpha1 + self.nspatial,
                                virt2,
                            ]
                            exops[tuple(exop)] = counter
                            counter += 1

            # Seniority 0 quadruples
            for occ_alpha1 in ex_from[: len(ex_from) // 2]:
                for occ_alpha2 in ex_from[ex_from.index(occ_alpha1) + 1 : len(ex_from) // 2]:
                    for virt_alpha1 in ex_to[: len(ex_to) // 2]:
                        for virt_alpha2 in ex_to[ex_to.index(virt_alpha1) + 1 : len(ex_to) // 2]:
                            exop = [
                                occ_alpha1,
                                occ_alpha1 + self.nspatial,
                                occ_alpha2,
                                occ_alpha2 + self.nspatial,
                                virt_alpha1,
                                virt_alpha1 + self.nspatial,
                                virt_alpha2,
                                virt_alpha2 + self.nspatial,
                            ]
                            exops[tuple(exop)] = counter
                            counter += 1
            self.exops = exops
