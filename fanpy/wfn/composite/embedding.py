from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.product import ProductWavefunction

import numpy as np
import warnings

class EmbeddedWavefunction(ProductWavefunction):
    """Embedding of multiple subsystems."""

    def __init__(self, nelec, nspin, indices_list, wfns, memory=None, disjoint=True):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec_list : list of int
            number of electrons in each subsystem.
        nspin_list : list of int
            number of spin orbitals in each subsystem.
        indices_list : list of list of ints
            list of spin orbitals in each subsystem.
        memory : {float, int, str, none}
            memory available for the wavefunction.
            default does not limit memory usage (i.e. infinite).
        refwfn : {int, none}
        params_list : {np.ndarray, basecc, none}

        """
        warnings.warn("Embedding wavefunction is experimental feature.", UserWarning)
        self.assign_nelec(nelec)
        self.assign_nspin(nspin)
        # FIXME: uses memory in each wavefunction
        # self.assign_memory(memory)

        if __debug__:
            nspatial = self.nspin // 2
            for indices in indices_list:
                if len(set(indices)) != len(indices):
                    raise ValueError("Indices for each subsystem must not repeat.")
                if len(set(i if i < nspatial else i - nspatial for i in indices)) != len(indices) // 2:
                    raise ValueError("Indices selected must correspond to alpha-beta orbital pairs.")
            if any(not isinstance(wfn, BaseWavefunction) for wfn in wfns):
                raise TypeError("Each wavefunction must be a instance of `BaseWavefunction`.")
            if set(i for indices in indices_list for i in indices) != set(range(nspin)):
                raise ValueError("Indices in the subsystems must span the indices in the system.")
            if len(wfns) != len(indices_list):
                raise ValueError("Number of wavefunctions must be equal to the number of index lists.")
            for wfn, indices in zip(wfns, indices_list):
                if wfn.nspin != len(indices):
                    raise ValueError(
                        "Number of spin orbitals of each wavefunction must be equal to the number "
                        "of indices associated with the wavefunctions."
                    )

            if disjoint:
                if nelec != sum(wfn.nelec for wfn in wfns):
                    raise ValueError(
                        "For disjoint partitions, number of electrons in the subsystems must add up"
                        " to the total number of electrons."
                    )
                if nspin != sum(wfn.nspin for wfn in wfns):
                    raise ValueError(
                        "For disjoint partitions, number of orbitals in the subsystems must add up"
                        " to the total number of orbitals."
                    )
                for i, indices1 in enumerate(indices_list):
                    indices1 = set(indices1)
                    for indices2 in indices_list[i + 1 :]:
                        if indices1.intersection(indices2):
                            raise ValueError(
                                "For disjoint partitions, indices for the subsytems cannot be "
                                "repeated in different subsytems."
                            )

        self.indices_list = indices_list

        dict_system_sub = {}
        dict_sub_system = {}
        for system_ind, indices in enumerate(indices_list):
            indices = sorted(indices)
            for i, j in zip(indices, range(len(indices))):
                if i not in dict_system_sub:
                    dict_system_sub[i] = [(system_ind, j)]
                else:
                    dict_system_sub[i].append((system_ind, j))
                dict_sub_system[(system_ind, j)] = i
        self.dict_system_sub = dict_system_sub
        self.dict_sub_system = dict_sub_system

        self.wfns = wfns

    @property
    def num_systems(self):
        return len(self.wfns)

    def split_sd(self, sd):
        sd_list = [0 for i in range(self.num_systems)]
        for i in slater.occ_indices(sd):
            for system_ind, j in self.dict_system_sub[i]:
                sd_list[system_ind] = slater.create(sd_list[system_ind], j)
        return sd_list

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Since the overlap can be drivatized with respect to the parameters of the underlying
        wavefunctions, the API of this method is different from the rest.

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
            waefunctions.
            If the provided indices is less than zero or greater than or equal to the number of the
            corresponding parameters.

        """
        sd_list = self.split_sd(sd)
        wfn_contrib = np.array([wfn.get_overlap(sd_system, deriv=None) for sd_system, wfn in zip(sd_list, self.wfns)])
        if deriv is None:
            return np.prod(wfn_contrib)

        if __debug__:
            if not (
                isinstance(deriv, tuple)
                and len(deriv) == 2
                and isinstance(deriv[0], BaseWavefunction)
                # and isinstance(deriv[1], np.ndarray)
                # and deriv[1].ndim == 1
                # and np.issubdtype(deriv[1].dtype, np.integer)
            ):
                raise TypeError(
                    "Derivative indices must be given as a 2-tuple whose first element is the "
                    "wavefunction and the second elment is the one-dimensional numpy array of "
                    "integer indices."
                )
            if deriv[0] not in self.wfns:
                raise ValueError(
                    "Selected wavefunction (for derivatization) is not one of the composite "
                    "wavefunction or its underlying wavefunctions."
                )
            # if deriv[0] == self and (np.any(deriv[1] < 0) or np.any(deriv[1] >= deriv[0].nparams)):
            #     raise ValueError(
            #         "Provided indices must be greater than or equal to zero and less than the "
            #         "number of parameters."
            #     )

        wfn, indices = deriv
        wfn_index = self.wfns.index(wfn)
        # NOTE: assume that parameters are not shared between wavefunctions
        output = np.prod(wfn_contrib[:wfn_index])
        output *= np.prod(wfn_contrib[wfn_index + 1 :])
        output *= wfn.get_overlap(sd_list[wfn_index], deriv=indices)
        return output

    def get_overlaps(self, sds, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Since the overlap can be drivatized with respect to the parameters of the underlying
        wavefunctions, the API of this method is different from the rest.

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
            waefunctions.
            If the provided indices is less than zero or greater than or equal to the number of the
            corresponding parameters.

        """
        sds_list = np.array([self.split_sd(sd) for sd in sds]).T
        # FIXME: MEMORY
        wfn_contrib = []
        for wfn, sds_system in zip(self.wfns, sds_list):
            if hasattr(wfn, "get_overlaps"):
                wfn_contrib.append(wfn.get_overlaps(sds_system, deriv=None))
            else:
                wfn_contrib.append([wfn.get_overlap(sd, deriv=None) for sd in sds_system])
        wfn_contrib = np.array(wfn_contrib)
        if deriv is None:
            return np.prod(wfn_contrib, axis=0)

        if __debug__:
            if not (
                isinstance(deriv, tuple)
                and len(deriv) == 2
                and isinstance(deriv[0], BaseWavefunction)
                # and isinstance(deriv[1], np.ndarray)
                # and deriv[1].ndim == 1
                # and np.issubdtype(deriv[1].dtype, np.integer)
            ):
                raise TypeError(
                    "Derivative indices must be given as a 2-tuple whose first element is the "
                    "wavefunction and the second elment is the one-dimensional numpy array of "
                    "integer indices."
                )
            if deriv[0] not in self.wfns:
                raise ValueError(
                    "Selected wavefunction (for derivatization) is not one of the composite "
                    "wavefunction or its underlying wavefunctions."
                )
            # if deriv[0] == self and (np.any(deriv[1] < 0) or np.any(deriv[1] >= deriv[0].nparams)):
            #     raise ValueError(
            #         "Provided indices must be greater than or equal to zero and less than the "
            #         "number of parameters."
            #     )

        wfn, indices = deriv
        wfn_index = self.wfns.index(wfn)
        # NOTE: assume that parameters are not shared between wavefunctions
        output = np.prod(wfn_contrib[:wfn_index], axis=0)
        output *= np.prod(wfn_contrib[wfn_index + 1 :], axis=0)
        if hasattr(wfn, "get_overlaps"):
            output = output[:, None] * wfn.get_overlaps(sds_list[wfn_index], deriv=indices)
        else:
            output = output[:, None] * np.array([wfn.get_overlap(sd, deriv=indices) for sd in sds_list[wfn_index]])
        return output
