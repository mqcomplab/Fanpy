from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.embedding import EmbeddedWavefunction

import numpy as np


class FixedEmbeddedWavefunction(EmbeddedWavefunction):
    """Embedding of multiple subsystems."""

    def __init__(self, nelec, nelec_list, nspin, indices_list, wfns, memory=None, disjoint=True):
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
        super().__init__(nelec, nspin, indices_list, wfns, memory=memory, disjoint=disjoint)
        self.nelec_list = nelec_list
        # TODO: nelec in nelec_list cannot be 0

    def split_sd(self, sd):
        sd_list = [0 for i in range(self.num_systems)]
        sd_nelecs = [0 for i in range(self.num_systems)]
        for i in slater.occ_indices(sd):
            for system_ind, j in self.dict_system_sub[i]:
                sd_list[system_ind] = slater.create(sd_list[system_ind], j)
                sd_nelecs[system_ind] += 1
        for i in range(self.num_systems):
            if self.nelec_list[i] != sd_nelecs[i]:
                sd_list[i] = 0

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
        if 0 in sd_list and deriv is None:
            return 0

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
        if 0 in sd_list and deriv is not None:
            if isinstance(deriv[1], np.ndarray):
                return np.zeros(deriv[1].shape)
            else:
                return 0.0

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
        orig_sds_list = np.array([self.split_sd(sd) for sd in sds]).T
        # dimension 0 is for systems
        # dimension 1 is for slater determinants
        mask_nonzero_sds = np.where(np.array([0 not in sds for sds in orig_sds_list.T]))[0]
        sds_list = orig_sds_list[:, mask_nonzero_sds]

        # FIXME: MEMORY
        wfn_contrib = []
        for wfn, sds_system in zip(self.wfns, sds_list):
            if hasattr(wfn, "get_overlaps"):
                wfn_contrib.append(wfn.get_overlaps(sds_system, deriv=None))
                # overlaps = np.zeros(sds_system.shape)
                # mask = sds_system != 0
                # overlaps[mask] = wfn.get_overlaps(sds_system[mask], deriv=None)
                # wfn_contrib.append(overlaps)
            else:
                wfn_contrib.append([wfn.get_overlap(sd, deriv=None) for sd in sds_system])
        wfn_contrib = np.array(wfn_contrib)
        if deriv is None:
            output = np.zeros(orig_sds_list.shape[1])
            output[mask_nonzero_sds] = np.prod(wfn_contrib, axis=0)
            return output

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
        # FIXME: how to handle zeros?
        nonzero_output = np.prod(wfn_contrib[:wfn_index], axis=0)
        nonzero_output *= np.prod(wfn_contrib[wfn_index + 1 :], axis=0)
        if hasattr(wfn, "get_overlaps"):
            nonzero_output = nonzero_output[:, None] * wfn.get_overlaps(sds_list[wfn_index], deriv=indices)

            # overlaps = np.zeros(sds_system.shape)
            # mask = sds_system != 0
            # overlaps[mask] = wfn.get_overlaps(sds_system[mask], deriv=None)
            # wfn_contrib.append(overlaps)
        else:
            nonzero_output = nonzero_output[:, None] * np.array(
                [wfn.get_overlap(sd, deriv=indices) for sd in sds_list[wfn_index]]
            )

        output = np.zeros((orig_sds_list.shape[1], deriv[1].size))
        output[mask_nonzero_sds] = nonzero_output
        return output
