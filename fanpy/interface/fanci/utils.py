"""Utility functions for the PyCI interface"""

import numpy as np

from fanpy.tools import slater


def convert_pyci_occs_to_fanpy_sds(occs_array: np.ndarray, nspatial: int):
    """Functiont to convert occupation arrays into Slater determinants represented by integers.  

    Parameters
    ----------
    occs_array : np.ndarray
        Occupation array as a numpy array. This can be either the orbital occupation, or the spinorbital occupation vector. 
    nspatial : int
        number of spatial orbitals. 
    
    Returns
    -------
    sds : list of ints
        The Slater Determinants represented as int 

    """
    #todo: check if length of occ array == nspatial!
    
    sds = []
    if isinstance(occs_array[0, 0], np.ndarray): # if pspace generated with FCI
        for i, occs in enumerate(occs_array):
            # FIXME: CHECK IF occs IS BOOLEAN OR INTEGERS
            # convert occupation vector to sd
            if occs.dtype == bool:
                occs = np.where(occs)[0]
            sd = slater.create(0, *occs[0])
            sd = slater.create(sd, *(occs[1] + nspatial))
            sds.append(sd)
    else: # if pspace generated with DOCI
        for i, occs in enumerate(occs_array):
            if occs.dtype == bool:
                occs = np.where(occs)
            sd = slater.create(0, *occs)
            sd = slater.create(sd, *(occs + nspatial))
            sds.append(sd)
    return sds

# todo: create utility function to calculate variational energy objective:
# This is the objective function from the old interface class for objective type energy. 
# Note: we do not have the features yet to implement this, thus it is staying here for now, until development on the variational interface is done. 
#   else:
#             # NOTE: ignores energy and constraints
#             # Allocate objective vector
#             output = np.zeros(self.nproj, dtype=pyci.c_double)

#             # Compute overlaps of determinants in sspace:
#             #
#             #   c_m
#             #
#             ovlp = self.compute_overlap(x[:-1], "S")

#             # Compute objective function:
#             #
#             #   f_n = (\sum_n <\Psi|n> <n|H|\Psi>) / \sum_n <\Psi|n> <n|\Psi>
#             #
#             # Note: we update ovlp in-place here
#             self.ci_op(ovlp, out=output)
#             output = np.sum(output * ovlp[: self.nproj])
#             output /= np.sum(ovlp[: self.nproj] ** 2)
#             self.print_queue["Electronic Energy"] = output
#             if self.step_print:
#                 print("(Mid Optimization) Electronic Energy: {}".format(self.print_queue["Electronic Energy"]))