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

# todo: create utility function to calculate variational energy objective derivative:
# This is the compute jacobian function from the old interface class for objective type energy. 
# Note: we do not have the features yet to implement this, thus it is staying here for now, until development on the variational interface is done. 
        # else: #todo: move to utility file. 
        #     # NOTE: ignores energy and constraints
        #     # Allocate Jacobian matrix (in transpose memory order)
        #     output = np.zeros((self.nproj, self.nactive), order="F", dtype=pyci.c_double)
        #     integrals = np.zeros(self.nproj, dtype=pyci.c_double)

        #     # Compute Jacobian:
        #     #
        #     #   J_{nk} = d(<n|H|\Psi>)/d(p_k) - E d(<n|\Psi>)/d(p_k) - dE/d(p_k) <n|\Psi>
        #     #   J_{nk} = (\sum_n d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) / \sum_n <\Psi|n>^2 -
        #     #            (\sum_n <\Psi|n> <n|H|\Psi>) / (\sum_n <\Psi|n> <n|\Psi>)^2 * (2 \sum_n <\Psi|n>)
        #     #   J_{nk} = ((\sum_n d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) (\sum_n <\Psi|n>^2)
        #     #             - (\sum_n <\Psi|n> <n|H|\Psi>) * (2 \sum_n <\Psi|n> d<\Psi|n>))
        #     #            / (\sum_n <\Psi|n>^2)^2
        #     #   J_{nk} = ((\sum_n d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) N
        #     #             - H * (2 \sum_n <\Psi|n> d<\Psi|n>))
        #     #            / N^2
        #     #   J_{nk} = (\sum_n N (d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) - 2 H <\Psi|n> d<\Psi|n>)
        #     #            / N^2
        #     #
        #     # Compute overlap derivatives in sspace:
        #     #
        #     #   d(c_m)/d(p_k)
        #     #
        #     overlaps = self.compute_overlap(x[:-1], "S")
        #     norm = np.sum(overlaps[: self.nproj] ** 2)
        #     self.ci_op(overlaps, out=integrals)
        #     energy_integral = np.sum(overlaps[: self.nproj] * integrals)

        #     d_ovlp = self.compute_overlap_deriv(x[:-1], "S")

        #     # Iterate over remaining columns of Jacobian and d_ovlp
        #     for output_col, d_ovlp_col in zip(output.transpose(), d_ovlp.transpose()):
        #         #
        #         # Compute each column of the Jacobian:
        #         #
        #         #   d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)
        #         #
        #         #   E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
        #         #
        #         # Note: we update d_ovlp in-place here
        #         self.ci_op(d_ovlp_col, out=output_col)
        #         output_col *= overlaps[: self.nproj]
        #         output_col += d_ovlp_col[: self.nproj] * integrals
        #         output_col *= norm
        #         output_col -= 2 * energy_integral * overlaps[: self.nproj] * d_ovlp_col[: self.nproj]
        #         output_col /= norm**2
        #     output = np.sum(output, axis=0)
        #     self.print_queue["Norm of the gradient of the energy"] = np.linalg.norm(output)
        #     if self.step_print:
        #         print(
        #             "(Mid Optimization) Norm of the gradient of the energy: {}".format(
        #                 self.print_queue["Norm of the gradient of the energy"]
        #             )
        #         )