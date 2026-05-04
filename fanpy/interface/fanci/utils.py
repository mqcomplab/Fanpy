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