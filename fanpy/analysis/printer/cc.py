r"""Collection of functions to print Slater determinants and related features.

Functions
---------
exops(wfn, nprint, threshold) : {tuple, int, float}
    Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
exops_indices(wfn, nprint, threshold) : {tuple, int, float}
    Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied MO indices.

"""

from fanpy.tools import slater
import numpy as np


def exops(wfn, nprint=None, threshold=1e-8):
    """
    Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
    Creation and annihilation operators indices represented by (+) and (-).

    Parameters
    ----------
    wfn : BaseCC
        Wavefunction object containing T operators and CC parameters.
    nprint : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.
    """
    nspatial = wfn.nspatial
    nspin = wfn.nspin
    exops = wfn.exops.keys()
    params = wfn.params

    # Create a list of binary values (0 or 1) representing the reference wavefunction, reversed
    ref_wfn_str = list(format(wfn.refwfn, f"0{nspin}b")[::-1])

    # Prepare the list to store formatted excitation operators
    exops_str = []

    # For each excitation operator, modify the reference wavefunction string
    for exop, param in zip(exops, params):
        temp_wfn = ref_wfn_str.copy()  # Copy reference wavefunction
        for ind in exop:
            # Change to '-' if occupied, '+' if unoccupied
            if temp_wfn[ind] == "1":
                temp_wfn[ind] = "-"  # Annihilation (occupied orbital)
            elif temp_wfn[ind] == "0":
                temp_wfn[ind] = "+"  # Creation (unoccupied orbital)

        # Split the result into alpha and beta components
        alpha_str = "".join(temp_wfn[:nspatial])
        beta_str = "".join(temp_wfn[nspatial:])

        # Add the formatted excitation operator and parameter to the list
        exops_str.append([alpha_str, beta_str, param])

    # Sort by the absolute value of the parameter, in descending order
    exops_str.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Apply the threshold filtering (only keep parameters larger than the threshold)
    if threshold is not None:
        exops_str = [exop_str for exop_str in exops_str if abs(exop_str[-1]) > threshold]

    # Limit the output to the top nprint results (if specified)
    if nprint is not None:
        exops_str = exops_str[:nprint]

    # Print header
    print("> Coupled-Cluster Amplitudes represented by MO indices\n")
    print(f"{'Alpha':<{nspatial}}  |  {'Beta':<{nspatial}}  |  CC Parameter")
    print("-" * (nspatial * 2 + 23))

    # Print the formatted results
    for alpha, beta, param in exops_str:
        print(f"{alpha:<{nspatial}}  |  {beta:<{nspatial}}  |  {param:12.8f}")

    # Print a footer separator
    print("-" * (nspatial * 2 + 23) + "\n")


def exops_indices(wfn, nprint=None, threshold=1e-8):
    """Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    wfn : BaseCC
        Wavefunction object containing T operators and CC parameters.
    nprint : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.

    """
    nspatial = wfn.nspatial
    exops = wfn.exops.keys()
    params = wfn.params

    # Occupied reference indices (for both alpha and beta spins)
    occ_ref_inds = np.concatenate(
        (
            slater.occ_indices(slater.split_spin(wfn.refwfn, nspatial)[0]),  # Alpha
            slater.occ_indices(slater.split_spin(wfn.refwfn, nspatial)[1]) + nspatial,  # Beta (shifted by nspatial)
        )
    )

    # Create list of strings for alpha and beta spin indices (annihilation and creation operators)
    exops_str = []
    for exop, param in zip(exops, params):
        alpha_occ = [str(x) for x in exop if x < nspatial and x in occ_ref_inds]
        beta_occ = [str(x - nspatial) for x in exop if x >= nspatial and x in occ_ref_inds]
        alpha_vir = [str(x) for x in exop if x < nspatial and x not in occ_ref_inds]
        beta_vir = [str(x - nspatial) for x in exop if x >= nspatial and x not in occ_ref_inds]
        exops_str.append([alpha_occ, beta_occ, alpha_vir, beta_vir, param])

    # Sort by the absolute value of param (in descending order)
    exops_str.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Apply threshold filtering (if specified)
    if threshold is not None:
        exops_str = [exop_str for exop_str in exops_str if abs(exop_str[-1]) > threshold]

    # Applying nprint (if specified) to limit the number of determinants printed
    if nprint is not None:
        exops_str = exops_str[:nprint]

    # Determine maximum lengths for formatting
    alpha_occ_len = max(5, max(len(" ".join(exop_str[0])) for exop_str in exops_str))
    beta_occ_len = max(5, max(len(" ".join(exop_str[1])) for exop_str in exops_str))
    alpha_vir_len = max(5, max(len(" ".join(exop_str[2])) for exop_str in exops_str))
    beta_vir_len = max(5, max(len(" ".join(exop_str[3])) for exop_str in exops_str))

    occ_len = max(15, alpha_occ_len + beta_occ_len)
    vir_len = max(15, alpha_vir_len + beta_vir_len)

    # Print header
    print("> Coupled-Cluster Amplitudes represented by MO indices\n")
    print(f"{'Annihilation':<{vir_len}}  |  {'Creation':<{occ_len}}  |")
    print(
        f"{'Alpha':<{alpha_occ_len}}  |  {'Beta':<{beta_occ_len}}  |  {'Alpha':<{alpha_vir_len}}  |  {'Beta':<{beta_vir_len}}  |  CC Parameter"
    )
    print("-" * (vir_len + occ_len + 23))

    # Print the Slater determinant strings with aligned formatting
    for alpha_occ, beta_occ, alpha_vir, beta_vir, param in exops_str:
        print(
            f"{' '.join(alpha_occ):<{alpha_occ_len}}  |  {' '.join(beta_occ):<{beta_occ_len}}  |  "
            f"{' '.join(alpha_vir):<{alpha_vir_len}}  |  {' '.join(beta_vir):<{beta_vir_len}}  |  {param:12.8f}"
        )
    print("-" * (occ_len + vir_len + 23) + "\n")
