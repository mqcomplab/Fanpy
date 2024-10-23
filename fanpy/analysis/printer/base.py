r"""Collection of functions to print Slater determinants and related features.

Functions
---------
sds_occ(sds, nspatial, nprint, threshold) : {tuple, int, int, float}
    Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
sds_occ_indices(sds, nspatial) : {tuple, int}
    Print Slater determinants from a wavefunction as lists of occupied MO indices.

"""

from fanpy.tools import slater
import numpy as np


# TODO: Check for which BaseWavefunctions can be applied
def sds_occ(sds, nspatial=None, nprint=None):
    """Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.

    Parameters
    ----------
    sds : BaseWavefunction or tuple
        A tuple containing Slater determinants.
        It can be obtained from BaseWavefunction objects by the sds attribute.
    nspatial : int, optional
        Number of spatial orbitals.
    nprint : int, optional
        Number of determinants to print (if specified).

    """
    # Check if sds is an instance of BaseWavefunction, if so, use its `sds` attribute
    if hasattr(sds, "sds"):
        sds = sds.sds
        nspatial = sds.nspatial

    # Create list of alpha and beta spin indices as zero-padded binary strings
    sds_str = [
        [
            format(slater.split_spin(sd, nspatial)[0], f"0{nspatial}b")[::-1],
            format(slater.split_spin(sd, nspatial)[1], f"0{nspatial}b")[::-1],
        ]
        for sd in sds
    ]

    # Sort by the param value
    sds_str.sort(key=lambda x: abs(x[2]), reverse=True)  # Sort in descending order by param

    # Applying nprint (if specified) to select the number of sds to be printed
    if nprint is not None:
        sds_str = sds_str[:nprint]

    # Print header
    print("> Slater determinants represented by MO occupancies\n")
    print(f"{'Alpha':<{nspatial}}  |  {'Beta':<{nspatial}}")
    print("-" * (nspatial * 2 + 6))

    # Print the Slater determinant strings with aligned formatting
    for alpha, beta in sds_str:
        print(f"{alpha:<{nspatial}}  |  {beta:<{nspatial}}")
    print("-" * (nspatial * 2 + 6) + "\n")


def sds_occ_indices(sds, nspatial=None, nprint=None):
    """Print Slater determinants from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    sds : BaseWavefunction or tuple
        A tuple containing Slater determinants.
        It can be obtained from BaseWavefunction objects by the sds attribute.
    nspatial : int, optional
        Number of spatial orbitals.
    nprint : int, optional
        Number of determinants to print (if specified).

    """
    # Check if sds is an instance of BaseWavefunction, if so, use its `sds` attribute
    if hasattr(sds, "sds"):
        sds = sds.sds
        nspatial = sds.nspatial

    # Create list of strings for alpha and beta spin indices
    sds_str = [
        [
            " ".join(map(str, slater.occ_indices(slater.split_spin(sd, nspatial)[0]))),
            " ".join(map(str, slater.occ_indices(slater.split_spin(sd, nspatial)[1]))),
        ]
        for sd in sds
    ]

    # Sort by the param value
    sds_str.sort(key=lambda x: abs(x[2]), reverse=True)  # Sort in descending order by param

    # Applying nprint (if specified) to select the number of sds to be printed
    if nprint is not None:
        sds_str = sds_str[:nprint]

    # Find the maximum lengths for formatting
    alpha_len = max(len(sd_str[0]) for sd_str in sds_str)
    beta_len = max(len(sd_str[1]) for sd_str in sds_str)

    # Print header
    print("> Slater determinants represented by occupied MO indices\n")
    print(f"{'Alpha':<{alpha_len}}  |  {'Beta':<{beta_len}}")
    print("-" * (alpha_len + beta_len + 6))

    # Print the Slater determinant strings with aligned formatting
    for alpha, beta in sds_str:
        print(f"{alpha:<{alpha_len}}  |  {beta:<{beta_len}}")
    print("-" * (alpha_len + beta_len + 6) + "\n")


def exops(sds, ref_wfn=None, nspatial=None, nprint=None):
    """
    Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
    Creation and annihilation operators indices represented by (+) and (-).

    Parameters
    ----------
    sds : BaseCC or tuple
        A tuple containing T operators.
        It can be obtained from BaseWavefunction objects by the sds attribute.
    nspatial : int, optional
        Number of spatial orbitals.
    nprint : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.
    """
    # Check if exops is an instance of BaseCC, if so, use its `exops` attribute
    if hasattr(exops, "exops"):
        exops = exops.exops.keys()
        nspatial = exops.nspatial
    nspin = nspatial * 2

    # Create a list of binary values (0 or 1) representing the reference wavefunction, reversed
    ref_wfn_str = list(format(ref_wfn, f"0{nspin}b")[::-1])

    # Prepare the list to store formatted excitation operators
    exops_str = []

    # For each excitation operator, modify the reference wavefunction string
    for exop in exops:
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
        exops_str.append([alpha_str, beta_str])

    # Sort by the absolute value of the parameter, in descending order
    exops_str.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Limit the output to the top nprint results (if specified)
    if nprint is not None:
        exops_str = exops_str[:nprint]

    # Print header
    print("> Coupled-Cluster Amplitudes represented by MO indices\n")
    print(f"{'Alpha':<{nspatial}}  |  {'Beta':<{nspatial}}")
    print("-" * (nspatial * 2 + 6))

    # Print the formatted results
    for alpha, beta in exops_str:
        print(f"{alpha:<{nspatial}}  |  {beta:<{nspatial}}")

    # Print a footer separator
    print("-" * (nspatial * 2 + 6) + "\n")


def exops_indices(wfn, nspatial=None, nprint=None):
    """Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    wfn : BaseCC
        Wavefunction object containing T operators and CC parameters.
    nspatial : int, optional
        Number of spatial orbitals.
    nprint : int, optional
        Number of determinants to print (if specified).

    """
    # Check if exops is an instance of BaseCC, if so, use its `exops` attribute
    if hasattr(exops, "exops"):
        exops = exops.exops.keys()
        nspatial = exops.nspatial

    # Occupied reference indices (for both alpha and beta spins)
    occ_ref_inds = np.concatenate(
        (
            slater.occ_indices(slater.split_spin(wfn.refwfn, nspatial)[0]),  # Alpha
            slater.occ_indices(slater.split_spin(wfn.refwfn, nspatial)[1]) + nspatial,  # Beta (shifted by nspatial)
        )
    )

    # Create list of strings for alpha and beta spin indices (annihilation and creation operators)
    exops_str = []
    for exop in exops:
        alpha_occ = [str(x) for x in exop if x < nspatial and x in occ_ref_inds]
        beta_occ = [str(x - nspatial) for x in exop if x >= nspatial and x in occ_ref_inds]
        alpha_vir = [str(x) for x in exop if x < nspatial and x not in occ_ref_inds]
        beta_vir = [str(x - nspatial) for x in exop if x >= nspatial and x not in occ_ref_inds]
        exops_str.append([alpha_occ, beta_occ, alpha_vir, beta_vir])

    # Sort by the absolute value of param (in descending order)
    exops_str.sort(key=lambda x: abs(x[-1]), reverse=True)

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
    print(f"{'Annihilation':<{vir_len}}  |  {'Creation':<{occ_len}}")
    print(
        f"{'Alpha':<{alpha_occ_len}}  |  {'Beta':<{beta_occ_len}}  |  {'Alpha':<{alpha_vir_len}}  |  {'Beta':<{beta_vir_len}}"
    )
    print("-" * (vir_len + occ_len + 6))

    # Print the Slater determinant strings with aligned formatting
    for alpha_occ, beta_occ, alpha_vir, beta_vir, param in exops_str:
        print(
            f"{' '.join(alpha_occ):<{alpha_occ_len}}  |  {' '.join(beta_occ):<{beta_occ_len}}"
            f"{' '.join(alpha_vir):<{alpha_vir_len}}  |  {' '.join(beta_vir):<{beta_vir_len}}"
        )
    print("-" * (occ_len + vir_len + 6) + "\n")
