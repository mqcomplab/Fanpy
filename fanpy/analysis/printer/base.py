r"""Collection of functions to print Slater determinants and related features.

Functions
---------
print_sds(sds, nspatial) : {tuple, int}
    Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
sds_occ_indices(sds, nspatial) : {tuple, int}
    Print Slater determinants from a wavefunction as lists of occupied MO indices.
"""

from fanpy.tools import slater


def sds_occ(sds, nspatial=None):
    """Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.

    Parameters
    ----------
    sds : BaseWavefunction or tuple
        A tuple containing Slater determinants.
        It can be obtained from BaseWavefunction objects by the sds attribute.
    nspatial : int, optional
        Number of spatial orbitals.

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

    # Print header
    print("> Slater determinants represented by MO occupancies\n")
    print(f"{'Alpha':<{nspatial}}  |  {'Beta':<{nspatial}}")
    print("-" * (nspatial * 2 + 6))

    # Print the Slater determinant strings with aligned formatting
    for alpha, beta in sds_str:
        print(f"{alpha:<{nspatial}}  |  {beta:<{nspatial}}")
    print("-" * (nspatial * 2 + 6) + "\n")


def sds_occ_indices(sds, nspatial=None):
    """Print Slater determinants from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    sds : BaseWavefunction or tuple
        A tuple containing Slater determinants.
        It can be obtained from BaseWavefunction objects by the sds attribute.
    nspatial : int, optional
        Number of spatial orbitals.

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
