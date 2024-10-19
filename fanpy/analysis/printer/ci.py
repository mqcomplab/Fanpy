r"""Collection of functions to print Slater determinants and related features.

Functions
---------
sds_occ(wfn, nprint, threshold) : {tuple, int, float}
    Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
sds_occ_indices(wfn, nprint, threshold) : {tuple, int, float}
    Print Slater determinants from a wavefunction as lists of occupied MO indices.
"""

from fanpy.tools import slater


def sds_occ(wfn, nprint=None, threshold=1e-8):
    """Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.

    Parameters
    ----------
    wfn : CIWavefunction
        Wavefunction object containing Slater determinants and CI coefficients.
    nprint : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.

    """
    nspatial = wfn.nspatial
    sds = wfn.sds
    params = wfn.params

    # Create list of alpha and beta spin indices as zero-padded binary strings
    sds_str = [
        [
            format(slater.split_spin(sd, nspatial)[0], f"0{nspatial}b")[::-1],
            format(slater.split_spin(sd, nspatial)[1], f"0{nspatial}b")[::-1],
            param,
        ]
        for sd, param in zip(sds, params)
    ]

    # Sort by the param value
    sds_str.sort(key=lambda x: abs(x[-1]), reverse=True)  # Sort in descending order by param

    # Apply threshold filtering (if specified)
    if threshold is not None:
        sds_str = [sd_str for sd_str in sds_str if abs(sd_str[-1]) > threshold]

    # Applying nprint (if specified) to select the number of sds to be printed
    if nprint is not None:
        sds_str = sds_str[:nprint]

    # Print header
    print("> Slater determinants represented by MO occupancies\n")
    print(f"{'Alpha':<{nspatial}}  |  {'Beta':<{nspatial}}  |  CI Parameter")
    print("-" * (nspatial * 2 + 23))

    # Print the Slater determinant strings with aligned formatting
    for alpha, beta, param in sds_str:
        print(f"{alpha:<{nspatial}}  |  {beta:<{nspatial}}  |  {param:12.8f}")
    print("-" * (nspatial * 2 + 23) + "\n")


def sds_occ_indices(wfn, nprint=None, threshold=1e-8):
    """Print Slater determinants from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    wfn : CIWavefunction
        Wavefunction object containing Slater determinants and CI coefficients.
    nprint : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.

    """
    nspatial = wfn.nspatial
    sds = wfn.sds
    params = wfn.params

    # Create list of strings for alpha and beta spin indices
    sds_str = [
        [
            " ".join(map(str, slater.occ_indices(slater.split_spin(sd, nspatial)[0]))),
            " ".join(map(str, slater.occ_indices(slater.split_spin(sd, nspatial)[1]))),
            param,
        ]
        for sd, param in zip(sds, params)
    ]

    # Sort by the param value
    sds_str.sort(key=lambda x: abs(x[2]), reverse=True)  # Sort in descending order by param

    # Apply threshold filtering (if specified)
    if threshold is not None:
        sds_str = [sd_str for sd_str in sds_str if abs(sd_str[-1]) > threshold]

    # Applying nprint (if specified) to select the number of sds to be printed
    if nprint is not None:
        sds_str = sds_str[:nprint]

    # Find the maximum lengths for formatting
    alpha_len = max(5, max(len(sd_str[0]) for sd_str in sds_str))
    beta_len = max(5, max(len(sd_str[1]) for sd_str in sds_str))

    # Print header
    print("> Slater determinants represented by occupied MO indices\n")
    print(f"{'Alpha':<{alpha_len}}  |  {'Beta':<{beta_len}}  |  CI Parameter")
    print("-" * (alpha_len + beta_len + 23))

    # Print the Slater determinant strings with aligned formatting
    for alpha, beta, param in sds_str:
        print(f"{alpha:<{alpha_len}}  |  {beta:<{beta_len}}  |  {param:12.8f}")
    print("-" * (alpha_len + beta_len + 23) + "\n")
