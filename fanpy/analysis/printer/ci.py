r"""Collection of functions to print Slater determinants and related features.

Functions
---------
print_determinants(wavefunction, max_print, threshold) : {CIWavefunction, int, float}
    Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
print_determinants_indices(wavefunction, max_print, threshold) : {CIWavefunction, int, float}
    Print Slater determinants from a wavefunction as lists of occupied MO indices.
"""


def print_determinants(wfn, max_print=None, threshold=1e-8):
    """Print Slater determinants from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.

    Parameters
    ----------
    wfn : CIWavefunction
        Wavefunction object containing Slater determinants and CI coefficients.
    max_print : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.

    """
    from fanpy.tools import slater

    n_spatial_orbitals = wfn.nspatial
    slater_determinants = wfn.sds
    ci_params = wfn.params

    # Create list of alpha and beta spin occupancies as zero-padded binary strings
    sds_occupancies = [
        [
            format(slater.split_spin(sd, n_spatial_orbitals)[0], f"0{n_spatial_orbitals}b")[::-1],
            format(slater.split_spin(sd, n_spatial_orbitals)[1], f"0{n_spatial_orbitals}b")[::-1],
            param,
        ]
        for sd, param in zip(slater_determinants, ci_params)
    ]

    # Sort by the absolute value of CI parameters, descending
    sds_occupancies.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Apply threshold filtering (if specified)
    if threshold is not None:
        sds_occupancies = [sd_occ for sd_occ in sds_occupancies if abs(sd_occ[-1]) > threshold]

    # Limit output to max_print (if specified)
    if max_print is not None:
        sds_occupancies = sds_occupancies[:max_print]

    # Print header
    print("\n> Slater determinants represented by MO occupancies\n")
    print(f"{'Alpha':<{n_spatial_orbitals}}  |  {'Beta':<{n_spatial_orbitals}}  |  CI Parameter")
    print("-" * (n_spatial_orbitals * 2 + 23))

    # Print formatted occupancies
    for alpha, beta, param in sds_occupancies:
        print(f"{alpha:<{n_spatial_orbitals}}  |  {beta:<{n_spatial_orbitals}}  |  {param:12.8f}")
    print("-" * (n_spatial_orbitals * 2 + 23) + "\n")


def print_determinants_indices(wfn, max_print=None, threshold=1e-8):
    """Print Slater determinants from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    wfn : CIWavefunction
        Wavefunction object containing Slater determinants and CI coefficients.
    max_print : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.

    """
    from fanpy.tools import slater

    n_spatial_orbitals = wfn.nspatial
    slater_determinants = wfn.sds
    ci_params = wfn.params

    # Create list of strings for alpha and beta spin indices of occupied MOs
    sds_occupancies = [
        [
            " ".join(map(str, slater.occ_indices(slater.split_spin(sd, n_spatial_orbitals)[0]))),
            " ".join(map(str, slater.occ_indices(slater.split_spin(sd, n_spatial_orbitals)[1]))),
            param,
        ]
        for sd, param in zip(slater_determinants, ci_params)
    ]

    # Sort by the absolute value of CI parameters, descending
    sds_occupancies.sort(key=lambda x: abs(x[2]), reverse=True)

    # Apply threshold filtering (if specified)
    if threshold is not None:
        sds_occupancies = [sd_ind for sd_ind in sds_occupancies if abs(sd_ind[-1]) > threshold]

    # Limit output to max_print (if specified)
    if max_print is not None:
        sds_occupancies = sds_occupancies[:max_print]

    # Determine maximum lengths for formatting
    alpha_len = max(5, max(len(sd_ind[0]) for sd_ind in sds_occupancies))
    beta_len = max(5, max(len(sd_ind[1]) for sd_ind in sds_occupancies))

    # Print header
    print("\n> Slater determinants represented by occupied MO indices\n")
    print(f"{'Alpha':<{alpha_len}}  |  {'Beta':<{beta_len}}  |  CI Parameter")
    print("-" * (alpha_len + beta_len + 23))

    # Print formatted indices
    for alpha, beta, param in sds_occupancies:
        print(f"{alpha:<{alpha_len}}  |  {beta:<{beta_len}}  |  {param:12.8f}")
    print("-" * (alpha_len + beta_len + 23) + "\n")
