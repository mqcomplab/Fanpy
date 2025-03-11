r"""Collection of functions to print Slater determinants and related features.

Functions
---------
print_geminal_ops(wfn, nprint, threshold) : {BaseGeminal, int, float}
    Print Geminal parameters from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
print_geminal_ops_indices(wfn, nprint, threshold) : {BaseGeminal, int, float}
    Print Geminal parameters from a wavefunction as lists of occupied MO indices.

"""


def print_geminal_ops(wfn, max_print=None, threshold=1e-8):
    """
    Print Geminal parameters from a wavefunction as lists of occupied (1) and unoccupied (0) MOs.
    Creation and annihilation operator indices represented by (+) and (-).

    Parameters
    ----------
    wfn : BaseGeminal
        Wavefunction object containing Geminal operators and parameters.
    max_print : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.
    """
    from fanpy.tools import slater
    import numpy as np

    n_spatial_orbitals = wfn.nspatial
    n_spin_orbitals = wfn.nspin
    orbital_pairs = list(wfn.dict_orbpair_ind.keys())
    geminal_params = wfn.params

    # Create binary representation of the reference wavefunction, reversed
    reference_wfn = list(format(wfn.ref_sd, f"0{n_spin_orbitals}b")[::-1])
    occupied_indices = np.concatenate(
        (
            slater.occ_indices(slater.split_spin(wfn.ref_sd, n_spatial_orbitals)[0]),  # Alpha spin
            slater.occ_indices(slater.split_spin(wfn.ref_sd, n_spatial_orbitals)[1]) + n_spatial_orbitals,  # Beta spin
        )
    )
    paired_occupied_indices = [
        [occupied_indices[i], occupied_indices[i + len(occupied_indices) // 2]]
        for i in range(len(occupied_indices) // 2)
    ]

    # Prepare formatted orbital pairs
    geminal_strings = []

    # Generate geminal operator strings
    for occ_indices, params_for_occ in zip(paired_occupied_indices, geminal_params):
        modified_wfn = reference_wfn.copy()

        # Mark annihilation (occupied) positions with '-'
        for idx in occ_indices:
            modified_wfn[idx] = "-"

        for orb_pair, parameter in zip(orbital_pairs, params_for_occ):
            final_wfn = modified_wfn.copy()

            # Mark creation (unoccupied) positions with '+'
            for idx in orb_pair:
                final_wfn[idx] = "+"

            # Split the string into alpha and beta components
            alpha_str = "".join(final_wfn[:n_spatial_orbitals])
            beta_str = "".join(final_wfn[n_spatial_orbitals:])

            geminal_strings.append([alpha_str, beta_str, parameter])

    # Sort by absolute value of parameter, descending
    geminal_strings.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Apply threshold filtering
    if threshold is not None:
        geminal_strings = [g_str for g_str in geminal_strings if abs(g_str[-1]) > threshold]

    # Limit output to max_print
    if max_print is not None:
        geminal_strings = geminal_strings[:max_print]

    # Print header
    print("\n> Geminal Coefficients represented by MO indices\n")
    print(f"{'Alpha':<{n_spatial_orbitals}}  |  {'Beta':<{n_spatial_orbitals}}  |  Gem Parameter")
    print("-" * (n_spatial_orbitals * 2 + 24))

    # Print formatted results
    for alpha, beta, param in geminal_strings:
        print(f"{alpha:<{n_spatial_orbitals}}  |  {beta:<{n_spatial_orbitals}}  |  {param:12.8f}")

    # Print footer
    print("-" * (n_spatial_orbitals * 2 + 24) + "\n")


def print_geminal_ops_indices(wfn, max_print=None, threshold=1e-8):
    """Print Geminal parameters from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    wfn : BaseGeminal
        Wavefunction object containing Geminal operators and parameters.
    max_print : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.

    """
    from fanpy.tools import slater
    import numpy as np

    n_spatial_orbitals = wfn.nspatial
    orbital_pairs = list(wfn.dict_orbpair_ind.keys())
    geminal_params = wfn.params

    # Occupied reference indices for both alpha and beta spins
    occupied_indices = np.concatenate(
        (
            slater.occ_indices(slater.split_spin(wfn.ref_sd, n_spatial_orbitals)[0]),  # Alpha
            slater.occ_indices(slater.split_spin(wfn.ref_sd, n_spatial_orbitals)[1]) + n_spatial_orbitals,  # Beta
        )
    )
    paired_occupied_indices = [
        [occupied_indices[i], occupied_indices[i + len(occupied_indices) // 2]]
        for i in range(len(occupied_indices) // 2)
    ]

    # Prepare formatted geminal strings
    geminal_strings = []
    for occ_indices, params_for_occ in zip(paired_occupied_indices, geminal_params):
        for orb_pair, parameter in zip(orbital_pairs, params_for_occ):
            alpha_occ_indices = [str(x) for x in occ_indices if x < n_spatial_orbitals and x in occ_indices]
            beta_occ_indices = [
                str(x - n_spatial_orbitals) for x in occ_indices if x >= n_spatial_orbitals and x in occ_indices
            ]
            alpha_vir_indices = [str(x) for x in orb_pair if x < n_spatial_orbitals and x not in occ_indices]
            beta_vir_indices = [
                str(x - n_spatial_orbitals) for x in orb_pair if x >= n_spatial_orbitals and x not in occ_indices
            ]
            geminal_strings.append(
                [alpha_occ_indices, beta_occ_indices, alpha_vir_indices, beta_vir_indices, parameter]
            )

    # Sort by the absolute value of parameter, descending
    geminal_strings.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Apply threshold filtering
    if threshold is not None:
        geminal_strings = [gem_str for gem_str in geminal_strings if abs(gem_str[-1]) > threshold]

    # Apply max_print limit
    if max_print is not None:
        geminal_strings = geminal_strings[:max_print]

    # Determine maximum lengths for formatting
    alpha_occ_len = max(5, max(len(" ".join(gem_str[0])) for gem_str in geminal_strings))
    beta_occ_len = max(5, max(len(" ".join(gem_str[1])) for gem_str in geminal_strings))
    alpha_vir_len = max(5, max(len(" ".join(gem_str[2])) for gem_str in geminal_strings))
    beta_vir_len = max(5, max(len(" ".join(gem_str[3])) for gem_str in geminal_strings))
    occ_len = max(15, alpha_occ_len + beta_occ_len)
    vir_len = max(15, alpha_vir_len + beta_vir_len)

    # Print header
    print("\n> Geminal Coefficients represented by MO indices\n")
    print(f"{'Occupied':<{vir_len}}  |  {'Creation':<{occ_len}}  |")
    print(
        f"{'Alpha':<{alpha_occ_len}}  |  {'Beta':<{beta_occ_len}}  |  {'Alpha':<{alpha_vir_len}}  |  {'Beta':<{beta_vir_len}}  |  Gem Parameter"
    )
    print("-" * (vir_len + occ_len + 24))

    # Print results with aligned formatting
    for alpha_occ, beta_occ, alpha_vir, beta_vir, param in geminal_strings:
        print(
            f"{' '.join(alpha_occ):<{alpha_occ_len}}  |  {' '.join(beta_occ):<{beta_occ_len}}  |  "
            f"{' '.join(alpha_vir):<{alpha_vir_len}}  |  {' '.join(beta_vir):<{beta_vir_len}}  |  {param:12.8f}"
        )
    print("-" * (occ_len + vir_len + 24) + "\n")
