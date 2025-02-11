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
    n_spatial_orbitals = wfn.nspatial
    n_geminals = wfn.ngem
    n_spin_orbitals = wfn.nspin
    orbital_pairs_indices = list(wfn.dict_orbpair_ind.keys())
    geminal_params = wfn.params

    # Create binary representation of the reference wavefunction, reversed
    if hasattr(wfn, "ref_sd"):
        reference_wfn = list(format(wfn.ref_sd, f"0{n_spin_orbitals}b")[::-1])
    else:
        reference_wfn = [
            0,
        ] * n_spin_orbitals

    # Create representation of the Geminal wavefunction
    if hasattr(wfn, "dict_reforbpair_ind"):
        geminals_indices = list(wfn.dict_reforbpair_ind.keys())

        # Prepare formatted orbital pairs
        geminal_strings = []

        # Generate geminal operator strings
        for occ_indices, params_for_occ in zip(geminals_indices, geminal_params):
            modified_wfn = reference_wfn.copy()

            # Mark annihilation (occupied) positions with '-'
            for idx in occ_indices:
                modified_wfn[idx] = "-"

            for orb_pair, parameter in zip(orbital_pairs_indices, params_for_occ):
                final_wfn = modified_wfn.copy()

                # Mark creation (unoccupied) positions with '+'
                for idx in orb_pair:
                    final_wfn[idx] = "+"

                # Split the string into alpha and beta components
                alpha_str = "".join(final_wfn[:n_spatial_orbitals])
                beta_str = "".join(final_wfn[n_spatial_orbitals:])

                geminal_strings.append([alpha_str, beta_str, parameter])

    else:
        geminals_indices = [[i] for i in range(n_geminals)]

        # Prepare formatted orbital pairs
        geminal_strings = []

        # Generate geminal operator strings
        for occ_indices, params_for_occ in zip(geminals_indices, geminal_params):
            for orb_pair, parameter in zip(orbital_pairs_indices, params_for_occ):
                final_wfn = reference_wfn.copy()

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
    n_spatial_orbitals = wfn.nspatial
    n_geminals = wfn.ngem
    orbital_pairs_indices = list(wfn.dict_orbpair_ind.keys())
    geminal_params = wfn.params

    # Create representation of the Geminal wavefunction
    if hasattr(wfn, "dict_reforbpair_ind"):
        geminals_indices = list(wfn.dict_reforbpair_ind.keys())

        # Prepare formatted geminal strings
        geminal_strings = []
        for geminal_indices, params_for_occ in zip(geminals_indices, geminal_params):
            for orb_pair, parameter in zip(orbital_pairs_indices, params_for_occ):
                alpha_gem_indices = [str(x) for x in geminal_indices if x < n_spatial_orbitals and x in geminal_indices]
                beta_gem_indices = [
                    str(x - n_spatial_orbitals)
                    for x in geminal_indices
                    if x >= n_spatial_orbitals and x in geminal_indices
                ]

                alpha_pair_indices = [str(x) for x in orb_pair if x < n_spatial_orbitals]
                beta_pair_indices = [str(x - n_spatial_orbitals) for x in orb_pair if x >= n_spatial_orbitals]

                geminal_strings.append(
                    [alpha_gem_indices, beta_gem_indices, alpha_pair_indices, beta_pair_indices, parameter]
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
        alpha_gem_len = max(5, max(len(" ".join(gem_str[0])) for gem_str in geminal_strings))
        beta_gem_len = max(5, max(len(" ".join(gem_str[1])) for gem_str in geminal_strings))
        alpha_occ_len = max(5, max(len(" ".join(gem_str[2])) for gem_str in geminal_strings))
        beta_occ_len = max(5, max(len(" ".join(gem_str[3])) for gem_str in geminal_strings))
        gem_len = max(15, alpha_gem_len + beta_gem_len)
        occ_len = max(15, alpha_occ_len + beta_occ_len)

        # Print header
        print("\n> Geminal Coefficients represented by MO indices\n")
        print(f"{'Geminal':<{gem_len}}  |  {'Pair':<{occ_len}}  |")
        print(
            f"{'Alpha':<{alpha_gem_len}}  |  {'Beta':<{beta_gem_len}}  |  {'Alpha':<{alpha_occ_len}}  |  {'Beta':<{beta_occ_len}}  |  Gem Parameter"
        )
        print("-" * (gem_len + occ_len + 24))

        # Print results with aligned formatting
        for alpha_gem, beta_gem, alpha_occ, beta_occ, param in geminal_strings:
            print(
                f"{' '.join(alpha_gem):<{alpha_gem_len}}  |  {' '.join(beta_gem):<{beta_gem_len}}  |  "
                f"{' '.join(alpha_occ):<{alpha_occ_len}}  |  {' '.join(beta_occ):<{beta_occ_len}}  |  {param:12.8f}"
            )
        print("-" * (gem_len + occ_len + 24) + "\n")

    else:
        geminals_indices = [[i] for i in range(n_geminals)]

        # Prepare formatted geminal strings
        geminal_strings = []
        for geminal_indices, params_for_occ in zip(geminals_indices, geminal_params):
            for orb_pair, parameter in zip(orbital_pairs_indices, params_for_occ):
                gem_indices = [str(x) for x in geminal_indices if x < n_spatial_orbitals and x in geminal_indices]

                alpha_pair_indices = [str(x) for x in orb_pair if x < n_spatial_orbitals]
                beta_pair_indices = [str(x - n_spatial_orbitals) for x in orb_pair if x >= n_spatial_orbitals]

                geminal_strings.append([gem_indices, alpha_pair_indices, beta_pair_indices, parameter])

        # Sort by the absolute value of parameter, descending
        geminal_strings.sort(key=lambda x: abs(x[-1]), reverse=True)

        # Apply threshold filtering
        if threshold is not None:
            geminal_strings = [gem_str for gem_str in geminal_strings if abs(gem_str[-1]) > threshold]

        # Apply max_print limit
        if max_print is not None:
            geminal_strings = geminal_strings[:max_print]

        # Determine maximum lengths for formatting
        gem_len = max(8, max(len(" ".join(gem_str[0])) for gem_str in geminal_strings))
        alpha_occ_len = max(5, max(len(" ".join(gem_str[1])) for gem_str in geminal_strings))
        beta_occ_len = max(5, max(len(" ".join(gem_str[2])) for gem_str in geminal_strings))
        gem_len = max(8, gem_len)
        occ_len = max(15, alpha_occ_len + beta_occ_len)

        # Print header
        print("\n> Geminal Coefficients represented by MO indices\n")
        print(f"{'Geminal':<{gem_len}}  |  {'Pair':<{occ_len}}  |")
        print(f"{' ':<{gem_len}}  |  {'Alpha':<{alpha_occ_len}}  |  {'Beta':<{beta_occ_len}}  |  Gem Parameter")
        print("-" * (gem_len + occ_len + 24))

        # Print results with aligned formatting
        for gem, alpha_occ, beta_occ, param in geminal_strings:
            print(
                f"{' '.join(gem):<{gem_len}}  |  "
                f"{' '.join(alpha_occ):<{alpha_occ_len}}  |  {' '.join(beta_occ):<{beta_occ_len}}  |  {param:12.8f}"
            )
        print("-" * (gem_len + occ_len + 24) + "\n")
