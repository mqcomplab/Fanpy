r"""Collection of functions to print Slater determinants and related features.

Functions
---------
print_excitation_operators_as_determinants(wavefunction, max_print, threshold) : {BaseCC, int, float}
    Print Coupled-Cluster Amplitudes from a wavefunction as lists of analogue Slater determinants, using occupied (1) and unoccupied (0) MOs.
print_excitation_operators_indices(wavefunction, max_print, threshold) : {BaseCC, int, float}
    Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied MO indices.
"""


def print_excitation_operators_as_determinants(wfn, max_print=None, threshold=1e-8):
    """
    Print Coupled-Cluster Amplitudes from a wavefunction as lists of  of analogue Slater determinants.
    Occupied and unoccupied MOs are denote by 1 and 0, respectively.
    Creation and annihilation operators are represented by (+) and (-).

    Parameters
    ----------
    wfn : BaseCC
        Wavefunction object containing T operators and CC parameters.
    max_print : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.
    """
    n_spatial = wfn.nspatial
    n_spin = wfn.nspin
    excitation_ops = wfn.exops.keys()
    cc_params = wfn.params

    # Create a list of binary values (0 or 1) representing the reference wavefunction, reversed
    ref_wavefunction_str = list(format(wfn.refwfn, f"0{n_spin}b")[::-1])

    # Prepare the list to store formatted excitation operators
    excitation_ops_str = []

    # Modify the reference wavefunction string for each excitation operator
    for ex_op, param in zip(excitation_ops, cc_params):
        temp_wavefunction = ref_wavefunction_str.copy()  # Copy reference wavefunction
        for index in ex_op:
            # Change to '-' if occupied, '+' if unoccupied
            if temp_wavefunction[index] == "1":
                temp_wavefunction[index] = "-"  # Annihilation (occupied orbital)
            elif temp_wavefunction[index] == "0":
                temp_wavefunction[index] = "+"  # Creation (unoccupied orbital)

        # Split the result into alpha and beta components
        alpha_str = "".join(temp_wavefunction[:n_spatial])
        beta_str = "".join(temp_wavefunction[n_spatial:])

        # Add the formatted excitation operator and parameter to the list
        excitation_ops_str.append([alpha_str, beta_str, param])

    # Sort by the absolute value of the parameter, in descending order
    excitation_ops_str.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Apply threshold filtering (only keep parameters larger than the threshold)
    if threshold is not None:
        excitation_ops_str = [ex_op_str for ex_op_str in excitation_ops_str if abs(ex_op_str[-1]) > threshold]

    # Limit the output to the top max_print results (if specified)
    if max_print is not None:
        excitation_ops_str = excitation_ops_str[:max_print]

    # Print header
    print("\n> Coupled-Cluster Amplitudes represented by MO indices of excited Slater Determinants\n")
    print(f"{'Alpha':<{n_spatial}}  |  {'Beta':<{n_spatial}}  |  CC Parameter")
    print("-" * (n_spatial * 2 + 23))

    # Print the formatted results
    for alpha, beta, param in excitation_ops_str:
        print(f"{alpha:<{n_spatial}}  |  {beta:<{n_spatial}}  |  {param:12.8f}")

    # Print a footer separator
    print("-" * (n_spatial * 2 + 23) + "\n")


def print_excitation_operators_indices(wfn, max_print=None, threshold=1e-8):
    """Print Coupled-Cluster Amplitudes from a wavefunction as lists of occupied MO indices.

    Parameters
    ----------
    wfn : BaseCC
        Wavefunction object containing T operators and CC parameters.
    max_print : int, optional
        Number of determinants to print (if specified).
    threshold : float, optional
        Only print determinants with |param| greater than this value.

    """
    from fanpy.tools import slater
    import numpy as np

    n_spatial = wfn.nspatial
    excitation_ops = wfn.exops.keys()
    cc_params = wfn.params

    # Occupied reference indices (for both alpha and beta spins)
    occupied_ref_indices = np.concatenate(
        (
            slater.occ_indices(slater.split_spin(wfn.refwfn, n_spatial)[0]),  # Alpha
            slater.occ_indices(slater.split_spin(wfn.refwfn, n_spatial)[1]) + n_spatial,  # Beta (shifted by n_spatial)
        )
    )

    # Create list of strings for alpha and beta spin indices (annihilation and creation operators)
    excitation_ops_str = []
    for ex_op, param in zip(excitation_ops, cc_params):
        alpha_occupied = [str(x) for x in ex_op if x < n_spatial and x in occupied_ref_indices]
        beta_occupied = [str(x - n_spatial) for x in ex_op if x >= n_spatial and x in occupied_ref_indices]
        alpha_virtual = [str(x) for x in ex_op if x < n_spatial and x not in occupied_ref_indices]
        beta_virtual = [str(x - n_spatial) for x in ex_op if x >= n_spatial and x not in occupied_ref_indices]
        excitation_ops_str.append([alpha_occupied, beta_occupied, alpha_virtual, beta_virtual, param])

    # Sort by the absolute value of param (in descending order)
    excitation_ops_str.sort(key=lambda x: abs(x[-1]), reverse=True)

    # Apply threshold filtering (if specified)
    if threshold is not None:
        excitation_ops_str = [ex_op_str for ex_op_str in excitation_ops_str if abs(ex_op_str[-1]) > threshold]

    # Limit the number of printed results based on max_print (if specified)
    if max_print is not None:
        excitation_ops_str = excitation_ops_str[:max_print]

    # Determine maximum lengths for formatting
    alpha_occ_len = max(5, max(len(" ".join(ex_op_str[0])) for ex_op_str in excitation_ops_str))
    beta_occ_len = max(5, max(len(" ".join(ex_op_str[1])) for ex_op_str in excitation_ops_str))
    alpha_vir_len = max(5, max(len(" ".join(ex_op_str[2])) for ex_op_str in excitation_ops_str))
    beta_vir_len = max(5, max(len(" ".join(ex_op_str[3])) for ex_op_str in excitation_ops_str))

    occ_len = max(15, alpha_occ_len + beta_occ_len)
    vir_len = max(15, alpha_vir_len + beta_vir_len)

    # Print header
    print("\n> Coupled-Cluster Amplitudes represented by MO indices\n")
    print(f"{'Annihilation':<{vir_len}}  |  {'Creation':<{occ_len}}  |")
    print(
        f"{'Alpha':<{alpha_occ_len}}  |  {'Beta':<{beta_occ_len}}  |  {'Alpha':<{alpha_vir_len}}  |  {'Beta':<{beta_vir_len}}  |  CC Parameter"
    )
    print("-" * (vir_len + occ_len + 23))

    # Print the formatted results with aligned formatting
    for alpha_occ, beta_occ, alpha_vir, beta_vir, param in excitation_ops_str:
        print(
            f"{' '.join(alpha_occ):<{alpha_occ_len}}  |  {' '.join(beta_occ):<{beta_occ_len}}  |  "
            f"{' '.join(alpha_vir):<{alpha_vir_len}}  |  {' '.join(beta_vir):<{beta_vir_len}}  |  {param:12.8f}"
        )
    print("-" * (occ_len + vir_len + 23) + "\n")
