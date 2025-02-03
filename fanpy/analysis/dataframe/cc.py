r"""Collection of functions to process Slater determinants data and related features.

Functions
---------
create_dataframe_from(wavefunction, label=None) : {BaseCC, str}
    Create a DataFrame containing Slater determinants and CC parameters from a given wavefunction.
add_wfn_to_dataframe(df, wfn, label=None): {DataFrame, BaseCC, str}
    Add column to dataframe containing Slater determinants and CC parameters.
concatenate_dataframes(*dfs): {DataFrames}
    Concatenate multiple CI dataframes in a single dataframe.
"""


def create_dataframe_from(wfn, label=None):
    """Create a dataframe containing Slater determinants and CC parameters based on a reference wavefunction.

    Parameters
    ----------
    wfn : BaseCC
        Wavefunction object containing Slater determinants and CC coefficients.
    label : str, optional
        Column label for CC parameters in the DataFrame. If None, defaults to the wavefunction class name.

    Returns
        -------
        pd.DataFrame
            DataFrame containing the Slater determinants and associated CC parameters.
    """
    import pandas as pd

    # Extract Slater determinants and CC parameters from wavefunction
    excitation_ops = wfn.exops.keys()
    params = wfn.params

    # Set default label if not provided
    label = label or wfn.__class__.__name__

    # Convert operators from tuple to strings
    excitation_ops_strs = []
    for item in excitation_ops:
        excitation_ops_strs.append(str(item))

    # Construct DataFrame
    df = pd.DataFrame({label: params}, index=excitation_ops_strs)

    return df


def add_wfn_to_dataframe(df, wfn, label=None):
    """Add column to dataframe containing Slater determinants and CC parameters.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing previously extracted BaseCC data.
    wfn : BaseCC
        Wavefunction object containing Slater determinants and CC coefficients.
    label : str, optional
        Column label for CC parameters in the DataFrame. If None, defaults to the wavefunction class name.
    """
    import pandas as pd

    # Extract Slater determinants and CC parameters from wavefunction
    sds = wfn.sds
    params = wfn.params

    # Set default label if not provided
    label = label or wfn.__class__.__name__

    # Check if the label already exists in the DataFrame and warn the user
    if label in df.columns:
        print(f"Column '{label}' already exists in the DataFrame. It will be overwritten.")

    # Create a Series mapping Slater determinants to their parameters
    param_series = pd.Series(params, index=sds)

    # Map parameters to existing Slater determinants in the DataFrame
    df[label] = param_series.reindex(df.index)


def concatenate_dataframes(*dfs):
    """Concatenate multiple CC dataframes in a single dataframe.

    Parameters
    ----------
    dfs: DataFrame
        Multiple DataFrames containg previously extracted BaseCC data.
    """
    import pandas as pd

    df = pd.concat(dfs, axis=1)

    return df


def convert_operators_to_determinants(df, wfn):
    """Convert DataFrame indices from cluster operators to excited Slater determinants notation.

    Parameters
    ----------
    df: DataFrame
        DataFrame containg previously extracted BaseCC data.
    wfn : BaseCC
        Wavefunction object containing T operators and CC parameters.
    """
    import ast

    # Extract cluster operators from the DataFrame
    excitation_ops = df.index

    # Extract reference wavefunction data from Wavefunction object
    n_spin = wfn.nspin

    # Create a list of binary values (0 or 1) representing the reference wavefunction, reversed
    ref_wavefunction_str = list(format(wfn.refwfn, f"0{n_spin}b")[::-1])

    # Prepare the list to store formatted excitation operators
    sds_indices = []

    # Modify the reference wavefunction string for each excitation operator
    for excitation_op_str in excitation_ops:
        excitation_wavefunction = ref_wavefunction_str[:]  # Copy reference wavefunction
        excitation_op = ast.literal_eval(excitation_op_str)  # Convert index from string back to tuple

        for index in excitation_op:
            # Change to '1' if occupied, '0' if unoccupied
            if excitation_wavefunction[index] == "1":
                excitation_wavefunction[index] = "0"  # Annihilation (occupied orbital)
            elif excitation_wavefunction[index] == "0":
                excitation_wavefunction[index] = "1"  # Creation (unoccupied orbital)

        # Add the formatted excitation operator and parameter to the list
        excitation_wavefunction = int("".join(excitation_wavefunction[::-1]), 2)
        sds_indices.append(excitation_wavefunction)

    # Update index to Slater determinants notation in-place
    df.index = sds_indices
