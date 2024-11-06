r"""Collection of functions to process Slater determinants data and related features.

Functions
---------
create_dataframe_from(wavefunction, label=None) : {CIWavefunction, str}
    Create a DataFrame containing Slater determinants and CI parameters from a given wavefunction.
add_wfn_to_dataframe(df, wfn, label=None): {DataFrame, CIWavefunction, str}
    Add column to dataframe containing Slater determinants and CI parameters.
concatenate_dataframes(*dfs): {DataFrames}
    Concatenate multiple CI dataframes in a single dataframe.
"""

import pandas as pd


def create_dataframe_from(wfn, label=None):
    """Create a dataframe containing Slater determinants and CI parameters based on a reference wavefunction.

    Parameters
    ----------
    wfn : CIWavefunction
        Wavefunction object containing Slater determinants and CI coefficients.
    label : str, optional
        Column label for CI parameters in the DataFrame. If None, defaults to the wavefunction class name.

    Returns
        -------
        pd.DataFrame
            DataFrame containing the Slater determinants and associated CI parameters.
    """

    # Extract Slater determinants and CI parameters from wavefunction
    sds = wfn.sds
    params = wfn.params

    # Set default label if not provided
    label = label or wfn.__class__.__name__

    # Construct DataFrame
    df = pd.DataFrame({label: params}, index=sds)

    return df


def add_wfn_to_dataframe(df, wfn, label=None):
    """Add column to dataframe containing Slater determinants and CI parameters.

    Parameters
    ----------
    df: DataFrame
        DataFrame containg previously extracted CIWavefunction data.
    wfn : CIWavefunction
        Wavefunction object containing Slater determinants and CI coefficients.
    label : str, optional
        Column label for CI parameters in the DataFrame. If None, defaults to the wavefunction class name.
    """

    # Extract Slater determinants and CI parameters from wavefunction
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
    df[label] = param_series.reindex(df.index).fillna(False)


def concatenate_dataframes(*dfs):
    """Concatenate multiple CI dataframes in a single dataframe.

    Parameters
    ----------
    dfs: DataFrame
        Multiple DataFrames containg previously extracted CIWavefunction data.
    """

    df = pd.concat(dfs, axis=1).fillna(False)

    return df
