import abc
import os
import pandas as pd
import json


class DataFrameFanpy:
    def __init__(self, wfn_label=None, wfn_params=None, wfn_index=None):
        """
        Initialize DataFrameFanpy with a labeled DataFrame based on wavefunction parameters.
        """
        if wfn_label is None or wfn_params is None or wfn_index is None:
            self.dataframe = pd.DataFrame()
        else:
            self.dataframe = pd.DataFrame({wfn_label: wfn_params}, index=wfn_index)

        self._metadata = None

    def __getattr__(self, attr):
        """Delegate attribute access to the underlying DataFrame."""
        if attr in ["loc", "iloc"]:  # Ensure accessors work
            return getattr(self.dataframe, attr)

        attr_value = getattr(self.dataframe, attr)
        if callable(attr_value):

            def wrapper(*args, **kwargs):
                result = attr_value(*args, **kwargs)
                return DataFrameFanpy.from_existing(result) if isinstance(result, pd.DataFrame) else result

            return wrapper
        return attr_value

    def __getitem__(self, key):
        """Allow indexing like a normal DataFrame."""
        result = self.dataframe[key]
        return DataFrameFanpy.from_existing(result) if isinstance(result, pd.DataFrame) else result

    def __setitem__(self, key, value):
        """Allow item assignment like a normal DataFrame."""
        self.dataframe[key] = value

    def __repr__(self):
        """Ensure the wrapper prints like a normal DataFrame."""
        return self.dataframe.__repr__()

    @classmethod
    def from_existing(cls, existing_df):
        """Ensure it only wraps a plain DataFrame, avoiding nested wrappers."""
        if isinstance(existing_df, DataFrameFanpy):
            return existing_df  # Return directly if it's already a wrapper
        instance = cls.__new__(cls)
        instance.dataframe = existing_df
        return instance

    def update_dataframe(self, new_df):
        """Safely update the internal DataFrame."""
        self.dataframe = new_df if isinstance(new_df, pd.DataFrame) else new_df.dataframe

    def to_csv(self, filename):
        """Import dataframe as a CSV file, including metadata as JSON file."""

        # Filename formatting
        name, ext = os.path.splitext(filename)
        if ext.lower() != ".csv":
            csv_filename = os.path.join(name + ".csv")
            metadata_filename = os.path.join(name + ".json")
        else:
            csv_filename = filename
            metadata_filename = os.path.join(name + ".json")

        # Save DataFrame
        self.dataframe.to_csv(csv_filename, index=True)

        # Save metadata
        with open(metadata_filename, "w") as f:
            json.dump(self._metadata, f)

    def read_csv(self, filename, **kwargs):
        """Import dataframe from a CSV file and a metadata JSON files."""

        # Filename formatting
        name, ext = os.path.splitext(filename)
        if ext.lower() != ".csv":
            csv_filename = os.path.join(name + ".csv")
            metadata_filename = os.path.join(name + ".json")
        else:
            csv_filename = filename
            metadata_filename = os.path.join(name + ".json")

        # Import DataFrame from CSV file
        self.dataframe = pd.read_csv(csv_filename, index_col=0, **kwargs)

        # Import metadata from JSON file
        with open(metadata_filename) as f:
            self._metadata = json.load(f)

    @abc.abstractmethod
    def add_wfn_to_dataframe(self, wfn, wfn_label=None):
        """Add column to dataframe containing Slater determinants and CI parameters.

        Parameters
        ----------
        df: DataFrame
            DataFrame containg previously extracted CIWavefunction data.
        wfn : BaseWavefunction
            Wavefunction object containing descriptors and coefficients.
        wfn_label : str, optional
            Column label for parameters in the DataFrame. If None, defaults to the wavefunction class name.
        """
