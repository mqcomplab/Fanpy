from fanpy.analysis.dataframe.base import DataFrameFanpy
from fanpy.wfn.ci.base import CIWavefunction
from pandas import Series


class DataFrameCI(DataFrameFanpy):
    def __init__(self, wfn, wfn_label=None):

        # Check if the given wavefunction object is valid
        if not isinstance(wfn, CIWavefunction):
            raise TypeError("Given wavefunction is not an instance of BaseWavefunction (or its child).")

        # Extract Slater determinants and CI parameters from wavefunction
        wfn_index = wfn.sds
        wfn_params = wfn.params

        # Store Slater determinants and wavefunction data as attributes
        self.wfn_nspatial = wfn.nspatial
        self._index_view = "determinants"

        # Set default label if not provided
        wfn_label = wfn_label or wfn.__class__.__name__

        super().__init__(wfn_label, wfn_params, wfn_index)

    @property
    def index_view(self):
        """Return a flag cointaing the format of the DataFrame index."""

        return self._index_view

    def add_wfn_to_dataframe(self, wfn, wfn_label=None):
        """Add column to dataframe containing Slater determinants and CI parameters.

        Parameters
        ----------
        wfn : CIWavefunction
            Wavefunction object containing Slater determinants and CI coefficients.
        wfn_label : str, optional
            Column label for CI parameters in the DataFrame. If None, defaults to the wavefunction class name.
        """

        # Extract Slater determinants and CI parameters from wavefunction
        wfn_index = wfn.sds
        wfn_params = wfn.params

        # Set default label if not provided
        wfn_label = wfn_label or wfn.__class__.__name__

        # Check if the label already exists in the DataFrame and warn the user
        if wfn_label in self.columns:
            print(f"Column '{wfn_label}' already exists in the DataFrame. It will be overwritten.")

        # Create a Series mapping Slater determinants to their parameters
        param_series = Series(wfn_params, index=wfn_index)

        # Expand the index first
        self.update_dataframe(self.reindex(self.index.union(param_series.index)))

        # Add the new column while ensuring alignment
        self.dataframe[wfn_label] = param_series.reindex(self.index)

    def set_formatted_sds_as_index(self):
        """Convert DataFrame index to the human-readable format of occupied (1) and unoccupied (0) MOs."""

        from fanpy.tools import slater

        if self.index_view == "determinants":
            sds_index = self.index

            formatted_sds = [
                " ".join(
                    [
                        format(slater.split_spin(sd, self.wfn_nspatial)[0], f"0{self.wfn_nspatial}b")[::-1],
                        format(slater.split_spin(sd, self.wfn_nspatial)[1], f"0{self.wfn_nspatial}b")[::-1],
                    ]
                )
                for sd in sds_index
            ]

            # Update index notation of the DataFrame
            self.dataframe.index = formatted_sds

            # Update index_view flag
            self._index_view = "formatted determinants"

    def set_sds_as_index(self):
        """Convert DataFrame index to the default format of binary numbers which represents SDs in Fanpy convention."""

        if self.index_view == "formatted determinants":
            formatted_sds = self.index

            sds = [
                int(sd_beta[::-1] + sd_alpha[::-1], 2)
                for sd_alpha, sd_beta in (formatted_sd.split() for formatted_sd in formatted_sds)
            ]

            # Update index notation of the DataFrame
            self.dataframe.index = sds

            # Update index_view flag
            self._index_view = "determinants"
