from fanpy.analysis.dataframe.base import DataFrameFanpy
from fanpy.wfn.cc.base import BaseCC
from pandas import Series


class DataFrameCC(DataFrameFanpy):
    def __init__(self, wfn=None, wfn_label=None):

        if wfn is None:
            # Set default attributes
            self.wfn_nspatial = None
            self.wfn_reference = None
            self._index_view = "operators"

            super().__init__()

        else:
            # Check if the given wavefunction object is valid
            if not isinstance(wfn, BaseCC):
                raise TypeError("Given wavefunction is not an instance of BaseWavefunction (or its child).")

            # Extract excitation operators and CC parameters from wavefunction
            wfn_excitation_ops = wfn.exops.keys()
            wfn_params = wfn.params

            # Convert operators from tuple to strings
            wfn_index = []
            for excitation_op in wfn_excitation_ops:
                wfn_index.append(" ".join(map(str, excitation_op)))

            # Store excitation operators and wavefunction data as attributes
            self.wfn_nspatial = wfn.nspatial
            self.wfn_reference = wfn.refwfn
            self._index_view = "operators"

            # Set default label if not provided
            wfn_label = wfn_label or wfn.__class__.__name__

            super().__init__(wfn_label, wfn_params, wfn_index)

    @property
    def index_view(self):
        """Return a flag cointaing the format of the DataFrame index."""

        return self._index_view

    def to_csv(self, filename):
        """Import dataframe as a CSV file, including metadata as JSON file."""

        # Prepare metadata
        self.metadata = {
            "wfn_nspatial": self.wfn_nspatial,
            "wfn_reference": self.wfn_reference,
            "index_view": self.index_view,
        }

        # Call to_csv function from parent class
        DataFrameFanpy.to_csv(self, filename)

    def read_csv(self, filename, **kwargs):
        """Import dataframe from a CSV file and a metadata JSON files."""

        # Call read_csv function from parent class
        DataFrameFanpy.read_csv(self, filename, **kwargs)

        # Import wavefunction information from metadata
        self.wfn_nspatial = self.metadata["wfn_nspatial"]
        self.wfn_reference = self.metadata["wfn_reference"]
        self._index_view = self.metadata["index_view"]

    def add_wfn_to_dataframe(self, wfn, wfn_label=None):
        """Add column to dataframe containing excitation operators and CC parameters.

        Parameters
        ----------
        wfn : BaseCC
            Wavefunction object containing excitation operators and CC parameters.
        wfn_label : str, optional
            Column label for CC parameters in the DataFrame. If None, defaults to the wavefunction class name.
        """

        # Extract excitation operators and CC parameters from wavefunction
        wfn_excitation_ops = wfn.exops.keys()
        wfn_params = wfn.params

        # Convert operators from tuple to strings
        wfn_index = []
        for excitation_op in wfn_excitation_ops:
            wfn_index.append(" ".join(map(str, excitation_op)))

        # Set default label if not provided
        wfn_label = wfn_label or wfn.__class__.__name__

        # Check if the label already exists in the DataFrame and warn the user
        if wfn_label in self.columns:
            print(f"Column '{wfn_label}' already exists in the DataFrame. It will be overwritten.")

        # Create a Series mapping excitation operators to their parameters
        param_series = Series(wfn_params, index=wfn_index)

        # Expand the index first
        self.update_dataframe(self.reindex(self.index.union(param_series.index)))

        # Add the new column while ensuring alignment
        self.dataframe[wfn_label] = param_series.reindex(self.index)

    def set_sds_as_index(self):
        """Convert DataFrame index to the default format of binary numbers which represents SDs in Fanpy convention."""

        if self.index_view == "operators":
            operators = self.index

            # Prepare the list to store formatted excitation operators
            sds = []

            for operator in operators:
                excitation_wfn = list(format(self.wfn_reference, f"0{2*self.wfn_nspatial}b")[::-1])
                excitation_op = list(map(int, operator.split()))

                for index in excitation_op:
                    # Change to '1' if occupied, '0' if unoccupied
                    if excitation_wfn[index] == "1":
                        excitation_wfn[index] = "0"  # Annihilation (occupied orbital)
                    elif excitation_wfn[index] == "0":
                        excitation_wfn[index] = "1"  # Creation (unoccupied orbital)

                # Add the formatted excitation operator and parameter to the list
                excitation_wfn = int("".join(excitation_wfn[::-1]), 2)
                sds.append(excitation_wfn)

            # Update index notation of the DataFrame
            self.dataframe.index = sds

            # Update index_view flag
            self._index_view = "determinants"

        elif self.index_view == "formatted determinants":
            formatted_sds = self.index

            sds = [
                int(sd_beta[::-1] + sd_alpha[::-1], 2)
                for sd_alpha, sd_beta in (formatted_sd.split() for formatted_sd in formatted_sds)
            ]

            # Update index notation of the DataFrame
            self.dataframe.index = sds

            # Update index_view flag
            self._index_view = "determinants"

    def set_formatted_sds_as_index(self):
        """Convert DataFrame index to the human-readable notation of excited Slater Determinants of occupied (1) and unoccupied (0) MOs."""

        from fanpy.tools import slater

        if self.index_view == "operators":
            self.set_sds_as_index()

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

    def set_operators_as_index(self):
        """Convert DataFrame index to coupled cluster operator indices."""

        from fanpy.tools import slater

        if self.index_view == "formatted determinants":
            self.set_sds_as_index()

        if self.index_view == "determinants":
            sds_index = self.index

            operators = [" ".join(map(str, slater.occ_indices(self.wfn_reference ^ sd))) for sd in sds_index]

            # Update index notation of the DataFrame
            self.dataframe.index = operators

            # Update index_view flag
            self._index_view = "operators"
