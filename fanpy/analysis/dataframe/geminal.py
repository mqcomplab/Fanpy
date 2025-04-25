from fanpy.analysis.dataframe.base import DataFrameFanpy
from fanpy.wfn.geminal.base import BaseGeminal
from pandas import Series, MultiIndex


class DataFrameGeminal(DataFrameFanpy):
    def __init__(self, wfn=None, wfn_label=None):

        if wfn is None:
            # Set default attributes
            self.wfn_nspatial = None
            self.wfn_nspin = None
            self._index_view = "geminals"

            super().__init__()

        else:
            # Check if the given wavefunction object is valid
            if not isinstance(wfn, BaseGeminal):
                raise TypeError("Given wavefunction is not an instance of BaseWavefunction (or its child).")

            import numpy as np

            # Extract excitation operators and Geminal parameters from wavefunction
            wfn_orbital_pairs = list(wfn.dict_orbpair_ind.keys())
            wfn_params = np.ravel(wfn.params)
            wfn_ngem = wfn.ngem

            # Store excitation operators and wavefunction data as attributes
            self.wfn_nspatial = wfn.nspatial
            self.wfn_nspin = wfn.nspin
            self._index_view = "geminals"

            if hasattr(wfn, "ref_sd"):
                self.wfn_ref_sd = wfn.ref_sd
            else:
                self.wfn_ref_sd = None

            # Convert operators from tuple to strings
            wfn_pairs_str = []
            for wfn_orbital_pair in wfn_orbital_pairs:
                wfn_pairs_str.append(" ".join(map(str, wfn_orbital_pair)))

            # Create representation of the Geminal wavefunction
            if hasattr(wfn, "dict_reforbpair_ind"):
                wfn_geminals = list(wfn.dict_reforbpair_ind.keys())

                wfn_geminals_str = []
                for wfn_geminal in wfn_geminals:
                    wfn_geminals_str.append(" ".join(map(str, wfn_geminal)))

            else:
                wfn_geminals_str = [i for i in range(wfn_ngem)]

            wfn_index = MultiIndex.from_product([wfn_geminals_str, wfn_pairs_str], names=["geminal", "pair"])

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
            "wfn_nspin": self.wfn_nspin,
            "wfn_ref_sd": self.wfn_ref_sd,
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
        self.wfn_nspin = self.metadata["wfn_nspin"]
        self.wfn_ref_sd = self.metadata["wfn_ref_sd"]
        self._index_view = self.metadata["index_view"]

        self.dataframe = self.dataframe.reset_index()
        if self.metadata["index_view"] == "geminals":
            self.dataframe = self.dataframe.set_index(["geminal", "pair"])
        elif self.metadata["index_view"] == "determinants":
            self.dataframe = self.dataframe.set_index(["geminal", "determinants"])
        elif self.metadata["index_view"] == "formatted determinants":
            self.dataframe = self.dataframe.set_index(["geminal", "formatted determinants"])

    def add_wfn_to_dataframe(self, wfn, wfn_label=None):
        """Add column to dataframe containing excitation operators and Geminal parameters.

        Parameters
        ----------
        wfn : BaseGeminal
            Wavefunction object containing excitation operators and Geminal parameters.
        wfn_label : str, optional
            Column label for Geminal parameters in the DataFrame. If None, defaults to the wavefunction class name.
        """

        # Extract excitation operators and Geminal parameters from wavefunction
        wfn_orbital_pairs = list(wfn.dict_orbpair_ind.keys())
        wfn_params = wfn.params
        wfn_ngem = wfn.ngem

        if hasattr(wfn, "ref_sd") and (wfn.ref_sd == None):
            self.wfn_ref_sd = wfn.ref_sd

        # Convert operators from tuple to strings
        wfn_pairs_str = []
        for wfn_orbital_pair in wfn_orbital_pairs:
            wfn_pairs_str.append(" ".join(map(str, wfn_orbital_pair)))

        # Create representation of the Geminal wavefunction
        if hasattr(wfn, "dict_reforbpair_ind"):
            wfn_geminals = list(wfn.dict_reforbpair_ind.keys())

            wfn_geminals_str = []
            for wfn_geminal in wfn_geminals:
                wfn_geminals_str.append(" ".join(map(str, wfn_geminal)))

        else:
            wfn_geminals_str = [i for i in range(wfn_ngem)]

        wfn_index = MultiIndex.from_product([wfn_geminals_str, wfn_pairs_str], names=["geminal", "pair"])

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
        """Convert DataFrame index to the default format of binary numbers which represents pairs in Fanpy convention."""

        if self.index_view == "geminals":
            wfn_geminals = self.index.get_level_values("geminal")
            wfn_orbital_pairs = self.index.get_level_values("pair")

            # Prepare the list to store Slater Determinants
            sds = []

            for geminal, orbital_pair in zip(wfn_geminals, wfn_orbital_pairs):
                if self.wfn_ref_sd:
                    wfn_pair = list(format(self.wfn_ref_sd, f"0{2*self.wfn_nspatial}b")[::-1])

                    for orbital in list(map(int, geminal.split())):
                        wfn_pair[orbital] = "0"

                else:
                    wfn_pair = [
                        "0",
                    ] * self.wfn_nspin

                for orbital in list(map(int, orbital_pair.split())):
                    wfn_pair[orbital] = "1"

                # Add Slater Determinant and parameter to the list
                wfn_pair = int("".join(wfn_pair[::-1]), 2)

                sds.append(wfn_pair)

            # Update index notation of the DataFrame
            self.dataframe.index = MultiIndex.from_arrays([wfn_geminals, sds], names=["geminal", "determinants"])

            # Update index_view flag
            self._index_view = "determinants"

        elif self.index_view == "formatted determinants":
            wfn_geminals = self.index.get_level_values("geminal")
            wfn_formatted_sds = self.index.get_level_values("formatted determinants")

            # Prepare the list to store Slater Determinants
            sds = []

            for formatted_sd_pair in wfn_formatted_sds:

                sd_alpha, sd_beta = formatted_sd_pair.split()
                sd_pair = int(sd_beta[::-1] + sd_alpha[::-1], 2)

                sds.append(sd_pair)

            # Update index notation of the DataFrame
            self.dataframe.index = MultiIndex.from_arrays([wfn_geminals, sds], names=["geminal", "determinants"])

            # Update index_view flag
            self._index_view = "determinants"

    def set_formatted_sds_as_index(self):
        """Convert DataFrame index to the human-readable notation of excited Slater Determinants of occupied (1) and unoccupied (0) MOs."""

        from fanpy.tools import slater

        if self.index_view == "geminals":
            self.set_sds_as_index()

        if self.index_view == "determinants":
            wfn_geminals = self.index.get_level_values("geminal")
            wfn_sds = self.index.get_level_values("determinants")

            # Prepare the list to store formatted Slater Determinants
            formatted_sds = []

            for sd_pair in wfn_sds:

                formatted_sd = " ".join(
                    [
                        format(slater.split_spin(sd_pair, self.wfn_nspatial)[0], f"0{self.wfn_nspatial}b")[::-1],
                        format(slater.split_spin(sd_pair, self.wfn_nspatial)[1], f"0{self.wfn_nspatial}b")[::-1],
                    ]
                )

                formatted_sds.append(formatted_sd)

            # Update index notation of the DataFrame
            self.dataframe.index = MultiIndex.from_arrays(
                [wfn_geminals, formatted_sds], names=["geminal", "formatted determinants"]
            )

            # Update index_view flag
            self._index_view = "formatted determinants"

    def set_geminals_as_index(self):
        """Convert DataFrame index to geminals operator indices."""

        from fanpy.tools import slater

        if self.index_view == "formatted determinants":
            self.set_sds_as_index()

        if self.index_view == "determinants":
            wfn_geminals = self.index.get_level_values("geminal")
            wfn_sds_pairs = self.index.get_level_values("pair")

            # Prepare the list to store formatted Slater Determinants
            operators = []

            for geminal, sd_pair in zip(wfn_geminals, wfn_sds_pairs):

                if self.wfn_ref_sd:
                    wfn_pair = list(format(self.wfn_ref_sd, f"0{2*self.wfn_nspatial}b")[::-1])

                    for orbital in list(map(int, geminal.split())):
                        wfn_pair[orbital] = "0"

                    wfn_pair = int("".join(wfn_pair[::-1]), 2)

                    operator = " ".join(map(str, slater.occ_indices(wfn_pair ^ sd_pair)))

                else:
                    operator = " ".join(map(str, slater.occ_indices(sd_pair)))

                operators.append(operator)

            # Update index notation of the DataFrame
            self.dataframe.index = MultiIndex.from_arrays([wfn_geminals, operators], names=["geminal", "pair"])

            # Update index_view flag
            self._index_view = "geminals"
