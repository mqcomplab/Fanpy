"""Test fanpy.analysis.dataframe."""

from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.cc.base import BaseCC
from fanpy.wfn.cc.standard_cc import StandardCC
from fanpy.wfn.geminal.base import BaseGeminal
from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.tools import slater, sd_list

from fanpy.analysis.dataframe.base import DataFrameFanpy
from fanpy.analysis.dataframe.ci import DataFrameCI
from fanpy.analysis.dataframe.cc import DataFrameCC
from fanpy.analysis.dataframe.geminal import DataFrameGeminal

import numpy as np
import pandas as pd
from itertools import combinations, product

import pytest


def test_empty_dataframe():
    """Test creating an empty analysis.dataframe.base.DataFrameFanpy object."""

    # Create an empty DataFrameFanpy object
    df = DataFrameFanpy()

    assert isinstance(df.dataframe, pd.DataFrame)
    assert df.empty


def test_base_dataframe():
    """Test creating an analysis.dataframe.base.DataFrameFanpy object."""

    # Create a custom DataFrameFanpy object
    index = [1, 2, 3, 4]
    params = [1.0, 0.75, 0.5, 0.25]

    df = DataFrameFanpy(wfn_label="params", wfn_params=params, wfn_index=index)

    assert df.shape == (len(params), 1)
    assert df.index.tolist() == index
    assert df["params"].tolist() == params


def test_update_dataframe():
    """Test wfn.analysis.dataframe.base.DataFrameFanpy.update_dataframe."""

    params = [1.0, 0.75, 0.5, 0.25]

    df1 = DataFrameFanpy()
    df2 = pd.DataFrame({"Wavefunction": params})
    df1.update_dataframe(df2)

    assert df1["Wavefunction"].tolist() == params


def test_base_dataframe_wfn():
    """Test creating an analysis.dataframe.base.DataFrameFanpy object."""

    # Create a custom BaseWavefunction object
    nelec = 4
    nspin = 8

    sds = sd_list.sd_list(nelec, nspin)
    params = np.random.rand(len(sds))

    wfn = BaseWavefunction(nelec, nspin)
    wfn.assign_params(params)

    # Create a DataFrameFanpy object
    df = DataFrameFanpy(wfn_label="Wavefunction", wfn_params=wfn.params, wfn_index=sds)

    assert df.shape == (len(sds), 1)
    assert df.index.tolist() == sds
    assert np.allclose(df["Wavefunction"], params)


def test_ci_dataframe_wfn():
    """Test creating an analysis.dataframe.ci.DataFrameCI object."""

    # Create a custom CIWavefunction object
    nelec = 4
    nspin = 8

    sds = sd_list.sd_list(nelec, nspin)

    wfn = CIWavefunction(nelec, nspin, sds=sds)

    params = np.random.rand(*wfn.params.shape)
    wfn.assign_params(params)

    # Create a DataFrameCI object
    df = DataFrameCI(wfn, wfn_label="CIWavefunction")

    assert df.shape == (len(sds), 1)
    assert df.index.tolist() == sds
    assert np.allclose(df["CIWavefunction"], params)


def test_ci_dataframe_formats():
    """Test analysis.dataframe.ci.DataFrameCI formatting methods."""

    # Create a custom CIWavefunction object
    nelec = 4
    nspin = 8
    nspatial = nspin // 2

    sds = sd_list.sd_list(nelec, nspin)

    wfn = CIWavefunction(nelec, nspin, sds=sds)

    # Create a DataFrameCI object
    df = DataFrameCI(wfn, wfn_label="CIWavefunction")

    # Convert Slater determinants from integers to formatted strings
    formatted_sds = [
        " ".join(
            [
                format(slater.split_spin(sd, nspatial)[0], f"0{nspatial}b")[::-1],
                format(slater.split_spin(sd, nspatial)[1], f"0{nspatial}b")[::-1],
            ]
        )
        for sd in sds
    ]

    # Convert Slater determinants from integers to strings in DataFrameCI
    df.set_formatted_sds_as_index()
    assert df.index.tolist() == formatted_sds

    # Convert Slater determinants from strings to integers in DataFrameCI
    df.set_sds_as_index()
    assert df.index.tolist() == sds


def test_cc_dataframe_wfn():
    """Test creating an analysis.dataframe.cc.DataFrameCC object."""

    # Create a custom CIWavefunction object
    nelec = 4
    nspin = 8
    ranks = [1, 2, 3, 4]

    counter = 0
    exops = {}
    for rank in ranks:
        for annihilators in combinations(range(nspin), rank):
            for creators in combinations([i for i in range(nspin) if i not in annihilators], rank):
                exops[(*annihilators, *creators)] = counter
                counter += 1

    wfn = BaseCC(nelec, nspin)
    wfn.exops = exops

    params = np.random.rand(len(exops))
    wfn.assign_params(params)

    # Convert operators from tuple to strings
    exops_str = []
    for excitation_op in exops:
        exops_str.append(" ".join(map(str, excitation_op)))

    # Create a DataFrameCC object
    df = DataFrameCC(wfn, wfn_label="BaseCC")

    assert df.shape == (len(exops), 1)
    assert df.index.tolist() == exops_str
    assert np.allclose(df["BaseCC"], params)


def test_cc_dataframe_formats():
    """Test analysis.dataframe.cc.DataFrameCC formatting methods."""

    # Create a StandardCC object
    nelec = 4
    nspin = 8
    nspatial = nspin // 2

    wfn = StandardCC(nelec, nspin)
    exops = wfn.exops.keys()
    ref_sd = wfn.refwfn

    # Convert operators from tuple to strings
    exops_str = []
    for excitation_op in exops:
        exops_str.append(" ".join(map(str, excitation_op)))

    # Create a DataFrameCC object
    df = DataFrameCC(wfn, wfn_label="BaseCC")

    sds = []

    # Convert excitation operators to Slater determinants format
    for excitation_op in exops:
        excitation_wfn = list(format(ref_sd, f"0{2*nspatial}b")[::-1])

        for index in excitation_op:
            if excitation_wfn[index] == "1":
                excitation_wfn[index] = "0"
            elif excitation_wfn[index] == "0":
                excitation_wfn[index] = "1"

        # Add the formatted excitation operator and parameter to the list
        excitation_wfn = int("".join(excitation_wfn[::-1]), 2)
        sds.append(excitation_wfn)

    # Convert excitation operators to Slater determinants in DataFrameCC
    df.set_sds_as_index()
    assert df.index.tolist() == sds

    # Convert Slater determinants from integers to formatted strings
    formatted_sds = [
        " ".join(
            [
                format(slater.split_spin(sd, nspatial)[0], f"0{nspatial}b")[::-1],
                format(slater.split_spin(sd, nspatial)[1], f"0{nspatial}b")[::-1],
            ]
        )
        for sd in sds
    ]

    # Convert Slater determinants integers to strings in DataFrameCC
    df.set_formatted_sds_as_index()
    assert df.index.tolist() == formatted_sds

    # Convert Slater determinants strings to operators indices in DataFrameCC
    df.set_operators_as_index()
    assert df.index.tolist() == exops_str


def test_geminal_dataframe_wfn():
    """Test creating an analysis.dataframe.geminal.DataFrameGeminal object."""

    # Create a custom BaseGeminal object
    nelec = 4
    nspin = 8
    ngem = 2

    orbpairs = tuple((i, j) for i in range(nspin) for j in range(i + 1, nspin))

    wfn = BaseGeminal(nelec, nspin, ngem=ngem, orbpairs=orbpairs)

    params = np.random.rand(len(orbpairs) * ngem)
    wfn.assign_params(params)

    # Create a DataFrameGeminal object
    df = DataFrameGeminal(wfn, wfn_label="BaseGeminal")

    orbpairs_str = tuple(" ".join(map(str, [i, j])) for i in range(nspin) for j in range(i + 1, nspin))
    ind_orbpairs = list(product(range(ngem), orbpairs_str))

    assert df.shape == (len(orbpairs) * ngem, 1)
    assert df.index.tolist() == ind_orbpairs
    assert np.allclose(df["BaseGeminal"], params)


def test_geminal_dataframe_formats():
    """Test analysis.dataframe.geminal.DataFrameGeminal formatting methods."""

    # Create a custom BaseGeminal object
    nelec = 4
    nspin = 8
    nspatial = nspin // 2

    wfn = BaseGeminal(nelec, nspin)

    geminals = [i for i in range(wfn.ngem)]
    orbpairs = tuple(" ".join(map(str, [i, j])) for i in range(nspin) for j in range(i + 1, nspin))
    ind_orbpairs = list(product(geminals, orbpairs))

    # Create a DataFrameGeminal object
    df = DataFrameGeminal(wfn, wfn_label="BaseGeminal")

    sds = []

    for orbital_pair in orbpairs:
        wfn_pair = [
            "0",
        ] * nspin

        for orbital in list(map(int, orbital_pair.split())):
            wfn_pair[orbital] = "1"

        wfn_pair = int("".join(wfn_pair[::-1]), 2)
        sds.append(wfn_pair)

    ind_sds = list(product(geminals, sds))

    # Convert geminal pairs to Slater determinants in DataFrameGeminal
    df.set_sds_as_index()
    assert df.index.tolist() == ind_sds

    # Convert Slater determinants from integers to formatted strings
    formatted_sds = [
        " ".join(
            [
                format(slater.split_spin(sd, nspatial)[0], f"0{nspatial}b")[::-1],
                format(slater.split_spin(sd, nspatial)[1], f"0{nspatial}b")[::-1],
            ]
        )
        for sd in sds
    ]

    ind_formatted_sds = list(product(geminals, formatted_sds))

    # Convert Slater determinants integers to strings in DataFrameGeminal
    df.set_formatted_sds_as_index()
    assert df.index.tolist() == ind_formatted_sds

    # Convert Slater determinants strings to geminal pairs in DataFrameGeminal
    df.set_geminals_as_index()
    assert df.index.tolist() == ind_orbpairs


def test_geminal_ref_sd_dataframe_formats():
    """Test analysis.dataframe.geminal.DataFrameGeminal formatting methods for Wavefunctions with reference Slater determinats."""

    # Create AP1roG object
    nelec = 4
    nspin = 8
    nspatial = nspin // 2

    wfn = AP1roG(nelec, nspin)

    geminals = list(wfn.dict_reforbpair_ind.keys())
    geminals_str = []
    for geminal in geminals:
        geminals_str.append(" ".join(map(str, geminal)))

    orbpairs = list(wfn.dict_orbpair_ind.keys())
    orbpairs_str = []
    for orbpair in orbpairs:
        orbpairs_str.append(" ".join(map(str, orbpair)))

    ind_orbpairs = list(product(geminals_str, orbpairs_str))

    # Create a DataFrameGeminal object
    df = DataFrameGeminal(wfn, wfn_label="APIG")

    ind_sds = []

    for geminal, orbital_pair in ind_orbpairs:
        sd_pair = list(format(wfn.ref_sd, f"0{2*nspatial}b")[::-1])

        for orbital in list(map(int, geminal.split())):
            sd_pair[orbital] = "0"

        for orbital in list(map(int, orbital_pair.split())):
            sd_pair[orbital] = "1"

        sd_pair = int("".join(sd_pair[::-1]), 2)
        ind_sds.append((geminal, sd_pair))

    # Convert geminal pairs to Slater determinants in DataFrameGeminal
    df.set_sds_as_index()
    assert df.index.tolist() == ind_sds

    # Convert Slater determinants from integers to formatted strings
    ind_formatted_sds = [
        (
            geminal,
            " ".join(
                [
                    format(slater.split_spin(sd, nspatial)[0], f"0{nspatial}b")[::-1],
                    format(slater.split_spin(sd, nspatial)[1], f"0{nspatial}b")[::-1],
                ]
            ),
        )
        for geminal, sd in ind_sds
    ]

    # Convert Slater determinants integers to strings in DataFrameGeminal
    df.set_formatted_sds_as_index()
    assert df.index.tolist() == ind_formatted_sds

    # Convert Slater determinants strings to geminal pairs in DataFrameGeminal
    df.set_geminals_as_index()
    assert df.index.tolist() == ind_orbpairs


def test_dataframe_io():
    """Test analysis.dataframe.base.DataFrameFanpy I/O methods."""

    nelec = 4
    nspin = 8

    sds = sd_list.sd_list(nelec, nspin)

    wfn = CIWavefunction(nelec, nspin, sds=sds)

    params = np.random.rand(*wfn.params.shape)
    wfn.assign_params(params)

    # Create a DataFrameCI object
    df = DataFrameCI(wfn, wfn_label="CIWavefunction")

    # Save the DataFrame to a CSV file
    df.to_csv("test.csv")

    # Load the DataFrame from the CSV file
    df_loaded = DataFrameCI()
    df_loaded.read_csv("test.csv")

    assert df_loaded.shape == df.shape
    assert df_loaded.index.tolist() == df.index.tolist()
    assert np.allclose(df_loaded["CIWavefunction"].tolist(), df["CIWavefunction"].tolist())
