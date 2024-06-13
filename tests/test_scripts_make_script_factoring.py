"""Test fanpy.script.make_script."""

import pytest
from fanpy.scripts.gaussian.wavefunction_info import get_wfn_info

def old_input(wfn_type, wfn_kwargs=None):
    from_imports = []
    if wfn_type == "ci_pairs":
        from_imports.append(("fanpy.wfn.ci.ci_pairs", "CIPairs"))
        wfn_name = "CIPairs"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "cisd":
        from_imports.append(("fanpy.wfn.ci.cisd", "CISD"))
        wfn_name = "CISD"
        if wfn_kwargs is None:
            wfn_kwargs = ""

    elif wfn_type == "hci":
        from_imports.append(("fanpy.wfn.ci.hci", "hCI"))
        wfn_name = "hCI"
        if wfn_kwargs is None:
            wfn_kwargs = "alpha1=0.5, alpha2=0.25, hierarchy=2.5"

    elif wfn_type == "fci":
        from_imports.append(("fanpy.wfn.ci.fci", "FCI"))
        wfn_name = "FCI"
        if wfn_kwargs is None:
            wfn_kwargs = "spin=None"
    elif wfn_type == "doci":
        from_imports.append(("fanpy.wfn.ci.doci", "DOCI"))
        wfn_name = "DOCI"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "mps":
        from_imports.append(("fanpy.wfn.network.mps", "MatrixProductState"))
        wfn_name = "MatrixProductState"
        if wfn_kwargs is None:
            wfn_kwargs = "dimension=None"
    elif wfn_type == "determinant-ratio":
        from_imports.append(("fanpy.wfn.quasiparticle.det_ratio", "DeterminantRatio"))
        wfn_name = "DeterminantRatio"
        if wfn_kwargs is None:
            wfn_kwargs = "numerator_mask=None"
    elif wfn_type == "ap1rog":
        from_imports.append(("fanpy.wfn.geminal.ap1rog", "AP1roG"))
        wfn_name = "AP1roG"
        if wfn_kwargs is None:
            wfn_kwargs = "ref_sd=None, ngem=None"
    elif wfn_type == "apr2g":
        from_imports.append(("fanpy.wfn.geminal.apr2g", "APr2G"))
        wfn_name = "APr2G"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apig":
        from_imports.append(("fanpy.wfn.geminal.apig", "APIG"))
        wfn_name = "APIG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apsetg":
        from_imports.append(("fanpy.wfn.geminal.apsetg", "BasicAPsetG"))
        wfn_name = "BasicAPsetG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apg":  # pragma: no branch
        from_imports.append(("fanpy.wfn.geminal.apg", "APG"))
        wfn_name = "APG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "network":
        from_imports.append(("fanpy.upgrades.numpy_network", "NumpyNetwork"))
        wfn_name = "NumpyNetwork"
        if wfn_kwargs is None:
            wfn_kwargs = "num_layers=2"
    elif wfn_type == "rbm":
        from_imports.append(("fanpy.wfn.network.rbm_standard", "RestrictedBoltzmannMachine"))
        wfn_name = "RestrictedBoltzmannMachine"
        if wfn_kwargs is None:
            wfn_kwargs = "nbath=nspin, num_layers=1, orders=(1, 2)"

    elif wfn_type == "basecc":
        from_imports.append(("fanpy.wfn.cc.base", "BaseCC"))
        wfn_name = "BaseCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "standardcc":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "generalizedcc":
        from_imports.append(("fanpy.wfn.cc.generalized_cc", "GeneralizedCC"))
        wfn_name = "GeneralizedCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "senioritycc":
        from_imports.append(("fanpy.wfn.cc.seniority_cc", "SeniorityCC"))
        wfn_name = "SeniorityCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "pccd":
        from_imports.append(("fanpy.wfn.cc.pccd_ap1rog", "PCCD"))
        wfn_name = "PCCD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ap1rogsd":
        from_imports.append(("fanpy.wfn.cc.ap1rog_generalized", "AP1roGSDGeneralized"))
        wfn_name = "AP1roGSDGeneralized"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ap1rogsd_spin":
        from_imports.append(("fanpy.wfn.cc.ap1rog_spin", "AP1roGSDSpin"))
        wfn_name = "AP1roGSDSpin"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apsetgd":
        from_imports.append(("fanpy.wfn.cc.apset1rog_d", "APset1roGD"))
        wfn_name = "APset1roGD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apsetgsd":
        from_imports.append(("fanpy.wfn.cc.apset1rog_sd", "APset1roGSD"))
        wfn_name = "APset1roGSD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apg1rod":
        from_imports.append(("fanpy.wfn.cc.apg1ro_d", "APG1roD"))
        wfn_name = "APG1roD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apg1rosd":
        from_imports.append(("fanpy.wfn.cc.apg1ro_sd", "APG1roSD"))
        wfn_name = "APG1roSD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdsen0":
        from_imports.append(("fanpy.wfn.cc.ccsd_sen0", "CCSDsen0"))
        wfn_name = "CCSDsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdqsen0":
        from_imports.append(("fanpy.wfn.cc.ccsdq_sen0", "CCSDQsen0"))
        wfn_name = "CCSDQsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdtqsen0":
        from_imports.append(("fanpy.wfn.cc.ccsdtq_sen0", "CCSDTQsen0"))
        wfn_name = "CCSDTQsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdtsen2qsen0":
        from_imports.append(("fanpy.wfn.cc.ccsdt_sen2_q_sen0", "CCSDTsen2Qsen0"))
        wfn_name = "CCSDTsen2Qsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsd":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
        print(wfn_kwargs)
        wfn_kwargs = f"ranks=[1, 2], {wfn_kwargs}"
    elif wfn_type == "ccsdt":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
        wfn_kwargs = f"ranks=[1, 2, 3], {wfn_kwargs}"
    elif wfn_type == "ccsdtq":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
        wfn_kwargs = f"ranks=[1, 2, 3, 4], {wfn_kwargs}"
    return from_imports, wfn_name, wfn_kwargs

def test_make_script():
    wfn_list = [
        "ci_pairs",
        "cisd",
        "hci", # "hci" is missing in the original test
        "fci",
        "doci",
        "mps",
        "determinant-ratio",
        "ap1rog",
        "apr2g",
        "apig",
        "apsetg",
        "apg",
        "network",
        "rbm",
        "basecc",
        "standardcc",
        "generalizedcc",
        "senioritycc",
        "pccd",
        "ap1rogsd",
        "ap1rogsd_spin",
        "apsetgd",
        "apsetgsd",
        "apg1rod",
        "apg1rosd",
        "ccsdsen0",
        "ccsdqsen0",
        "ccsdtqsen0",
        "ccsdtsen2qsen0",
        "ccsd",
        "ccsdt",
        "ccsdtq"
    ]
    for wfn in wfn_list:
        wfn_info = get_wfn_info(wfn)
        import_line, wfn_name, wfn_kwargs = wfn_info(wfn_kwargs=None)
        old_import_line, old_wfn_name, old_wfn_kwargs = old_input(wfn)
        assert import_line == old_import_line[0]
        assert wfn_name == old_wfn_name
        assert wfn_kwargs == old_wfn_kwargs

def test_import_line():
    wfn_list = [
        "ci_pairs",
        "cisd",
        "hci", # "hci" is missing in the original test
        "fci",
        "doci",
        "mps",
        "determinant-ratio",
        "ap1rog",
        "apr2g",
        "apig",
        "apsetg",
        "apg",
        "network",
        "rbm",
        "basecc",
        "standardcc",
        "generalizedcc",
        "senioritycc",
        "pccd",
        "ap1rogsd",
        "ap1rogsd_spin",
        "apsetgd",
        "apsetgsd",
        "apg1rod",
        "apg1rosd",
        "ccsdsen0",
        "ccsdqsen0",
        "ccsdtqsen0",
        "ccsdtsen2qsen0",
        "ccsd",
        "ccsdt",
        "ccsdtq"
    ]
    for wfn in wfn_list:
        wfn_info = get_wfn_info(wfn)
        import_line, wfn_name, wfn_kwargs = wfn_info(wfn_kwargs=None)
        command = f"from {import_line[0]} import {import_line[1]}"
        try:
            exec(command)
        except ModuleNotFoundError:
            pytest.fail(f"ModuleNotFoundError: {import_line[0]}")

@pytest.mark.parametrize("wfn_type", ["42 is the answer to everything", 41, None, 3.14])
def test_invalid_wfn_type(wfn_type):
    with pytest.raises(ValueError):
        get_wfn_info(wfn_type)