"""Factory for importing and initializing wavefunction classes.

This module provides a factory function `get_wfn_info` that takes a wavefunction type as input and returns the corresponding 
private function.

Methods
-------
get_wfn_info(wfn_type)
    Returns the private function for the given wavefunction type. All private functions return 
    the import line, wavefunction name, and wavefunction arguments for their wavefunction type.

Private Functions
-----------------
_get_ci_pairs_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ci_pairs" wavefunction type.
_get_cisd_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "cisd" wavefunction type.
_get_hci_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "hci" wavefunction type.
_get_fci_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "fci" wavefunction type.
_get_doci_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "doci" wavefunction type.
_get_mps_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "mps" wavefunction type.
_get_det_ratio_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "determinant-ratio" wavefunction type.
_get_ap1rog_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ap1rog" wavefunction type.
_get_apr2g_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apr2g" wavefunction type.
_get_apig_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apig" wavefunction type.
_get_apsetg_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apsetg" wavefunction type.
_get_apg_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apg" wavefunction type.
_get_network_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "network" wavefunction type.
_get_rbm_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "rbm" wavefunction type.
_get_basecc_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "basecc" wavefunction type.
_get_standardcc_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "standardcc" wavefunction type.
_get_generalizedcc_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "generalizedcc" wavefunction type.
_get_senioritycc_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "senioritycc" wavefunction type.
_get_pccd_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "pccd" wavefunction type.
_get_ap1rogsd_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ap1rogsd" wavefunction type.
_get_ap1rogsd_spin_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ap1rogsd_spin" wavefunction type.
_get_apsetgd_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apsetgd" wavefunction type.
_get_apsetgsd_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apsetgsd" wavefunction type.
_get_apg1rod_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apg1rod" wavefunction type.
_get_apg1rosd_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "apg1rosd" wavefunction type.
_get_ccsdsen0_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ccsdsen0" wavefunction type.
_get_ccsdqsen0_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ccsdqsen0" wavefunction type.
_get_ccsdtqsen0_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ccsdtqsen0" wavefunction type.
_get_ccsdtsen2qsen0_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ccsdtsen2qsen0" wavefunction type.
_get_ccsd_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ccsd" wavefunction type.
_get_ccsdt_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ccsdt" wavefunction type.
_get_ccsdtq_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "ccsdtq" wavefunction type.
_get_custom_info(wfn_kwargs)
    Returns the import line, wavefunction name, and wavefunction arguments for the "custom" wavefunction type.

"""

def get_wfn_info(wfn_type):
    """
    Returns the corresponding wavefunction information function based on the given wavefunction type.

    Parameters
    ----------
    wfn_type : str
        The type of wavefunction.

    Returns
    -------
    function : function
        The wavefunction information function corresponding to the given wavefunction type.

    Raises 
    ------
    ValueError
        If the given wavefunction type is invalid.
    """

    if wfn_type == "ci_pairs":
        return _get_ci_pairs_info
    elif wfn_type == "cisd":
        return _get_cisd_info
    elif wfn_type == "hci":
        return _get_hci_info
    elif wfn_type == "fci":
        return _get_fci_info
    elif wfn_type == "doci":
        return _get_doci_info
    elif wfn_type == "mps":
        return _get_mps_info
    elif wfn_type == "determinant-ratio":
        return _get_det_ratio_info
    elif wfn_type == "ap1rog":
        return _get_ap1rog_info
    elif wfn_type == "apr2g":
        return _get_apr2g_info
    elif wfn_type == "apig":
        return _get_apig_info
    elif wfn_type == "apsetg":
        return _get_apsetg_info
    elif wfn_type == "apg":
        return _get_apg_info
    elif wfn_type == "network": 
        return _get_network_info
    elif wfn_type == "rbm":
        return _get_rbm_info
    elif wfn_type == "basecc":
        return _get_basecc_info
    elif wfn_type == "standardcc":
        return _get_standardcc_info
    elif wfn_type == "generalizedcc":
        return _get_generalizedcc_info
    elif wfn_type == "senioritycc":
        return _get_senioritycc_info
    elif wfn_type == "pccd":
        return _get_pccd_info
    elif wfn_type == "ap1rogsd":
        return _get_ap1rogsd_info
    elif wfn_type == "ap1rogsd_spin":
        return _get_ap1rogsd_spin_info
    elif wfn_type == "apsetgd":
        return _get_apsetgd_info
    elif wfn_type == "apsetgsd":
        return _get_apsetgsd_info
    elif wfn_type == "apg1rod":
        return _get_apg1rod_info
    elif wfn_type == "apg1rosd":
        return _get_apg1rosd_info
    elif wfn_type == "ccsdsen0":
        return _get_ccsdsen0_info
    elif wfn_type == "ccsdqsen0":
        return _get_ccsdqsen0_info
    elif wfn_type == "ccsdtqsen0":
        return _get_ccsdtqsen0_info
    elif wfn_type == "ccsdtsen2qsen0":
        return _get_ccsdtsen2qsen0_info
    elif wfn_type == "ccs":
        return _get_ccs_info
    elif wfn_type == "ccsd":
        return _get_ccsd_info
    elif wfn_type == "ccsdt":
        return _get_ccsdt_info
    elif wfn_type == "ccsdtq":
        return _get_ccsdtq_info
    elif wfn_type == "custom":
        return _get_custom_info
    else:
        raise ValueError("Invalid wavefunction type: {}".format(wfn_type))


def _get_ci_pairs_info(wfn_kwargs):
    """
    Get the information about the CI pairs.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the CI pairs.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the CI pairs class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.ci.ci_pairs", "CIPairs")
    wfn_name = "CIPairs"
    if wfn_kwargs is None:
        wfn_kwargs = ""
    return import_line, wfn_name, wfn_kwargs

def _get_cisd_info(wfn_kwargs):
    """
    Get the information about CISD.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for CISD.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the CISD class, and the additional keyword arguments.

    """

    import_line = ("fanpy.wfn.ci.cisd", "CISD")
    wfn_name = "CISD"
    if wfn_kwargs is None:
        wfn_kwargs = ""
    return import_line, wfn_name, wfn_kwargs

def _get_hci_info(wfn_kwargs):
    """
    Get the information about HCI.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for HCI.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the HCI class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.ci.hci", "hCI")
    wfn_name = "hCI"
    if wfn_kwargs is None:
        wfn_kwargs = "alpha1=0.5, alpha2=0.25, hierarchy=2.5"
    return import_line, wfn_name, wfn_kwargs

def _get_fci_info(wfn_kwargs):
    """
    Get the information about FCI.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for FCI.
    
    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the FCI class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.ci.fci", "FCI")
    wfn_name = "FCI"
    if wfn_kwargs is None:
        wfn_kwargs = "spin=None"
    return import_line, wfn_name, wfn_kwargs

def _get_doci_info(wfn_kwargs):
    """
    Get the information about DOCI.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for DOCI.
    
    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the DOCI class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.ci.doci", "DOCI")
    wfn_name = "DOCI"
    if wfn_kwargs is None:
        wfn_kwargs = ""
    return import_line, wfn_name, wfn_kwargs

def _get_mps_info(wfn_kwargs):
    """
    Get the information about MPS.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for MPS.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the MPS class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.network.mps", "MatrixProductState")
    wfn_name = "MatrixProductState"
    if wfn_kwargs is None:
        wfn_kwargs = "dimension=None"
    return import_line, wfn_name, wfn_kwargs

def _get_det_ratio_info(wfn_kwargs):
    """
    Get the information about the DeterminantRatio wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the DeterminantRatio wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the DeterminantRatio class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.quasiparticle.det_ratio", "DeterminantRatio")
    wfn_name = "DeterminantRatio"
    if wfn_kwargs is None:
        wfn_kwargs = "numerator_mask=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ap1rog_info(wfn_kwargs):
    """
    Get the information about the AP1roG wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the AP1roG wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the AP1roG class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.geminal.ap1rog", "AP1roG")
    wfn_name = "AP1roG"
    if wfn_kwargs is None:
        wfn_kwargs = "ref_sd=None, ngem=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apr2g_info(wfn_kwargs):
    """
    Get the information about the APr2G wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the APr2G wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the APr2G class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.geminal.apr2g", "APr2G")
    wfn_name = "APr2G"
    if wfn_kwargs is None:
        wfn_kwargs = "ngem=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apig_info(wfn_kwargs):
    """
    Get the information about the APIG wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the APIG wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the APIG class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.geminal.apig", "APIG")
    wfn_name = "APIG"
    if wfn_kwargs is None:
        wfn_kwargs = "ngem=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apsetg_info(wfn_kwargs):
    """
    Get the information about the APsetG wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None    
        Additional keyword arguments for the APsetG wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the APsetG class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.geminal.apsetg", "BasicAPsetG")
    wfn_name = "BasicAPsetG"
    if wfn_kwargs is None:
        wfn_kwargs = "ngem=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apg_info(wfn_kwargs):
    """
    Get the information about the APG wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the APG wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the APG class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.geminal.apg", "APG")
    wfn_name = "APG"
    if wfn_kwargs is None:
        wfn_kwargs = "ngem=None"
    return import_line, wfn_name, wfn_kwargs

def _get_network_info(wfn_kwargs):
    """
    Get the information about the network wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the network wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the network class, and the additional keyword arguments.
    """

    import_line = ("fanpy.upgrades.numpy_network", "NumpyNetwork")
    wfn_name = "NumpyNetwork"
    if wfn_kwargs is None:
        wfn_kwargs = "num_layers=2"
    return import_line, wfn_name, wfn_kwargs

def _get_rbm_info(wfn_kwargs):
    """
    Get the information about the restricted Boltzmann machine wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Additional keyword arguments for the restricted Boltzmann machine wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, the name of the restricted Boltzmann machine class, and the additional keyword arguments.
    """

    import_line = ("fanpy.wfn.network.rbm_standard", "RestrictedBoltzmannMachine")
    wfn_name = "RestrictedBoltzmannMachine"
    if wfn_kwargs is None:
        wfn_kwargs = "nbath=nspin, num_layers=1, orders=(1, 2)"
    return import_line, wfn_name, wfn_kwargs

def _get_basecc_info(wfn_kwargs):
    """
    Get information about the BaseCC wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None 
        Keyword arguments for the BaseCC wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.base", "BaseCC")
    wfn_name = "BaseCC"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_standardcc_info(wfn_kwargs):
    """
    Get information about the StandardCC wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the StandardCC wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.standard_cc", "StandardCC")
    wfn_name = "StandardCC"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_generalizedcc_info(wfn_kwargs):
    """
    Get information about the GeneralizedCC wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the GeneralizedCC wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.generalized_cc", "GeneralizedCC")
    wfn_name = "GeneralizedCC"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_senioritycc_info(wfn_kwargs):
    """
    Get information about the SeniorityCC wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the SeniorityCC wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.seniority_cc", "SeniorityCC")
    wfn_name = "SeniorityCC"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_pccd_info(wfn_kwargs):
    """
    Get information about the PCCD wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the PCCD wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.pccd_ap1rog", "PCCD")
    wfn_name = "PCCD"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ap1rogsd_info(wfn_kwargs):
    """
    Get information about the AP1roGSD wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the AP1roGSD wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.ap1rog_generalized", "AP1roGSDGeneralized")
    wfn_name = "AP1roGSDGeneralized"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ap1rogsd_spin_info(wfn_kwargs):
    """
    Get information about the AP1roGSDSpin wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the AP1roGSDSpin wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.ap1rog_spin", "AP1roGSDSpin")
    wfn_name = "AP1roGSDSpin"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apsetgd_info(wfn_kwargs):
    """
    Get information about the APset1roGD wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the APset1roGD wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.apset1rog_d", "APset1roGD")
    wfn_name = "APset1roGD"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apsetgsd_info(wfn_kwargs):
    """
    Get information about the APset1roGSD wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the APset1roGSD wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.apset1rog_sd", "APset1roGSD")
    wfn_name = "APset1roGSD"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apg1rod_info(wfn_kwargs):
    """
    Get information about the APG1roD wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the APG1roD wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.apg1ro_d", "APG1roD")
    wfn_name = "APG1roD"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_apg1rosd_info(wfn_kwargs):
    """
    Get information about the APG1roSD wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the APG1roSD wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.apg1ro_sd", "APG1roSD")
    wfn_name = "APG1roSD"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ccsdsen0_info(wfn_kwargs):
    """
    Get information about the CCSDsen0 wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCSDsen0 wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.ccsd_sen0", "CCSDsen0")
    wfn_name = "CCSDsen0"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ccsdqsen0_info(wfn_kwargs):
    """
    Get information about the CCSDQsen0 wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCSDQsen0 wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.ccsdq_sen0", "CCSDQsen0")
    wfn_name = "CCSDQsen0"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ccsdtqsen0_info(wfn_kwargs):
    """
    Get information about the CCSDTQsen0 wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCSDTQsen0 wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.ccsdtq_sen0", "CCSDTQsen0")
    wfn_name = "CCSDTQsen0"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ccsdtsen2qsen0_info(wfn_kwargs):
    """
    Get information about the CCSDTsen2Qsen0 wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCSDTsen2Qsen0 wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.ccsdt_sen2_q_sen0", "CCSDTsen2Qsen0")
    wfn_name = "CCSDTsen2Qsen0"
    if wfn_kwargs is None:
        wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    return import_line, wfn_name, wfn_kwargs

def _get_ccsd_info(wfn_kwargs):
    """
    Get information about the CCSD wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCSD wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.standard_cc", "StandardCC")
    wfn_name = "StandardCC"
    if wfn_kwargs is None:
        wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
    wfn_kwargs = f"ranks=[1, 2], {wfn_kwargs}"
    return import_line, wfn_name, wfn_kwargs

def _get_ccsdt_info(wfn_kwargs):
    """
    Get information about the CCSDT wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCSDT wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.standard_cc", "StandardCC")
    wfn_name = "StandardCC"
    if wfn_kwargs is None:
        wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
    wfn_kwargs = f"ranks=[1, 2, 3], {wfn_kwargs}"
    return import_line, wfn_name, wfn_kwargs

def _get_ccsdtq_info(wfn_kwargs):
    """
    Get information about the CCSDTQ wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCSDTQ wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.standard_cc", "StandardCC")
    wfn_name = "StandardCC"
    if wfn_kwargs is None:
        wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
    wfn_kwargs = f"ranks=[1, 2, 3, 4], {wfn_kwargs}"
    return import_line, wfn_name, wfn_kwargs

def _get_ccs_info(wfn_kwargs):
    """
    Get information about the CCS wavefunction.

    Parameters
    ----------
    wfn_kwargs : str, None
        Keyword arguments for the CCS wavefunction.

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.
    """

    import_line = ("fanpy.wfn.cc.standard_cc", "StandardCC")
    wfn_name = "StandardCC"
    if wfn_kwargs is None:
        wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
    wfn_kwargs = f"ranks=[1], {wfn_kwargs}"
    return import_line, wfn_name, wfn_kwargs

def _get_custom_info(wfn_kwargs):
    """
    Get information about the custom wavefunction.

    Parameters
    ----------
    wfn_kwargs : str
        Keyword arguments for the custom wavefunction. Must contain "import_line" and "wfn_name".

    Returns
    -------
    results : tuple
        A tuple containing the import line, wavefunction name, and keyword arguments.

    Raises
    ------
    ValueError
        If the wfn_kwargs is not a string or does not contain "import_line" and "wfn_name".
    """
    if type(wfn_kwargs) is not str:
        raise ValueError("wfn_kwargs for custom wavefunction must be a string.")
    if "import_line" not in wfn_kwargs:
        raise ValueError("Custom wavefunction must specify import_line.")
    if "wfn_name" not in wfn_kwargs:
        raise ValueError("Custom wavefunction must specify wfn_name.")
    
    wfn_kwarg_list = wfn_kwargs.split(",")
    output_wfn_kwargs = ""
    for kwarg in wfn_kwarg_list:
        if "import_line" in kwarg:
            import_line = kwarg.split("=")[1]
            import_line = import_line.strip()
        elif "wfn_name" in kwarg:
            wfn_name = kwarg.split("=")[1]
            wfn_name = wfn_name.strip()
        else:
            output_wfn_kwargs += kwarg + ","
    import_line = (import_line, wfn_name)
    return import_line, wfn_name, output_wfn_kwargs[:-1]