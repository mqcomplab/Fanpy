"""Test fanpy.wavefunction.doci."""
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.ham.senzero import SeniorityZeroHamiltonian
from fanpy.solver.equation import cma
from fanpy.wfn.ci.doci import DOCI

import numpy as np

import pytest

from utils import find_datafile, skip_init

def test_doci_h4_hf_sto6g():
    """Test DOCI wavefunction for square H4 (STO-6G) against Peter's orbital optimized DOCI results.

    HF energy: -1.13126983927
    OO DOCI energy: -1.884948574812363

    NOTE
    ----
    Optimized orbitals are read in from Peter's code
    """
    nelec = 4
    nspin = 8
    doci = DOCI(nelec, nspin)

    # orbital rotation obtained from Peter Limacher's code
    # orb_rot = np.array([[0.707106752870, -0.000004484084, 0.000006172115, -0.707106809462],
    #                     [0.707106809472, -0.000004868924, -0.000006704609, 0.707106752852],
    #                     [0.000004942751, 0.707106849959, 0.707106712365, 0.000006630781],
    #                     [0.000004410256, 0.707106712383, -0.707106849949, -0.000006245943]])

    one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))
    nuc_nuc = 2.70710678119

    # rotate orbitals
    # one_int = orb_rot.T.dot(one_int).dot(orb_rot)
    # two_int = np.einsum('ijkl,ia->ajkl', two_int, orb_rot)
    # two_int = np.einsum('ajkl,jb->abkl', two_int, orb_rot)
    # two_int = np.einsum('abkl,kc->abcl', two_int, orb_rot)
    # two_int = np.einsum('abcl,ld->abcd', two_int, orb_rot)

    # NOTE: This works with random guess for the parameters most of the times but not always.
    #       So a "good" initial guess is used to converge to a consistent result
    ham = SeniorityZeroHamiltonian(
        one_int, two_int, params=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5])
    )

    obj = EnergyOneSideProjection(
        doci,
        ham,
        param_selection=[
            (doci, np.ones(doci.nparams, dtype=bool)),
            (ham, np.ones(ham.nparams, dtype=bool)),
        ],
    )
    results = cma(
        obj,
        sigma0=0.01,
        options={
            "ftarget": None,
            "timeout": np.inf,
            "tolfun": 1e-11,
            "verb_filenameprefix": "outcmaes",
            "verb_log": 0,
        },
    )
    energy = results["function"]

    # compare with number from Gaussian
    assert abs(energy + nuc_nuc - (-1.884948574812363)) < 1e-7


def test_doci_h2_hf_631gdp_slow():
    """Test DOCI wavefunction for H2 (6-31G**)."""
    nelec = 2
    nspin = 20
    doci = DOCI(nelec, nspin)

    # transform orbitals
    # Peter's orbitals need to be reordered
    # orb_rot = np.array([[0.488169047143, -0.543185538076, -0.411601097620, -0.000000403150,
    #                      0.000001613245, -0.388739227860, 0.303171079258, 0.000006888161,
    #                      -0.000003111236, 0.232796259502],
    #                     [0.503298667825, -0.293907169278, 0.461587797414, -0.000000054879,
    #                      -0.000002339911, 0.579287242867, -0.182498320074, -0.000009461656,
    #                      0.000004256464, 0.279939967492],
    #                     [0.000000002376, -0.000000458104, -0.000000245684, 0.707091981947,
    #                      -0.003178725269, 0.000005319524, -0.000000256975, -0.001618581793,
    #                      -0.707112582929, -0.000000446558],
    #                     [-0.000000013848, 0.000000722493, -0.000001875001, 0.003178673760,
    #                      0.707091982001, -0.000007897018, -0.000000481474, -0.707112583195,
    #                      0.001618530225, 0.000001889435],
    #                     [0.091549505235, 0.344332883161, -0.342816419640, -0.000000734051,
    #                      -0.000000876752, -0.117351069059, -0.611827958397, 0.000003721555,
    #                      -0.000001885537, 0.606156524329],
    #                     [0.488169105550, 0.543185764514, -0.411602641767, 0.000000325673,
    #                      -0.000000886083, 0.387810769711, 0.304363736938, -0.000004400987,
    #                      0.000003086321, -0.232785174377],
    #                     [0.503298863137, 0.293908060439, 0.461584812911, 0.000000136369,
    #                      0.000001903559, -0.578727232405, -0.184262403760, 0.000006801308,
    #                      -0.000004346795, -0.279946287399],
    #                     [0.000000009714, 0.000000469899, 0.000000085206, 0.707107221036,
    #                      -0.003194171012, -0.000005808918, 0.000000207363, 0.001603067038,
    #                      0.707097309689, 0.000001905864],
    #                     [-0.000000002638, -0.000000370181, 0.000001927471, 0.003194222521,
    #                      0.707107220974, 0.000012676707, -0.000001105440, 0.707097309311,
    #                      -0.001603118441, -0.000000680451],
    #                     [-0.091549435933, 0.344340224027, 0.342813170568, -0.000000606375,
    #                      0.000000474065, -0.119224270382, 0.611439523904, 0.000002445211,
    #                      -0.000002458517, 0.606180602568]])

    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    nuc_nuc = 0.71317683129

    # rotate orbitals
    # one_int = orb_rot.T.dot(one_int).dot(orb_rot)
    # two_int = np.einsum('ijkl,ia->ajkl', two_int, orb_rot)
    # two_int = np.einsum('ajkl,jb->abkl', two_int, orb_rot)
    # two_int = np.einsum('abkl,kc->abcl', two_int, orb_rot)
    # two_int = np.einsum('abcl,ld->abcd', two_int, orb_rot)

    ham = SeniorityZeroHamiltonian(one_int, two_int, params=np.random.rand(45))
    obj = EnergyOneSideProjection(
        doci,
        ham,
        param_selection=[
            (doci, np.ones(doci.nparams, dtype=bool)),
            (ham, np.ones(ham.nparams, dtype=bool)),
        ],
    )
    results = cma(obj)
    energy = results["energy"]

    # compare with number from Gaussian
    assert abs(energy + nuc_nuc - (-1.1651486697)) < 1e-7