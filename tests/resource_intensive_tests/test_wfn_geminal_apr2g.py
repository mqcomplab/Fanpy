"""Test fanpy.wavefunction.geminals.apr2g.APr2G."""
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.senzero import SeniorityZeroHamiltonian
from fanpy.solver.equation import cma, minimize
from fanpy.solver.system import least_squares
from fanpy.wfn.geminal.apr2g import APr2G
from fanpy.wfn.geminal.rank2_approx import full_to_rank2

import numpy as np

from utils import find_datafile


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apr2g_h2_631gdp():
    """Find the APr2G/6-31G** wavefunction by scanning through the coefficients.

    Notes
    -----
    Uses APIG answer from test_apr2g_apig.answer_apig_h2_631gdp, converting it to APr2G.

    """
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int)

    full_sds = (
        0b00000000010000000001,
        0b00000000100000000010,
        0b00000001000000000100,
        0b00000010000000001000,
        0b00000100000000010000,
        0b00001000000000100000,
        0b00010000000001000000,
        0b00100000000010000000,
        0b01000000000100000000,
        0b10000000001000000000,
    )
    apr2g = APr2G(
        2,
        20,
        params=full_to_rank2(
            np.array(
                [
                    [
                        0.995079200788,
                        -0.059166892062,
                        -0.054284175189,
                        -0.036920061272,
                        -0.028848919079,
                        -0.028847742282,
                        -0.013108383833,
                        -0.008485392433,
                        -0.008485285973,
                        -0.005149411511,
                    ]
                ]
            )
        ),
    )
    objective = EnergyOneSideProjection(apr2g, ham, refwfn=full_sds)
    results = minimize(objective)
    print(results)
    print(apr2g.params)


def test_apr2g_apr2g_h2_631gdp_slow():
    """Test ground state APr2G wavefunction using H2 with HF/6-31G** orbitals.

    Answers obtained from answer_apr2g_h2_631gdp

    HF (Electronic) Energy : -1.84444667247
    APr2G Energy : -1.86968286083049e+00
    APr2G Coeffs : [5.00024139947352e-01, -5.04920993272312e-01, 1.74433861913406e-03,
                    1.46912778996944e-03, 6.80650579364171e-04, 4.15804099437146e-04,
                    4.15770205441596e-04, 8.59042503314636e-05, 3.60000884233533e-05,
                    3.59991851562772e-05, 1.32585077026253e-05, 9.99999999977375e-01,
                    -2.94816671672589e-02, -2.70636476435323e-02, -1.84357922467577e-02,
                    -1.44131604476661e-02, -1.44125734820933e-02, -6.55338227991008e-03,
                    -4.24259557902134e-03, -4.24254235743836e-03, -2.57476173228552e-03,]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apr2g = APr2G(2, 20)
    full_sds = (
        0b00000000010000000001,
        0b00000000100000000010,
        0b00000001000000000100,
        0b00000010000000001000,
        0b00000100000000010000,
        0b00001000000000100000,
        0b00010000000001000000,
        0b00100000000010000000,
        0b01000000000100000000,
        0b10000000001000000000,
    )

    # Solve system of equations
    objective = ProjectedSchrodinger(apr2g, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], -1.86968286083049)


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apr2g_lih_sto6g():
    """Find the APr2G/STO-6G wavefunction for LiH by scanning through the coefficients.

    Notes
    -----
    Uses APIG answer from test_apr2g_apig.answer_apig_lih_sto6g, converting it to APr2G.

    """
    one_int = np.load(find_datafile("data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int)

    apr2g = APr2G(
        4,
        12,
        params=full_to_rank2(
            np.array(
                [
                    [
                        1.0335931818e00,
                        3.1309033507e-04,
                        -4.3212475389e-03,
                        -1.7672513953e-03,
                        -1.7692149535e-03,
                        -1.1697291799e-03,
                    ],
                    [
                        -5.3278893571e-01,
                        9.6025806293e-01,
                        -1.1398393606e-02,
                        -2.8586983706e-02,
                        -2.8782700436e-02,
                        -1.1293245734e-01,
                    ],
                ]
            )
        ),
    )
    full_sds = (
        0b000011000011,
        0b000101000101,
        0b001001001001,
        0b010001010001,
        0b100001100001,
        0b000110000110,
        0b001010001010,
        0b010010010010,
        0b100010100010,
        0b001100001100,
        0b010100010100,
        0b100100100100,
        0b011000011000,
        0b101000101000,
        0b110000110000,
    )

    objective = EnergyOneSideProjection(apr2g, ham, refwfn=full_sds)
    results = cma(objective)
    results = minimize(objective)
    print(results)
    print(apr2g.params)


def test_apr2g_apr2g_lih_sto6g_slow():
    """Test APr2G wavefunction using LiH with HF/STO-6G orbitals.

    Answers obtained from answer_apr2g_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    APIG Energy : -8.96353110958190e+00
    APIG Coeffs : [2.52192912907165e+00, -3.22730595025637e-01, 1.55436195824780e+00,
                   -3.23195369339517e-01, -2.05933948914417e+00, -5.09311039678737e-01,
                   -5.08826074574783e-01, -3.52888520424345e-01, 9.99590527398775e-01,
                   4.46478511362153e-04, -1.98165460930347e-02, -5.36827124240751e-03,
                   -5.35329619359940e-03, -3.40758975682932e-03]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apr2g = APr2G(4, 12)
    full_sds = (
        0b000011000011,
        0b000101000101,
        0b001001001001,
        0b010001010001,
        0b100001100001,
        0b000110000110,
        0b001010001010,
        0b010010010010,
        0b100010100010,
        0b001100001100,
        0b010100010100,
        0b100100100100,
        0b011000011000,
        0b101000101000,
        0b110000110000,
    )

    # Solve system of equations
    objective = ProjectedSchrodinger(apr2g, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], -8.96353110958190)
