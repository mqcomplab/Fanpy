"""Test fanpy.wavefunction.geminals.apig."""
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.senzero import SeniorityZeroHamiltonian
from fanpy.solver.equation import minimize
from fanpy.solver.system import least_squares
from fanpy.wfn.geminal.apig import APIG

import numpy as np

import pytest

import scipy

from utils import find_datafile, skip_init


def test_apig_spin():
    """Test APIG.spin."""
    test = skip_init(APIG)
    assert test.spin == 0


def test_apig_seniority():
    """Test APIG.seniority."""
    test = skip_init(APIG)
    assert test.seniority == 0


def test_apig_assign_orbpairs():
    """Test APIG.assign_orbpairs."""
    test = skip_init(APIG)
    test.assign_nspin(10)
    test.assign_orbpairs()
    assert test.dict_orbpair_ind == {(0, 5): 0, (1, 6): 1, (2, 7): 2, (3, 8): 3, (4, 9): 4}
    test.assign_orbpairs(((0, 5), (2, 7), (1, 6), (3, 8), (4, 9)))
    assert test.dict_orbpair_ind == {(0, 5): 0, (1, 6): 2, (2, 7): 1, (3, 8): 3, (4, 9): 4}
    with pytest.raises(ValueError):
        test.assign_orbpairs(((0, 1), (2, 3), (4, 5), (5, 7), (6, 7)))
    with pytest.raises(ValueError):
        test.assign_orbpairs(((0, 1), (2, 3), (4, 5)))


def test_apig_generate_possible_orbpairs():
    """Test APIG.generate_possible_orbpair."""
    test = skip_init(APIG)
    test.assign_nspin(10)
    test.assign_nelec(6)
    test.assign_orbpairs()
    possible_orbpairs = list(test.generate_possible_orbpairs([0, 5, 2, 7, 4, 9]))
    assert len(possible_orbpairs) == 1
    assert len(possible_orbpairs[0][0]) == 3
    assert (0, 5) in possible_orbpairs[0][0]
    assert (2, 7) in possible_orbpairs[0][0]
    assert (4, 9) in possible_orbpairs[0][0]
    with pytest.raises(ValueError):
        next(test.generate_possible_orbpairs([0, 5, 2, 7]))


def answer_apig_h2_sto6g():
    """Find the APIG/STO-6G wavefunction by scanning for the coefficients with the lowest energy."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apig = APIG(2, 4)

    # FIXME: need normalize
    # plot all possible values (within normalization constraint)
    xs = []
    ys = []
    energies = []
    for i in np.arange(-2, 2, 0.1):
        for j in np.arange(-2, 2, 0.1):
            apig.assign_params(np.array([[i, j]]))
            # FIXME: need function to compute norm
            norm = sum(apig.get_overlap(sd) ** 2 for sd in (0b0101, 0b1010))
            if np.allclose(norm, 0):
                continue
            # FIXME: need to normalize
            xs.append(apig.params[0, 0])
            ys.append(apig.params[0, 1])
            # FIXME: need function to compute energy
            energy = sum(
                (ham.integrate_sd_wfn(sd, apig)) * apig.get_overlap(sd) for sd in (0b0101, 0b1010)
            )
            energy /= norm
            energies.append(energy)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ys, energies)
    print(sorted(zip(xs, ys, energies), key=lambda x: x[2])[:10])
    # ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    # ax2.scatter(xs, ys, energies)
    # plt.show()

    # print out results
    # print(sorted(zip(xs, ys, energies), key=lambda x: x[2])[0])
    # print(sorted(zip(xs, ys, energies), key=lambda x: x[2])[-1])

    # find exact minimum
    def min_energy(params):
        """Find minimum energy."""
        apig.assign_params(params.reshape(apig.params.shape))
        energy = sum(
            (ham.integrate_sd_wfn(sd, apig)) * apig.get_overlap(sd) for sd in (0b0101, 0b1010)
        )
        energy /= sum(apig.get_overlap(sd) ** 2 for sd in (0b0101, 0b1010))
        return energy

    res = scipy.optimize.minimize(min_energy, np.array([0.99388373467361912, -0.1104315260748444]))
    print("Minimum energy")
    print(res)

    # find exact maximum
    def max_energy(params):
        """Find maximum energy."""
        apig.assign_params(params.reshape(apig.params.shape))
        energy = sum(
            (ham.integrate_sd_wfn(sd, apig)) * apig.get_overlap(sd) for sd in (0b0101, 0b1010)
        )
        energy /= sum(apig.get_overlap(sd) ** 2 for sd in (0b0101, 0b1010))
        return -energy

    res = scipy.optimize.minimize(max_energy, np.array([0.11043152607484739, 0.99388373467361879]))
    print("Maximum energy")
    print(res)


def test_apig_h2_sto6g_ground():
    """Test ground state APIG wavefunction using H2 with HF/STO-6G orbital.

    Answers obtained from answer_apig_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APIG Energy : -1.8590898441488894
    APIG Coeffs : [0.99359749, -0.11300768]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apig = APIG(2, 4)

    # Solve system of equations
    objective = ProjectedSchrodinger(apig, ham, refwfn=[0b0101, 0b1010])
    results = least_squares(objective)
    assert np.allclose(results["energy"], -1.8590898441488894)


def test_apig_h2_sto6g_excited():
    """Test excited state APIG wavefunction using H2 with HF/STO-6G orbital.

    Answers obtained from answer_apig_h2_sto6g

    APIG Energy : -0.2416648697421632
    APIG Coeffs : [0.11300769, 0.99359749]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apig = APIG(2, 4, params=np.array([0.0, 1.0]))

    # Solve system of equations
    objective = ProjectedSchrodinger(apig, ham, refwfn=[0b0101, 0b1010])
    results = least_squares(objective)
    assert np.allclose(results["energy"], -0.2416648697421632)


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apig_h2_631gdp():
    """Find the ground state APIG/6-31G** wavefunction by scanning through the coefficients."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_631gdp_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apig = APIG(2, 20, params=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    objective = EnergyOneSideProjection(apig, ham)
    results = minimize(objective)
    print(results)
    return apig.params


def test_apig_h2_631gdp():
    """Test APIG wavefunction using H2 with HF/6-31G** orbitals.

    Answers obtained from answer_apig_h2_631gdp

    HF (Electronic) Energy : -1.84444667247
    APIG Energy : -1.8696828608304896
    APIG Coeffs : [0.995079200788, -0.059166892062, -0.054284175189, -0.036920061272,
                   -0.028848919079, -0.028847742282, -0.013108383833, -0.008485392433,
                   -0.008485285973, -0.005149411511]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_631gdp_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apig = APIG(2, 20)
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
    objective = ProjectedSchrodinger(apig, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], -1.8696828608304896)


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apig_lih_sto6g_slow():
    """Find the ground state APIG/STO-6G wavefunction for LiH by scanning for the lowest energy."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_lih_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apig = APIG(4, 12)

    objective = EnergyOneSideProjection(apig, ham)
    results = minimize(objective)
    print(results)
    return apig.params


def test_apig_lih_sto6g():
    """Test APIG wavefunction using LiH with HF/STO-6G orbital.

    Answers obtained from answer_apig_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    APIG Energy : -8.963531109581904
    APIG Coeffs : [1.033593181822e+00, 3.130903350751e-04, -4.321247538977e-03,
                   -1.767251395337e-03, -1.769214953534e-03, -1.169729179981e-03,
                   -5.327889357199e-01, 9.602580629349e-01, -1.139839360648e-02,
                   -2.858698370621e-02, -2.878270043699e-02, -1.129324573431e-01]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_lih_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    apig = APIG(4, 12)
    full_sds = [
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
    ]

    # Solve system of equations
    objective = ProjectedSchrodinger(apig, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], -8.963531109581904)
