""" Tests olsens.geminals.geminal

"""

#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np
from inspect import ismethod
from copy import deepcopy as copy
from itertools import combinations
from newton import newton
from geminal import APIG, AP1roG, quasinewton, lstsq
from slater_det import excite_pairs, excite_orbs
from horton_wrapper import from_horton

def check_if_exception_raised(func, exception):
    """ Passes if given exception is raised

    Parameter
    ---------
    func : function object
        Function that contains the desired test code
    exception : Exception
        Exception that is raised

    Returns
    -------
    bool
        True if Exception is raised
        False if Exception is not raised
    """
    try:
        func()
    except exception:
        return True
    except:
        return False
    else:
        return False

def test_init():
    """ Check if initialization with bad values raises error
    """
    # check first sets of init
    def f():
        gem = APIG(1., 4, [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 4., [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 4, 0b0011)
    assert check_if_exception_raised(f, AssertionError)
    # check setters
    def f():
        gem = APIG(5, 4, [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 2, [0b0111])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(3, 6, [0b0101])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 2, [0b110000])
    assert check_if_exception_raised(f, AssertionError)

def test_setters_getters():
    """ Check if setters and getters are working properly
    """
    gem = APIG(1, 4, [0b00011])
    # check npairs
    assert gem.npairs == 1
    assert gem.npairs == gem._npairs
    # check npairs setter
    gem.npairs = 2
    assert gem.npairs == 2
    def f():
        gem.npairs = 5
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.npairs = 0
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.npairs = 0.5
    assert check_if_exception_raised(f, AssertionError)
    # check nelec
    gem.npairs = 1
    assert gem.nelec == 2
    gem.npairs = 2
    assert gem.nelec == 4
    # check norbs
    assert gem.norbs == 4
    assert gem.norbs == gem._norbs
    # check norbs setter
    gem.norbs = 7
    assert gem.norbs == 7
    gem.npairs = 5
    def f():
        gem.norbs = 4
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.norbs = 9.0
    assert check_if_exception_raised(f, AssertionError)
    # check pspace
    assert gem.pspace == tuple([0b00011])
    assert gem.pspace == tuple([3])
    assert gem.pspace == gem._pspace
    # check pspace setter
    gem.npairs = 2
    gem.norbs = 4
    gem.pspace = tuple([0b0001111])
    assert gem.pspace == tuple([0b001111])
    gem.pspace = tuple([0b001111])
    assert gem.pspace == tuple([0b001111])
    def f():
        gem.pspace = tuple([0b0])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.pspace = tuple([0b01])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.pspace = tuple([0b1100000011])
    assert check_if_exception_raised(f, AssertionError)
    gem.npairs = 3
    gem.norbs = 6
    def f():
        gem.pspace = tuple([0b10111])
    assert check_if_exception_raised(f, AssertionError)

def test_generate_pspace():
    """ test APIG.generate_pspace
    """
    def f():
        gem.generate_pspace()
    # 99 occupied, No virtuals (Not enough SD generated)
    gem = APIG(99, 99, [int('11'*99, 2)])
    assert check_if_exception_raised(f, AssertionError)
    # 1 occupied, 1 virtual (Enough SD with single excitation)
    gem = APIG(1, 2, [0b11])
    assert gem.generate_pspace() == (0b0011, 0b1100, 0b0101)
    # 1 occupied, 99 virtuals (Enough SD with single excitation)
    gem = APIG(1, 99, [0b11])
    pspace = [0b11 << i*2 for i in range(99)] + [0b0101]
    assert gem.generate_pspace() == tuple(pspace)
    # 2 occupied, 2 virtuals (Enough SD with single excitation)
    gem = APIG(2, 4, [0b1111])
    assert gem.generate_pspace() == (0b1111, 0b110011, 0b11000011, 0b00111100,
                                     0b11001100, 0b11110000,
                                     0b010111, 0b01000111, 0b011101)
    # 2 occupied, 3 virtuals (Enough SD with single excitation)
    gem = APIG(2, 5, [0b1111])
    #print([bin(i) for i in gem.generate_pspace()])
    assert gem.generate_pspace() == (0b1111, 0b110011, 0b11000011, 0b1100000011,
                                     0b111100, 0b11001100, 0b1100001100,
                                     0b11110000, 0b1100110000,
                                     0b1111000000,
                                     0b010111)
    # 2 occupied, 4 virtuals
    gem = APIG(2, 6, [0b1111])
    pspace = gem.generate_pspace()
    assert len(pspace) == 13
    #  check that each of the generated slated determinant is in the total space
    all_sds = (sum(0b11 << i*2 for i in pairs)
               for pairs in combinations(range(gem.norbs), gem.npairs))
    for i in pspace:
        assert i in all_sds
    #  hand checking
    assert pspace == (0b1111,
                      0b110011, 0b11000011, 0b1100000011, 0b110000000011,
                      0b111100, 0b11001100, 0b1100001100, 0b110000001100,
                      0b11110000, 0b1100110000, 0b110000110000,
                      0b1111000000)
    #  check __init__
    gem2 = APIG(2, 6)
    assert gem.generate_pspace() == gem2.generate_pspace()
    # 3 occupied, 3 virtuals
    gem = APIG(3, 6, [0b111111])
    pspace = gem.generate_pspace()
    assert len(pspace) == 19
    #  check that each of the generated slated determinant is in the total space
    all_sds = [sum(0b11 << i*2 for i in pairs)
               for pairs in combinations(range(gem.norbs), gem.npairs)]
    for i in pspace:
        assert i in all_sds

def test_permanent():
    """ test permanent
    """
    # zero matrix
    matrix = np.zeros((6,6))
    assert APIG.permanent(matrix) == 0
    # identity matrix
    matrix = np.identity(6)
    assert APIG.permanent(matrix) == 1
    # one matrix
    matrix = np.zeros((6,6)) + 1
    assert APIG.permanent(matrix) == np.math.factorial(6)
    # random matrix
    matrix = np.arange(1, 10).reshape((3,3))
    assert APIG.permanent(matrix) == 450

def test_permanent_derivative():
    """ test partial derivative of permanent wrt one of its elements
    """
    # zero matrix
    matrix = np.zeros((6,6))
    assert APIG.permanent_derivative(matrix,5,0) == 0
    # identity matrix
    matrix = np.identity(6)
    assert APIG.permanent_derivative(matrix,3,2) == 0
    assert APIG.permanent_derivative(matrix,3,3) == 1
    # one matrix
    matrix = np.zeros((6,6)) + 1
    assert APIG.permanent_derivative(matrix,5,5) == 2
    # random matrix
    matrix = np.arange(1, 10).reshape((3,3))
    assert APIG.permanent_derivative(matrix,1,2) == 22
    #
    # APIG permanent_derivative in APIG.overlap
    #
    apig = APIG(4,7)
    C = np.random.rand(4,7)
    # coefficient not in included column
    assert apig.overlap(0b111100110011, C, derivative=True, indices=(2,6)) == 0
    # coefficient in included column
    assert 1.0 > apig.overlap(0b111100110011, C, derivative=True, indices=(2,4)) > 0.0
    #
    # AP1roG permanent_derivative in AP1roG.overlap
    #
    ap1rog = AP1roG(3,9)
    C = np.random.rand(3,9)
    C[:,:3] = np.eye(3)
    # no excitation
    assert ap1rog.overlap(0b111111, C, derivative=True, indices=(2,2)) == 1
    assert ap1rog.overlap(0b111111, C, derivative=True, indices=(1,2)) == 0
    # single excitation
    assert ap1rog.overlap(0b11001111, C, derivative=True, indices=(2,3)) == 1
    assert ap1rog.overlap(0b11001111, C, derivative=True, indices=(3,2)) == 0


def test_overlap():
    """ Tests APIG.overlap
    """
    gem = APIG(3, 6)
    coeff = np.random.rand(3, 6)
    # Bad Slater determinant
    sd = None
    assert gem.overlap(sd, coeff) == 0
    # Slater determinant with different number of electrons
    sd = 0b11
    assert gem.overlap(sd, coeff) == 0
    # Ground state SD
    sd = 0b111111
    assert gem.overlap(sd, coeff) == gem.permanent(coeff[:, :3])
    # Excited SD
    sd = 0b110011001100
    assert gem.overlap(sd, coeff) == gem.permanent(coeff[:, [1, 3, 5]])

def test_double_phi_H_psi():
    """ Tests, APIG.double_phi_H_psi
    """
    gem = APIG(2, 4)
    sd = 0b1111
    coeff = np.eye(2,4)
    one_non = np.random.rand(4,4)
    one = one_non + one_non.T
    two_non = np.random.rand(4,4,4,4)
    two = two_non + np.einsum('jilk', two_non)
    two = two + np.einsum('klij', two)
    # Non Hermitian one
    f = lambda: gem.double_phi_H_psi(sd, coeff, one_non, two)
    assert check_if_exception_raised(f, AssertionError)
    # Non Hermitian two
    f = lambda: gem.double_phi_H_psi(sd, coeff, one, two_non)
    assert check_if_exception_raised(f, AssertionError)
    # Wavefunction is HF ground state
    # Projecting onto HF ground state
    coeff = np.eye(2,4)
    '''
        H = \sum_i (2*h_{ii} + V_{i\bar{i}i\bar{i}}) +
            \sum_{ij} (4*V_{ijij} - 2*V_{ijji} +
            \sum_{ia} V_{i\bar{i}a\bar{a}}
    '''
    # h_{ii} + h_{\bar{i}\bar{i}}
    integral_one = 2*(one[0,0] + one[1,1])
    # V_{i\bar{i}i\bar{i}
    integral_two = two[0,0,0,0] + two[1,1,1,1]
    # 4*V_{ijij} - 2*V_{ijji}
    integral_two += 4*two[0,1,0,1] - 2*two[0,1,1,0]
    # V_{i\bar{i}a\bar{a}}
    # none because excitations of gs sd has zero overlap with this wavefunction
    assert np.allclose(gem.double_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)
    # Wavefunction is a combination of HF ground, Excited states (to 2nd orbital)
    # Projecting onto HF ground state
    coeff[:, 2] = 1
    integral_two += two[0,0,2,2]+two[1,1,2,2]
    assert np.allclose(gem.double_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)
    # Wavefunction is a combination of HF ground, Excited states (to 2nd and 3rd orbitals)
    # Projecting onto HF ground state
    coeff[:, 3] = 1
    integral_two += two[0,0,3,3]+two[1,1,3,3]
    assert np.allclose(gem.double_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)

def test_brute_phi_H_psi():
    # Same test as test_double_phi_H_psi for pair occupied slater determinant
    gem = APIG(2, 4)
    sd = 0b1111
    one = np.random.rand(4,4)
    two = np.random.rand(4,4,4,4)
    one_non = np.random.rand(4,4)
    one = one_non + one_non.T
    two_non = np.random.rand(4,4,4,4)
    two = two_non + np.einsum('jilk', two_non)
    two = two + np.einsum('klij', two)
    # Wavefunction is HF ground state
    # Projecting onto HF ground state
    coeff = np.eye(2,4)
    '''
        H = \sum_i (2*h_{ii} + V_{i\bar{i}i\bar{i}}) +
            \sum_{ij} (4*V_{ijij} - 2*V_{ijji} +
            \sum_{ia} V_{i\bar{i}a\bar{a}}
    '''
    #  h_{ii} + h_{\bar{i}\bar{i}}
    integral_one = 2*(one[0,0] + one[1,1])
    #  V_{i\bar{i}i\bar{i}
    integral_two = two[0,0,0,0] + two[1,1,1,1]
    #  4*V_{ijij} - 2*V_{ijji}
    integral_two += 4*two[0,1,0,1] - 2*two[0,1,1,0]
    #  V_{i\bar{i}a\bar{a}}
    #  none because excitations of gs sd has zero overlap with this wavefunction
    assert np.allclose(gem.brute_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)
    #  Second test
    coeff[:, 2] = 1
    integral_two += two[0,0,2,2] + two[1,1,2,2]
    #print(integral_one, integral_two)
    assert np.allclose(gem.brute_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)
    #  Third test
    coeff[:, 3] = 1
    integral_two += two[0,0,3,3] + two[1,1,3,3]
    assert np.allclose(gem.brute_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)
    # Non pair occupied slater determinant (beta to beta excitation)
    sd = 0b100111
    # Wavefunction is a combination of HF ground
    # Projecting onto single excitation of HF ground
    coeff = np.eye(2,4)
    ''' \sum_ij <00100111|h_ij a_i^\dagger a_j|00001111>
         = <00001111| h_\bar{21} |00001111>
        h_ij = 0 if i and k do not have the same spin
    '''
    integral_one = one[2,1]
    ''' \sum_ijkl <00100111|g_ijkl a_i^\dagger a_k^\dagger a_j a_l|00001111>
         = <00001111| g_0\bar{2}0\bar{1} + g_\bar{0201} + g_1\bar{2}1\bar{1} |00001111>
         = <00001111| V_0\bar{2}0\bar{1} + V_\bar{0201} + V_1\bar{2}1\bar{1} - V_\bar{0210}|00001111> -
       V_ijkl = 0 if i and k do not have the same spin and
                  if j and l do not have the same spin
    '''
    integral_two = two[0,2,0,1] + two[0,2,0,1] + two[1,2,1,1] - two[0,2,1,0]
    assert np.allclose(gem.brute_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)
    # Wavefunction is a combination of HF ground, Excited states (to 2nd orbital)
    # Projecting onto single excitation of HF ground
    coeff[:, 2] = 1
    ''' \sum_ij <00100111|h_ij a_i^\dagger a_j|00110011>
         = <00100111| h_12 |00110011>
        \sum_ijkl <00100111|g_ijkl a_i^\dagger a_k^\dagger a_j a_l|00110011>
         = <00110011| g_0102 + g_\bar{0}1\bar{0}2 + g_1\bar{2}2\bar{2} |00110011>
         = <00110011| V_0102 + V_\bar{0}1\bar{0}2 + V_1\bar{2}2\bar{2} - V_0120 |00110011>
    '''
    integral_one += one[1,2]
    integral_two += two[0,1,0,2] + two[0,1,0,2] + two[1,2,2,2] - two[0,1,2,0]
    ''' \sum_ij <00100111|h_ij a_i^\dagger a_j|00111100> = 0
        \sum_ijkl <00100111|g_ijkl a_i^\dagger a_k^\dagger a_j a_l|00111100>
         = <00111100| g_0\bar{01}2 |00111100>
         = <00110011| V_0\bar{01}2 - V_0\bar{0}2\bar{1} |00110011>
         = -<00110011| V_0\bar{0}2\bar{1} |00110011>
    '''
    integral_two -= two[0,0,2,1]
    assert np.allclose(gem.brute_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)
    # Wavefunction is a combination of HF ground, Excited states (to 2nd and 3rd orbitals)
    # Projecting onto single excitation of HF ground
    coeff[:, 3] = 1
    ''' \sum_ij <00100111|h_ij a_i^\dagger a_j|11000011> = 0
        \sum_ijkl <00100111|g_ijkl a_i^\dagger a_k^\dagger a_j a_l|11000011>
         = <11000011| g_1\bar{2}3\bar{3} |11000011>
         = <11000011| V_1\bar{2}3\bar{3} |11000011>
    '''
    integral_two += two[1,2,3,3]
    ''' \sum_ij <00100111|h_ij a_i^\dagger a_j|11001100> =0
        \sum_ijkl <00100111|g_ijkl a_i^\dagger a_k^\dagger a_j a_l|11001100> = 0
    '''
    ''' \sum_ij <00100111|h_ij a_i^\dagger a_j|11110000> = 0
        \sum_ijkl <00100111|g_ijkl a_i^\dagger a_k^\dagger a_j a_l|11110000> = 0
    '''
    assert np.allclose(gem.brute_phi_H_psi(sd, coeff, one, two), integral_one + integral_two)

def test_jacobian():
    """ Tests, APIG.jacobian()
    """
    # Test that it works
    npairs = 3
    norbs = 9
    gem = APIG(npairs, norbs)
    coeffs = np.ones((npairs, norbs))
    coeffs[:,:npairs] += np.eye(npairs)
    one = np.ones((norbs, norbs))
    two = np.ones((norbs, norbs, norbs, norbs))
    fun = gem.nonlin(coeffs, one, two, gem.pspace)
    jac = gem.nonlin_jac(coeffs, one, two, gem.pspace)
    assert jac.shape == (fun.size, coeffs.size)
    # Test that overlap gets restored as the proper instancemethod after nonlin_jac()
    # is called
    assert ismethod(gem.overlap)

def test_APIG_quasinewton():
    """
    """
    # Define user input
    fn = 'test/h4.xyz'
    basis = '3-21g'
    nocc = 2
    # Make geminal guess from HORTON's AP1roG module
    ht_out = from_horton(fn=fn, basis=basis, nocc=nocc, guess='apig')
    basis  = ht_out['basis']
    core = ht_out['ham'][2]
    one = ht_out['ham'][0]
    two = ht_out['ham'][1]
    energy_old = ht_out['energy']
    coeffs_old = ht_out['coeffs'].ravel()
    coeffs_old += 2*(0.001*np.random.rand(coeffs_old.size) - 0.0005)
    coeffs = copy(coeffs_old)
    options = { 'options': { 'maxfev': 250*coeffs.size,
                             'xatol': 1.0e-9,
                             'fatol': 1.0e-9,
                           },
                'jac' : True,
                'method' : 'hybr',
              }
    # Optimize with quasinewton method
    gem = APIG(nocc, basis.nbasis)
    result_qn = gem(coeffs, one, two, solver=quasinewton, **options)
    assert result_qn['success']
    energy = gem.phi_H_psi(gem.ground, gem.coeffs, one, two) + core
    olp = gem.overlap(gem.ground, gem.coeffs)
    """
    str0 = "GUESS\n{}\n{}\n".format(coeffs_old, energy_old)
    str1 = "GEMINAL\n{}\n{}\n".format(gem.coeffs, energy)
    str2 = "OVERLAP: {}".format(olp)
    print(str0 + str1 + str2)
    """


test_init()
test_setters_getters()
test_generate_pspace()
test_permanent()
test_permanent_derivative()
test_overlap()
test_double_phi_H_psi()
test_brute_phi_H_psi()
test_jacobian()
test_APIG_quasinewton()


# vim: set textwidth=90 :