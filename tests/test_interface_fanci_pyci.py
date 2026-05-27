""" Test for fanpy.interface.fanci.pyci """

import numpy as np
import pytest
from unittest.mock import patch 

from fanpy.interface.fanci.pyci import ProjectedSchrodingerPyCI
import pyci

from interface_utils import FakeHamiltonian, FakeWavefunction, FakeSchrodinger, FakeCC

############## Tools for testing purposes #########################

def make_test_instance(**overrides):
    """make test instance of ProjectedSchrodingerPyCI with fake fanpy objective and fake pyci hamiltonian and wavefunction
    This helps set up a class that requires a lot of parameters.
    """
    # build Fake fanpy objective
    wfn = FakeWavefunction(2, 4, np.ones(4))
    nocc = wfn.nelec // 2
    ham = FakeHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    obj = FakeSchrodinger(wfn, ham)

    # build fake pyci hamiltonian
    energy_nuc = 0.0
    pyci_ham = pyci.hamiltonian(energy_nuc, ham.one_int, ham.two_int)

    # build pyci wavefunction 
    # use FCI pspace wavefunction
    pyci_wfn = pyci.fullci_wfn(pyci_ham.nbasis, wfn.nelec - nocc, nocc)

    defaults = {
        "fanpy_objective" : obj,
        "ham" : pyci_ham,
        "wfn" : pyci_wfn,
        "nocc" : 2,
        "seniority": wfn.seniority,
        "nproj": 1,
        "fill": "excitation",
        "mask": np.ones(wfn.params.shape[0]+1, dtype=bool),
        "constraints": {},
        "param_selection": obj.indices_component_params,
        "norm_param": None,
        "norm_det": None,
        "max_memory": 8000,
        "step_print": False,
        "step_save": False,
        "tmpfile": ""
    }
    defaults.update(overrides)
    return ProjectedSchrodingerPyCI(**defaults)

################# init tests ###################################
def test_init():
    # errors
    ham = "non_pyci_hamiltonian"
    with pytest.raises(TypeError):
        make_test_instance(ham=ham)
    fanpy_ham = FakeHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    with pytest.raises(TypeError):
        make_test_instance(fanpy_ham=fanpy_ham)

    # normalization constraints
    # default is normalization constraint. 
    # Note: the constraint attribute is just the keys of the constraints dictionary. This is a PyCI feature, not a Fanpy feature. 
    constraints = None
    norm_det  = None
    pyci_obj = make_test_instance(constraints=constraints, norm_det=norm_det)
    assert type(pyci_obj.constraints) == tuple
    assert "<\\Phi|\\Psi> - 1>" in pyci_obj.constraints

################# compute overlap tests ###################################

def test_compute_overlap():
    pyci_obj = make_test_instance()

    # compute overlap between the pyci wavefunction and a random vector
    # occ indices are the p-space:
    overlap = pyci_obj.compute_overlap(np.random.rand(4), "P")
    olp_size = len(pyci_obj.pspace) # note p-space contains occ indices
    assert len(overlap) == olp_size
    assert np.allclose(overlap, np.ones(olp_size))

    # compute overlap between the pyci wavefunction and a random vector
    # occ indices are the s-space:
    overlap = pyci_obj.compute_overlap(np.random.rand(4), "S")
    olp_size = len(pyci_obj.sspace) # note s-space contains occ indices
    assert len(overlap) == len(pyci_obj.sspace)
    assert np.allclose(overlap, np.ones(olp_size))

    # compute overlap between the pyci wavefunction and a random vector
    occ_indices = np.asarray([[0, 1]]) # use DOCI occs representation
    overlap = pyci_obj.compute_overlap(np.random.rand(4), occ_indices)
    olp_size = len(occ_indices) 
    assert overlap.shape == (olp_size,)
    assert np.allclose(overlap, np.ones(olp_size))

    # compute overlap between the pyci wavefunction and a random vector
    occ_indices = np.asarray([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]) # use FCI occs representation, with two sd dets
    overlap = pyci_obj.compute_overlap(np.random.rand(4), occ_indices)
    olp_size = len(occ_indices) 
    assert overlap.shape == (olp_size,)
    assert np.allclose(overlap, np.ones(olp_size))

def test_compute_overlap_type_check():
    pyci_obj = make_test_instance()
    with pytest.raises(ValueError):
        pyci_obj.compute_overlap(np.array([[0, 1]]), "not_a_vector")


################# compute overlap deriv tests ###################################

def test_compute_overlap_deriv():
    pyci_obj = make_test_instance()
    # compute overlap derivatives between the pyci wavefunction and a random vector
    overlap_deriv = pyci_obj.compute_overlap_deriv(np.random.rand(4), "P")
    assert overlap_deriv.shape == (len(pyci_obj.pspace), pyci_obj.nactive - pyci_obj.mask[-1])
    assert np.allclose(overlap_deriv, np.zeros(overlap_deriv.shape))

    # compute overlap derivatives between the pyci wavefunction and a random vector
    overlap_deriv = pyci_obj.compute_overlap_deriv(np.random.rand(4), "S")
    assert overlap_deriv.shape == (len(pyci_obj.sspace), pyci_obj.nactive - pyci_obj.mask[-1])
    assert np.allclose(overlap_deriv, np.zeros(overlap_deriv.shape))

    # compute overlap derivatives between the pyci wavefunction and a random vector
    occs_array = np.asarray([[0, 1]])
    overlap_deriv = pyci_obj.compute_overlap_deriv(np.random.rand(4), occs_array=occs_array)
    assert overlap_deriv.shape == (len(occs_array), pyci_obj.nactive - pyci_obj.mask[-1])
    assert np.allclose(overlap_deriv, np.zeros(overlap_deriv.shape))

def test_compute_overlap_deriv_type_check():
    pyci_obj = make_test_instance()
    with pytest.raises(ValueError):
        pyci_obj.compute_overlap_deriv(np.array([[0, 1]]), "not_a_vector")

################# compute overlap double derivtests ###################################

def test_compute_overlap_double_deriv_errors():
    pyci_obj = make_test_instance()
    with pytest.raises(ValueError):
        pyci_obj.compute_overlap_double_deriv(np.random.rand(4), "not_a_vector")
    # double deriv is only implemented for CC as of now. 
    with pytest.raises(NotImplementedError):
        pyci_obj.compute_overlap_double_deriv(np.random.rand(4), "P")

def test_compute_overlap_double_deriv():

    # build python objective with CC wfn
    wfn = FakeCC(nelec=2, nspin=4)
    ham = FakeHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    obj = FakeSchrodinger(wfn, ham)
    pyci_obj = make_test_instance(fanpy_objective=obj)

    # pspace double derivatives
    double_deriv = pyci_obj.compute_overlap_double_deriv(np.random.rand(4), "P")
    assert double_deriv.shape == (len(pyci_obj.pspace), wfn.nparams, wfn.nparams)
    assert np.allclose(double_deriv, np.ones((len(pyci_obj.pspace), wfn.nparams, wfn.nparams)))

    # sspace double derivatives
    double_deriv = pyci_obj.compute_overlap_double_deriv(np.random.rand(4), "S")
    assert double_deriv.shape == (len(pyci_obj.sspace), wfn.nparams, wfn.nparams)
    assert np.allclose(double_deriv, np.ones((len(pyci_obj.sspace), wfn.nparams, wfn.nparams)))

    # FCI occs vector double derivatives
    occ_indices = np.asarray([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]) # use FCI occs representation, with two sd dets
    double_deriv = pyci_obj.compute_overlap_double_deriv(np.random.rand(4), occs_array=occ_indices)
    assert double_deriv.shape == (len(occ_indices), wfn.nparams, wfn.nparams)
    assert np.allclose(double_deriv, np.ones((len(occ_indices), wfn.nparams, wfn.nparams)))

################# compute objective tests ###################################

def test_compute_objective():
    pyci_obj = make_test_instance()

    params = np.random.rand(pyci_obj.nactive)
    mock_result = np.ones(pyci_obj.nactive) * 42.0

    # patch the compute objective method from PyCI to return a fixed value. 
    # we just need to check if we call the correct method with the parameters
    # the rest is up to PyCI
    with patch.object(pyci.fanci.FanCI, "compute_objective", return_value=mock_result) as mock_method:
        result = pyci_obj.compute_objective(params)
        mock_method.assert_called_once_with(params)
    
    assert np.allclose(result, mock_result)


################# compute jacobian tests ###################################

def test_compute_jacobian():
    pyci_obj = make_test_instance()
    params = np.random.rand(pyci_obj.nactive)
    mock_result = np.ones((pyci_obj.nactive, pyci_obj.nproj)) * 42.0 # this is likely the wrong dimension. Just checking here that 2D arrays work. 

    # patch the compute jacobian method from PyCI to return a fixed value (mock result). 
    # we just need to check if we call the correct method with the parameters
    # the rest is up to PyCI
    with patch.object(pyci.fanci.FanCI, "compute_jacobian", return_value=mock_result) as mock_method:
        result = pyci_obj.compute_jacobian(params)
        mock_method.assert_called_once_with(params)
    
    assert np.allclose(result, mock_result)

# the last index of the mask corresponds to the energy, so we check both cases where E is active or inactive. 
@pytest.mark.parametrize("mask", [np.asarray([1, 0, 1, 0, 0], dtype=bool), np.asarray([1, 0, 1, 0, 1], dtype=bool)])
def test_compute_masked_jacobian(mask):
    pyci_obj = make_test_instance() # wfn has four params

    pyci_obj_masked = make_test_instance(mask=mask)
    x = np.random.rand(5)
    jac = pyci_obj.compute_jacobian(x)
    print("DEBUG >>> normal jac shape: ", jac.shape)
    masked_jac = pyci_obj_masked.compute_jacobian(x)
    cols = [i for i in range(len(mask)) if mask[i]]
    jac_sliced = jac[:, cols]
    print(jac)
    assert masked_jac.shape[1] == sum(mask)
    assert np.allclose(jac_sliced, masked_jac) # todo this is checking a ton, as olp deriv is 0. Only energy is -1. Having a check similar to this with an actual H might be useful 


################# optimize tests ###################################

# NOTE: none of these tests check if we get sensible results, the point here is to check that we can run optimizers successfully with different modes and parameter masks

@pytest.mark.parametrize("mask", [np.ones(5, dtype=int), np.asarray([1, 0, 1, 0, 0], dtype=bool), np.asarray([1, 0, 1, 0, 1], dtype=bool)])
def test_optimize_lstsq(mask):
    """ Check if optimize method runs without errors and energy is one of the keys"""
    pyci_obj = make_test_instance(mask=mask)
    initial_guess = np.random.rand(pyci_obj.fanpy_wfn.nparams+1)
    results = pyci_obj.optimize(initial_guess, mode="lstsq")
    assert "energy" in results.keys()

def test_optimize_root():
    """ Check if optimize method runs without errors and energy is one of the keys"""
    # set up objective with less wfn params. We cannot generate 5 or more projections
    # changing the FakeWavefunction in setup step is more complicated with hardcoded 4 params. 
    wfn = FakeWavefunction(2, 4, np.ones(3))
    ham = FakeHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    obj = FakeSchrodinger(wfn, ham)
    mask = np.ones(wfn.nparams + 1, dtype=bool) # determines nactive in PyCI object
    param_sel = obj.indices_component_params # parameter selection based on fanpy objective

    pyci_obj = make_test_instance(fanpy_objective=obj, nproj=wfn.nparams + 1, mask=mask, param_selection=param_sel)

    # initial guess
    x0 = np.random.rand(pyci_obj.nactive)

    results = pyci_obj.optimize(x0, mode='root')

    assert "energy" in results.keys()

def test_optimize_errors():
    pyci_obj = make_test_instance()
    initial_guess = np.random.rand(pyci_obj.nactive)
    with pytest.raises(ValueError):
        pyci_obj.optimize(initial_guess, "not a mode")
    with pytest.raises(ValueError): # default is an overdetermined system
        pyci_obj.optimize(initial_guess, "root")

def test_optimize_norm_const():
    pyci_obj = make_test_instance(constraints=None)

    initial_guess = np.random.rand(pyci_obj.nactive)
    results = pyci_obj.optimize(initial_guess, mode='lstsq')
    assert "energy" in results.keys()

################# optimize stochastic tests ###################################

# todo: this test fails because stochastic optimizer has bugs in it. It has not been updated to the new interface class, so the re-initialization does not work. Additionally, there are other bugs in the code as well. 
@pytest.mark.xfail()
def test_optimize_stochasitc_lstsq(mask):
    """ Check if optimize method runs without errors and energy is one of the keys"""
    mask = np.ones(5, dtype=int)
    pyci_obj = make_test_instance(mask=mask)
    initial_guess = np.random.rand(pyci_obj.fanpy_wfn.nparams+1)
    results = pyci_obj.optimize_stochastic(nsamp=3, x0=initial_guess, mode="lstsq", fill = pyci_obj.fill)
    assert "energy" in results.keys()

################# utility methods tests ###################################

@pytest.mark.parametrize("freeze_idx", [[2], [1, 3, 4], [0, 1, 2, 3, 4]])
def test_freeze_parameters(freeze_idx):
    pyci_obj = make_test_instance()
    mask = np.ones(pyci_obj.nparam, dtype=bool)
    freeze_idx = [2]
    pyci_obj.freeze_parameter(freeze_idx)
    # manually update mask
    mask[freeze_idx] = np.zeros(len(freeze_idx), dtype=bool)
    assert np.all(np.equal(mask, pyci_obj.mask))

    # unfreeze parameters:
    pyci_obj.unfreeze_parameter(freeze_idx)
    assert np.all(np.equal(np.ones(pyci_obj.nparam, dtype=bool), pyci_obj.mask)) # all parameters are unfrozen --> mask is all 1