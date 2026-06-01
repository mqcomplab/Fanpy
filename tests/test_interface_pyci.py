import pytest
import numpy as np
import pyci
from unittest.mock import patch 

from utils import find_datafile

from fanpy.interface.pyci import PYCI
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.constraints.norm import NormConstraint
from fanpy.eqn.constraints.energy import EnergyConstraint
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.cc.standard_cc import StandardCC
from fanpy.tools.sd_list import sd_list
from fanpy.tools.performance import current_memory
from interface_utils import FakeWavefunction, FakeHamiltonian


@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_norm_constraint_chunking(legacy_fanci):
    """Test norm constraint chunking in PYCI interface. We do this by comparing the jacobian of the norm constraint with and without chunking.
    """

    # todo: technically this test relies only on the classes in fanpy.interface.fanci. We do not necessarily need to use PYCI here.
    # However, utilizing PYCI gets rid of some of the setup code that would otherwise be necessary. Once we have unit testing for the interface,
    # we can move this test to that location.

    # set up fanpy wfn, ham, and objective
    test_wfn = StandardCC(4, 8)
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)
    test_ham = RestrictedMolecularHamiltonian(
        one_int, two_int
    )
    pspace = sd_list(4, 8, num_limit=None, exc_orders=[1, 2, 3, 4], spin=0)
    fanpy_objective = ProjectedSchrodinger(test_wfn, test_ham, energy_type="compute", pspace = pspace)

    # compute jacobian without chunking
    pyci_no_chunk = PYCI(fanpy_objective, 0.0, legacy_fanci=legacy_fanci)
    chunks = pyci_no_chunk.objective.calculate_overlap_deriv_chunks()
    if len(chunks) > 1: # chunks depend on memory available, so this check ensures that no chunking is happening
        raise RuntimeError("Test is invalid because chunking is occurring for the 'no chunking reference'.")
    jac_constraint = pyci_no_chunk.objective.make_norm_constraint()[1] # index 1 corresponds to jacobian, while 0 is constraint value
    x = np.random.rand(len(test_wfn.params) + 1)
    jac_constraint = jac_constraint(x)

    # compute jacobian with chunking
    max_mem = 4.5 * len(pspace) * 8 / (0.9 * 10**6) + current_memory() # creates chunks of length 4 (this equation depends on calculate_overlap_deriv_chunks)
    pyci_chunk = PYCI(fanpy_objective, 0.0, legacy_fanci=legacy_fanci, max_memory=max_mem)
    chunks = pyci_chunk.objective.calculate_overlap_deriv_chunks()
    # Note: since we determine max memory based on the equations in calculate_overlap_deriv_chunks, this should not be an issue.
    # This check ensures that chunks are generated as expected, even if there are changes to that method in the future.
    if len(chunks) < 2: 
        raise RuntimeError("Test is invalid because no chunking is occurring.")
    jac_constraint_chunk = pyci_chunk.objective.make_norm_constraint()[1] # index 1 corresponds to jacobian, while 0 is constraint value
    jac_constraint_chunk = jac_constraint_chunk(x)

    # compare jacobians
    assert np.allclose(jac_constraint, jac_constraint_chunk)


class PyCITestSetup:
    """ A test setup for PyCI interface. It sets up the Restricted Hamiltonian and a fake wavefunction.
    """
    def __init__(self):
        self.nelec = 2
        self.nspin = 4
        self.e_nuc = 0
        self.params = np.array([0.5, 0.5])
        self.wfn = FakeWavefunction(self.nelec, self.nspin, self.params)
        one_int = np.zeros((self.nelec, self.nelec)) # one electron integrals
        two_int = np.zeros((self.nelec, self.nelec, self.nelec, self.nelec)) # two electron integrals
        self.ham = FakeHamiltonian(one_int=one_int, two_int=two_int)
        self.eqn = ProjectedSchrodinger(self.wfn, self.ham) # default eqn setup

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_pyci_interface_nproj(legacy_fanci):
    """ Test whether nproj is correctly set in PyCI interface. For the default case.
    """
    setup_class = PyCITestSetup()

    interface = PYCI(setup_class.eqn, setup_class.e_nuc, legacy_fanci=legacy_fanci)

    # check nproj type
    assert isinstance(interface.nproj, int)

    # check nproj range: should be between 1 and FCI
    fci_pspace = sd_list(setup_class.nelec, setup_class.nspin, spin=0)
    assert 1 <= interface.nproj <= len(fci_pspace)

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_pspace_trimming(legacy_fanci):
    """ Test whether spin unrestricted pspace is trimmed to spin restricted in PyCI interface.
    """
    setup_data = PyCITestSetup()

    # set up Objective with unrestricted pspace
    pspace_unrestr = sd_list(setup_data.nelec, setup_data.nspin) 
    # spin unrestricted FCI space not supported in PyCI -> interface should trim it to spin restricted
    eqn = ProjectedSchrodinger(setup_data.wfn, setup_data.ham, pspace=pspace_unrestr)

    pspace_restr = sd_list(setup_data.nelec, setup_data.nspin, spin=0)

    # interface setup
    interface = PYCI(eqn, setup_data.e_nuc, legacy_fanci=legacy_fanci)
    assert interface.nproj == len(pspace_restr)

    # check that pspace wfn is pyci 
    # FakeWavefunction has seniority = None
    assert isinstance(interface.pspace_wfn, pyci.fullci_wfn)

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_mask(legacy_fanci):
    """ Test whether mask is correctly set in PyCI interface.
    """
    setup_data = PyCITestSetup()

    interface = PYCI(setup_data.eqn, setup_data.e_nuc, legacy_fanci=legacy_fanci)

    assert interface.mask.shape == (interface.nparam,) # consistent with number of parameters
    nparams = len(setup_data.wfn.params) + 1 # wfn params + energy
    assert interface.mask.shape == (nparams,) 

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_jac(legacy_fanci):
    """ Basic test to see if jac can be computed and has the correct dimensions.
    """
    setup_data = PyCITestSetup()
    interface = PYCI(setup_data.eqn, setup_data.e_nuc, legacy_fanci=legacy_fanci)

    nparams = len(setup_data.wfn.params) + 1 # wfn params + energy
    jac_params = np.ones(interface.nparam) 
    jac = interface.objective.compute_jacobian(jac_params)

    assert jac.shape == (interface.nproj+1, nparams) # we need to add one for the normalization condition

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_objective(legacy_fanci):
    """ Test shape of objective computation."""
    setup_data = PyCITestSetup()
    eqn = ProjectedSchrodinger(setup_data.wfn, setup_data.ham)

    interface = PYCI(eqn, setup_data.e_nuc, legacy_fanci=legacy_fanci)
    obj_params = np.ones(interface.nparam) 
    objective = interface.objective.compute_objective(obj_params)

    assert objective.shape == (interface.nproj+1,) # we need to add one for the normalization condition

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_energy(legacy_fanci):
    """ Basic test to see if energy can be computed and appears in results dictionary."""
    setup_data = PyCITestSetup()
    eqn = ProjectedSchrodinger(setup_data.wfn, setup_data.ham)

    interface = PYCI(eqn, setup_data.e_nuc, legacy_fanci=legacy_fanci)
    x0 = np.ones(interface.nparam)
    results = interface.objective.optimize(x0=x0) # dictionary with energy and other info

    # check if energy keyword is present
    assert 'energy' in results

    # check if energy is a float
    assert isinstance(results['energy'], float)

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_integration(legacy_fanci):
    """ Test if interface with real hamiltonian and wavefunction can compute jacobian, objective, and optimize without returning NaN or inf values. This is a basic sanity check to ensure that the interface is working as expected with real data.
    """
    test_wfn = StandardCC(2, 4)
    one_int = np.load(find_datafile("data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_sto6g_twoint.npy"))
    test_ham = RestrictedMolecularHamiltonian(
        one_int, two_int
    )
    pspace = sd_list(2, 4, num_limit=None, exc_orders=[1, 2, 3, 4], spin=0)
    fanpy_objective = ProjectedSchrodinger(test_wfn, test_ham, energy_type="compute", pspace = pspace)

    interface = PYCI(fanpy_objective, 0.0, legacy_fanci=legacy_fanci)

    # compute jacobian 
    jac = interface.objective.compute_jacobian(np.ones(interface.nparam))
    # check for NaN or inf values in jacobian
    assert not np.any(np.isnan(jac))
    assert not np.any(np.isinf(jac))

    # compute objective
    obj = interface.objective.compute_objective(np.ones(interface.nparam))
    # check for NaN or inf values in objective
    assert not np.any(np.isnan(obj))
    assert not np.any(np.isinf(obj))

    # optimize 
    results = interface.objective.optimize(x0=np.ones(interface.nparam))
    assert results["success"]
    assert not np.isnan(results['energy'])
    assert not np.isinf(results['energy'])
    assert not np.any(np.isnan(results['x']))
    assert not np.any(np.isinf(results['x']))

@pytest.mark.parametrize("legacy_fanci", [True, False])
def test_behavior_regression_small_system(legacy_fanci):
    test_wfn = StandardCC(2, 4)
    one_int = np.load(find_datafile("data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_sto6g_twoint.npy"))
    test_ham = RestrictedMolecularHamiltonian(
        one_int, two_int
    )
    pspace = sd_list(2, 4, num_limit=None, exc_orders=[1, 2, 3, 4], spin=0)
    fanpy_objective = ProjectedSchrodinger(test_wfn, test_ham, energy_type="compute", pspace = pspace)

    interface = PYCI(fanpy_objective, 0.0, legacy_fanci=legacy_fanci)

    x0 = np.ones(interface.nparam)
    # the expected values are base on the output of the integration test. If the integration test is passing, then these expected values should be correct. If there are changes to the interface that affect the objective or jacobian computation, then these expected values may need to be updated.
    expected_obj = np.array([-3.0200443 , -1.89131832, -1.89131832,  1.4439305 ,  3.0])
    expected_jac = np.array([[ 1.81610047e-01, -1.81610047e-01, -1.81610047e-01,
         1.81610047e-01, -1.81610047e-01, -1.00000000e+00],
       [-2.07292837e+00, -4.75554844e-16, -4.75554844e-16,
         1.81610047e-01, -4.75554844e-16, -1.00000000e+00],
       [ 1.81610047e-01, -4.75554844e-16, -4.75554844e-16,
        -2.07292837e+00, -4.75554844e-16, -1.00000000e+00],
       [-1.26232046e+00,  1.26232046e+00,  1.26232046e+00,
        -1.26232046e+00,  1.26232046e+00,  1.00000000e+00],
       [ 0.00000000e+00,  2.00000000e+00,  2.00000000e+00,
         0.00000000e+00,  2.00000000e+00,  0.00000000e+00]])
    
    obj = interface.objective.compute_objective(x0)
    jac = interface.objective.compute_jacobian(x0)
    assert np.allclose(obj, expected_obj)
    assert np.allclose(jac, expected_jac)

    # make sure initial x0 is not already optimal or a weird point. 
    # using expected objective, since we have already verified that the objective is computed correctly 
    initial_cost = np.linalg.norm(expected_obj) # the objective is the residual value of the projected schrodinger equation 
    results = interface.objective.optimize(x0=x0)
    assert results["cost"] < initial_cost # optimization should reduce cost
    assert not np.allclose(results['energy'], expected_obj[-1], atol=10**-3) # energy should be different from initial objective value


def test_projected_check():
    """make sure we cannot initialize PYCI class with an objective that is not the projected schrodinger equation"""
    setup_data = PyCITestSetup()
    objective = EnergyOneSideProjection(setup_data.wfn, setup_data.ham)
    with pytest.raises(TypeError):
        PYCI(objective, 0.0)
    
def test_fill_seniority():
    from fanpy.wfn.geminal.apig import APIG
    sen_o_wfn = APIG(4, 8)
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)
    test_ham = RestrictedMolecularHamiltonian(
        one_int, two_int
    )
    pspace = sd_list(4, 8, num_limit=None, exc_orders=[1, 2], spin=0, seniority=0)
    fanpy_objective = ProjectedSchrodinger(sen_o_wfn, test_ham, energy_type="compute", pspace = pspace)
    interface = PYCI(fanpy_objective, 0.0)
    assert interface.nproj == len(pspace)
    assert isinstance(interface.pspace_wfn, pyci.doci_wfn)

def test_update_objective_fanpy_ham():
    setup_data = PyCITestSetup() # ham params initialized to zeros
    fanpy_obj = ProjectedSchrodinger(setup_data.wfn , setup_data.ham)
    pyci_obj = PYCI(fanpy_obj, 0.0)

    one_int =  np.random.rand(2, 2)
    two_int = np.random.rand(2, 2, 2, 2)
    new_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    pyci_obj.update_objective(new_ham)

    # check if Fanpy ham got updated
    assert np.allclose(pyci_obj.fanpy_ham.one_int, one_int)
    assert np.allclose(pyci_obj.fanpy_ham.two_int, two_int)

    # check if PyCI ham got updated
    assert np.allclose(pyci_obj.pyci_ham.one_mo, one_int)
    assert np.allclose(pyci_obj.pyci_ham.two_mo, two_int)

def test_update_objective_pyci_ham():
    setup_data = PyCITestSetup() # ham params initialized to zeros
    fanpy_obj = ProjectedSchrodinger(setup_data.wfn, setup_data.ham)
    pyci_obj = PYCI(fanpy_obj, 0.0)

    one_int =  np.random.rand(2, 2)
    two_int = np.random.rand(2, 2, 2, 2)
    new_ham = pyci.hamiltonian(0.0, one_int, two_int)
    pyci_obj.update_objective(new_ham)

    # check if Fanpy ham got updated
    assert np.allclose(pyci_obj.fanpy_ham.one_int, one_int)
    assert np.allclose(pyci_obj.fanpy_ham.two_int, two_int)

    # check if PyCI ham got updated
    assert np.allclose(pyci_obj.pyci_ham.one_mo, one_int)
    assert np.allclose(pyci_obj.pyci_ham.two_mo, two_int)

def test_pyci_ham_setter():
    setup_data = PyCITestSetup() # ham params initialized to zeros
    fanpy_obj = ProjectedSchrodinger(setup_data.wfn, setup_data.ham)
    pyci_obj = PYCI(fanpy_obj, 0.0)

    # setting it with pyci ham
    one_int =  np.random.rand(2, 2)
    two_int = np.random.rand(2, 2, 2, 2)
    new_ham = pyci.hamiltonian(0.0, one_int, two_int)
    pyci_obj.pyci_ham = new_ham
    assert np.allclose(pyci_obj.pyci_ham.one_mo, one_int)
    assert np.allclose(pyci_obj.pyci_ham.two_mo, two_int)
    assert np.allclose(pyci_obj.fanpy_ham.one_int, one_int)
    assert np.allclose(pyci_obj.fanpy_ham.two_int, two_int)

    # setting it with fanpy ham
    one_int =  np.ones((2, 2))
    two_int = np.ones((2, 2, 2, 2))
    new_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    pyci_obj.pyci_ham = new_ham
    assert np.allclose(pyci_obj.pyci_ham.one_mo, one_int)
    assert np.allclose(pyci_obj.pyci_ham.two_mo, two_int)
    assert np.allclose(pyci_obj.fanpy_ham.one_int, one_int)
    assert np.allclose(pyci_obj.fanpy_ham.two_int, two_int)

def test_fanpy_ham_setter():
    setup_data = PyCITestSetup() # ham params initialized to zeros
    fanpy_obj = ProjectedSchrodinger(setup_data.wfn, setup_data.ham)
    pyci_obj = PYCI(fanpy_obj, 0.0)

    # setting it with pyci ham
    one_int =  np.random.rand(2, 2)
    two_int = np.random.rand(2, 2, 2, 2)
    new_ham = pyci.hamiltonian(0.0, one_int, two_int)
    pyci_obj.fanpy_ham = new_ham
    assert np.allclose(pyci_obj.pyci_ham.one_mo, one_int)
    assert np.allclose(pyci_obj.pyci_ham.two_mo, two_int)
    assert np.allclose(pyci_obj.fanpy_ham.one_int, one_int)
    assert np.allclose(pyci_obj.fanpy_ham.two_int, two_int)

    # setting it with fanpy ham
    one_int =  np.ones((2, 2))
    two_int = np.ones((2, 2, 2, 2))
    new_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    pyci_obj.fanpy_ham = new_ham
    assert np.allclose(pyci_obj.pyci_ham.one_mo, one_int)
    assert np.allclose(pyci_obj.pyci_ham.two_mo, two_int)
    assert np.allclose(pyci_obj.fanpy_ham.one_int, one_int)
    assert np.allclose(pyci_obj.fanpy_ham.two_int, two_int)

def test_ham_setter_type_check():
    setup_data = PyCITestSetup() # ham params initialized to zeros
    fanpy_obj = ProjectedSchrodinger(setup_data.wfn, setup_data.ham)
    pyci_obj = PYCI(fanpy_obj, 0.0)

    with pytest.raises(TypeError):
        pyci_obj.fanpy_ham = "not a Hamiltonian"
    with pytest.raises(TypeError):
        pyci_obj.pyci_ham = "not a Hamiltonian"

def test_constraints_init():
    """ Check if constraints are set up properly"""
    setup_data = PyCITestSetup()
    norm_const = NormConstraint(setup_data.wfn)
    e_const = EnergyConstraint(setup_data.wfn, setup_data.ham)
    fanpy_obj = ProjectedSchrodinger(setup_data.wfn, setup_data.ham, constraints=[norm_const, e_const])
    pyci_obj = PYCI(fanpy_obj, 0.0)
    n_pyci_consts = len(pyci_obj.objective.constraints)
    assert n_pyci_consts == 2
    # check for compute objective 
    with patch.object(EnergyConstraint, "objective", return_value = 3.08 ) as mock_method:
        x = np.random.rand(pyci_obj.objective.nactive)
        res = pyci_obj.objective.compute_objective(x)
        
        # check if we call the method at least once
        # we do not check how many times we call the objective method,
        # as it depends on pyci
        assert mock_method.call_count > 0 

        # make sure we get expected return value
        assert res[-1] == 3.08 # last element is the energy constraint

        # check if we passed the expected elements of x
        adapted_x = x[:-1]
        assert np.allclose(mock_method.call_args[0][0], adapted_x)

def test_ham_update():
    """ Make sure the hamiltonian gets updated in constraints that have a ham attribute."""
    setup_data = PyCITestSetup()
    norm_const = NormConstraint(setup_data.wfn)
    e_const = EnergyConstraint(setup_data.wfn, setup_data.ham)
    fanpy_obj = ProjectedSchrodinger(setup_data.wfn, setup_data.ham, constraints=[norm_const, e_const])
    pyci_obj = PYCI(fanpy_obj, 0.0)

    norb = setup_data.ham.one_int.shape[0]
    one_int = np.random.rand(norb, norb)
    two_int = np.random.rand(norb, norb, norb, norb)
    new_ham = FakeHamiltonian(one_int, two_int)
    pyci_obj.update_objective(new_ham)
    assert len(pyci_obj.fanpy_objective.constraints) == 2
    # NOTE: this assumes that energy constraint is the second in the list.
    # this is because we set up the constraints to be [norm, e] for the fanpy obj in this test case
    new_energy_const = pyci_obj.fanpy_objective.constraints[1] 
    assert type(new_energy_const.ham) == type(new_ham)
    assert np.allclose(new_energy_const.ham.one_int, new_ham.one_int)
    assert np.allclose(new_energy_const.ham.two_int, new_ham.two_int)