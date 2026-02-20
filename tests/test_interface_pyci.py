import pytest
import numpy as np

from fanpy.interface.pyci import PYCI
from fanpy.wfn.base import BaseWavefunction
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.cc.standard_cc import StandardCC
from fanpy.tools.sd_list import sd_list
from fanpy.tools.performance import current_memory


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

class FakeWavefunction(BaseWavefunction):
    """ A fake wavefunction for testing purposes.
    """

    def __init__(self, nelec, nspin, params):
        """ Initialize FakeWavefunction.

        Args:
            nelec (int): Number of electrons.
            nspin (int): Number of spin orbitals.
        """
        self.assign_params(params)
        super().__init__(nelec, nspin)

    def get_overlap(self, sd, deriv=None):
        """ Get overlap with a Slater determinant.

        Args:
            sd (int): Slater determinant in integer representation. -> not used for fake implementation.
            deriv (np.ndarray, optional): If provided, compute the derivative of the overlap.

        Returns:
            float: Overlap value.
        """
        if deriv is not None:
            return np.zeros(len(deriv))
        else:
            return 1.0 
        
    def assign_params(self, params):
        """ Assign parameters to the wavefunction.

        Args:
            params (np.ndarray): Parameters to assign.
        """
        self.params = params


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
        self.ham = RestrictedMolecularHamiltonian(one_int=one_int, two_int=two_int)
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