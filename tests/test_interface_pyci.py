import pytest
import numpy as np

from fanpy.interface.pyci import PYCI
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
