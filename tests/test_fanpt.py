import numpy as np
import pyci
from fanpy.wfn.utils import convert_to_fanci
from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from scipy.special import comb
import scipy.linalg
import pytest
from utils import find_datafile
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.wfn.cc.standard_cc import StandardCC
from fanpy.wfn.utils import convert_to_fanci
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools.sd_list import sd_list
from scipy.special import comb
from fanpy.wfn.cc.ap1rog_generalized_NEW import AP1roGSDGeneralized
from fanpy.wfn.cc.standard_cc import StandardCC
import fanpy.interface as interface
import fanpy.interface as interface
# Initialize Hamiltonian
def reduce_to_fock(two_int):
    """Reduce given two electron integrals to that of the correspoding Fock operator.

    Parameters
    ----------
    two_int : np.ndarray(K, K, K, K)
        Two electron integrals of restricted orbitals.

    """
    fock_two_int = np.zeros(two_int.shape)
    nspatial = two_int.shape[0]
    indices = np.arange(nspatial)
    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]]
    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]]
    #fock_two_int[indices[None, :, None], indices[:, None, None], indices[None, None, :], indices[None, :, None]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]]
    #fock_two_int[indices[None, :, None], indices[:, None, None], indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]]
    
    #occ_indices = np.arange(nelec // 2)
    #fock_two_int[indices[:, None, None], occ_indices[None, :, None], indices[None, None, :], occ_indices[None, :, None]] =  two_int[indices[:, None, None], occ_indices[None, :, None], indices[None, None, :], occ_indices[None, :, None]]
    #fock_two_int[indices[:, None, None], occ_indices[None, :, None], occ_indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], occ_indices[None, :, None], occ_indices[None, :, None], indices[None, None, :]]
    #fock_two_int[occ_indices[None, :, None], indices[:, None, None], indices[None, None, :], occ_indices[None, :, None]] =  two_int[indices[:, None, None], occ_indices[None, :, None], indices[None, None, :], occ_indices[None, :, None]]
    #fock_two_int[occ_indices[None, :, None], indices[:, None, None], occ_indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], occ_indices[None, :, None], occ_indices[None, :, None], indices[None, None, :]]

    return fock_two_int


#@pytest.mark.skip(reason="This test fails and is being worked on (Issue 23).")
def test_fock_energy():
    """Test that Fock operator and Hamiltonian operator gives same energy for ground state HF."""
    nelec = 6
    one_int_file = find_datafile("data/data_beh2_r3.0_hf_sto6g_oneint.npy")
    one_int = np.load(one_int_file)
    two_int_file = find_datafile('data/data_beh2_r3.0_hf_sto6g_twoint.npy')
    two_int = np.load(two_int_file)
    nspin = one_int.shape[0] * 2

    wfn = AP1roG(nelec, nspin, params=None, memory='6gb')
    nproj = 35 
    orig = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
    pspace = sd_list(nelec, nspin, num_limit=None, exc_orders=[1, 2, 3, 4, 5, 6], spin=0)
    objective = ProjectedSchrodinger(wfn, orig, energy_type="compute", pspace=pspace)
    pyci_interface = interface.pyci.PYCI(objective, 2.9182427095417482, max_memory=64000, legacy_fanci=False)
    pyci_interface.objective_type = "projected"
    pyci_interface.build_pyci_objective()
    integrals_orig = np.zeros(nproj, dtype=pyci.c_double)
    olps_orig = pyci_interface.objective.compute_overlap(objective.active_params, 'S')[:nproj]
    pyci_interface.objective._ci_op(olps_orig, out=integrals_orig)
    energy_val_orig = np.sum(integrals_orig * olps_orig) / np.sum(olps_orig ** 2)

    fock_two_int = reduce_to_fock(two_int)
    fock = RestrictedMolecularHamiltonian(one_int, fock_two_int, update_prev_params=True)
    objective_fock = ProjectedSchrodinger(wfn, fock, energy_type="compute", pspace=pspace)
    pyci_interface_fock = interface.pyci.PYCI(objective_fock, 2.9182427095417482, max_memory=64000, legacy_fanci=False)
    pyci_interface_fock.objective_type = "projected"
    pyci_interface_fock.build_pyci_objective()
    integrals_fock = np.zeros(nproj, dtype=pyci.c_double)
    olps_fock = pyci_interface_fock.objective.compute_overlap(objective_fock.active_params, 'S')[:nproj]
    pyci_interface_fock.objective._ci_op(olps_fock, out=integrals_fock)
    energy_val_fock = np.sum(integrals_fock * olps_fock) / np.sum(olps_fock ** 2)

    assert np.allclose(energy_val_orig, energy_val_fock)

#@pytest.mark.skip(reason="This test fails and is being worked on (Issue 23).")
def test_fock_objective():
    """Test that Fock operator with HF ground state satisfies projected Schrodinger equation."""
    nelec = 6
    one_int_file = find_datafile('data/data_beh2_r3.0_hf_sto6g_oneint.npy')
    one_int = np.load(one_int_file)
    two_int_file = find_datafile('data/data_beh2_r3.0_hf_sto6g_twoint.npy')
    two_int = np.load(two_int_file)
    nspin = one_int.shape[0] * 2

    wfn = AP1roG(nelec, nspin, params=None, memory='6gb')
    nproj = int(comb(nspin // 2, nelec - nelec // 2) * comb(nspin // 2, nelec // 2))
    nproj=35
    fock_two_int = reduce_to_fock(two_int)
    fock = RestrictedMolecularHamiltonian(one_int, fock_two_int, update_prev_params=True)
    pspace = sd_list(nelec, nspin, num_limit=None, exc_orders=[1, 2, 3, 4, 5, 6], spin=0)
    objective_fock = ProjectedSchrodinger(wfn, fock, energy_type="compute", pspace=pspace)
    pyci_interface_fock = interface.pyci.PYCI(objective_fock, 2.9182427095417482, max_memory=64000, legacy_fanci=False)
    pyci_interface_fock.objective_type = "projected"
    pyci_interface_fock.build_pyci_objective()
    integrals_fock = np.zeros(nproj, dtype=pyci.c_double)
    olps_fock = pyci_interface_fock.objective.compute_overlap(objective_fock.active_params, 'S')[:nproj]
    pyci_interface_fock.objective._ci_op(olps_fock, out=integrals_fock)
    energy_val_fock = np.sum(integrals_fock * olps_fock) / np.sum(olps_fock ** 2)

    assert np.allclose(np.sum(np.abs(pyci_interface_fock.objective.compute_objective(np.hstack([objective_fock.active_params, energy_val_fock])))), 0)

############################################################
# check orbital rotation invariance
#@pytest.mark.skip(reason="This test fails and is being worked on (Issue 23).")
def test_fock_rotation():
    """Test that Fock operator invariance to orbital rotation."""
    nelec = 6
    one_int_file = find_datafile('data/data_beh2_r3.0_hf_sto6g_oneint.npy')
    one_int = np.load(one_int_file)
    two_int_file = find_datafile('data/data_beh2_r3.0_hf_sto6g_twoint.npy')
    two_int = np.load(two_int_file)
    nspin = one_int.shape[0] * 2
    #original before orb rot
    wfn = AP1roG(nelec, nspin, params=None, memory='6gb')
    nproj = 35 
    orig = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
    pspace = sd_list(nelec, nspin, num_limit=None, exc_orders=[1, 2, 3, 4, 5, 6], spin=0)
    objective = ProjectedSchrodinger(wfn, orig, energy_type="compute", pspace=pspace)
    pyci_interface = interface.pyci.PYCI(objective, 2.9182427095417482, max_memory=64000, legacy_fanci=False)
    pyci_interface.objective_type = "projected"
    pyci_interface.build_pyci_objective()
    integrals_orig = np.zeros(nproj, dtype=pyci.c_double)
    olps_orig = pyci_interface.objective.compute_overlap(objective.active_params, 'S')[:nproj]
    pyci_interface.objective._ci_op(olps_orig, out=integrals_orig)
    energy_val_orig = np.sum(integrals_orig * olps_orig) / np.sum(olps_orig ** 2)

    
    #fanci_wfn_orig = convert_to_fanci(wfn, pyci_ham_orig, seniority=wfn.seniority, param_selection=None, nproj=nproj, objective_type='projected')
    #integrals_orig = np.zeros(fanci_wfn_orig.nproj, dtype=pyci.c_double)
    #olps_orig = pyci_interface_fock.objective.compute_overlap(objective_fock.active_params, 'S')[:nproj]
    #pyci_interface_fock.objective._ci_op(olps_orig, out=integrals_orig)
    #energy_val_orig = np.sum(integrals_orig * olps_orig) / np.sum(olps_orig ** 2)

    # random orbital rotation of occupied
    _, _, v = np.linalg.svd(np.random.rand(nelec // 2, nelec // 2))
    # random orbital rotation of virtual
    _, _, v2 = np.linalg.svd(np.random.rand((nspin-nelec) // 2, (nspin-nelec) // 2))
    v = scipy.linalg.block_diag(v, v2)

    #_, _, v = np.linalg.svd(np.random.rand(nspatial, nspatial))

    # rotate integrals
    one_int = v.T.dot(one_int).dot(v)
    two_int = np.einsum('ijkl,ia->ajkl', two_int, v)
    two_int = np.einsum('ajkl,jb->abkl', two_int, v)
    two_int = np.einsum('abkl,kc->abcl', two_int, v)
    two_int = np.einsum('abcl,ld->abcd', two_int, v)

    # check that fock and hamiltonian gives same energy for initial state hf ground state
    orbrot = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
#    pyci_ham_orbrot = pyci.hamiltonian(0, orbrot.one_int, orbrot.two_int)
    pspace = sd_list(nelec, nspin, num_limit=None, exc_orders=[1, 2, 3, 4, 5, 6], spin=0)
    objective_orbrot = ProjectedSchrodinger(wfn, orbrot, energy_type="compute", pspace=pspace)
    pyci_interface_orbrot = interface.pyci.PYCI(objective_orbrot, 2.9182427095417482, max_memory=64000, legacy_fanci=False)
    pyci_interface_orbrot.objective_type = "projected"
    pyci_interface_orbrot.build_pyci_objective()
    integrals_orbrot = np.zeros(nproj, dtype=pyci.c_double)
    olps_orbrot = pyci_interface_orbrot.objective.compute_overlap(objective_orbrot.active_params, 'S')[:nproj]
    pyci_interface_orbrot.objective._ci_op(olps_orbrot, out=integrals_orbrot)
    energy_val_orbrot = np.sum(integrals_orbrot * olps_orbrot) / np.sum(olps_orbrot ** 2)

    nproj=35
    # original before orbital rotation
    fock = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
    pspace = sd_list(nelec, nspin, num_limit=None, exc_orders=[1, 2, 3, 4, 5, 6], spin=0)
    objective_fock = ProjectedSchrodinger(wfn, fock, energy_type="compute", pspace=pspace)
    pyci_interface_fock = interface.pyci.PYCI(objective_fock, 2.9182427095417482, max_memory=64000, legacy_fanci=False)
    pyci_interface_fock.objective_type = "projected"
    pyci_interface_fock.build_pyci_objective()
    integrals_fock = np.zeros(nproj, dtype=pyci.c_double)
    olps_fock = pyci_interface_fock.objective.compute_overlap(objective_fock.active_params, 'S')[:nproj]
    pyci_interface_fock.objective._ci_op(olps_fock, out=integrals_fock)
    energy_val_fock = np.sum(integrals_fock * olps_fock) / np.sum(olps_fock ** 2)




    assert np.allclose(energy_val_orig, energy_val_fock)
    assert np.allclose(energy_val_orbrot, energy_val_fock)
#    #check that objective values are all zero for fock operator
#    assert np.allclose(np.sum(np.abs(pyci_interface_fock.objective.compute_objective(np.hstack([objective_fock.active_params, energy_val_fock])))), 0)

