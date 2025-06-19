import numpy as np
import os
import sys
import pyci
from fanpy.wfn.utils import convert_to_fanci
from fanpy.wfn.cc.standard_cc import StandardCC
from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.composite.embedding_fixedelectron import FixedEmbeddedWavefunction
import fanpy.tools.slater as slater
from fanpy.tools.sd_list import sd_list
from scipy.special import comb
from fanci.fanpt_wrapper import reduce_to_fock, solve_fanpt


# Number of electrons
nelec = 16
print('Number of Electrons: {}'.format(nelec))

# One-electron integrals
one_int_file = '/blue/rmirandaquintana/kimt1/beh2_h2o/beh2_r3.0/sto-6g/hf_boys/oneint.npy'
one_int = np.load(one_int_file)
print('One-Electron Integrals: {}'.format(os.path.abspath(one_int_file)))

# Two-electron integrals
two_int_file = '/blue/rmirandaquintana/kimt1/beh2_h2o/beh2_r3.0/sto-6g/hf_boys/twoint.npy'
two_int = np.load(two_int_file)
print('Two-Electron Integrals: {}'.format(os.path.abspath(two_int_file)))

# Number of spin orbitals
nspatial = one_int.shape[0]
nspin = nspatial * 2
print('Number of Spin Orbitals: {}'.format(nspin))

# Nuclear-nuclear repulsion
nuc_nuc = 2.918242709541748
print('Nuclear-nuclear repulsion: {}'.format(nuc_nuc))



system_inds = [0, 0, 0, 1, 1, 1]

ao_inds = np.load('/blue/rmirandaquintana/kimt1/beh2_h2o/beh2_r3.0/sto-6g/hf_boys/ao_inds.npy')
indices_list = [[] for _ in range(max(system_inds) + 1)]
for i, ao_ind in enumerate(ao_inds):
    # for other localizations
    indices_list[system_inds[ao_ind]].append(i)
    indices_list[system_inds[ao_ind]].append(slater.spatial_to_spin_indices(i, nspin // 2, to_beta=True))

    # for svd
    #indices_list[ao_ind].append(i)
    #indices_list[ao_ind].append(slater.spatial_to_spin_indices(i, nspin // 2, to_beta=True))
indices_list = [sorted(i) for i in indices_list]
print(indices_list)
wfn_list = []

# number of electrons
npairs = nelec // 2
# hard coding orbital structure
nelec1 = len([i for i in indices_list[0] if i < npairs or nspatial <= i < nspatial + npairs])
nelec2 = len([i for i in indices_list[1] if i < npairs or nspatial <= i < nspatial + npairs])
print('Number of Electrons in System 1: {}'.format(nelec1))
print('Number of Electrons in System 2: {}'.format(nelec2))

# system 1: beh2
# Initialize wavefunction
wfn1 = StandardCC(nelec1, len(indices_list[0]), params=None, memory='6gb', ranks=[1, 2], indices=None, refwfn=None,
                 exop_combinations=None, refresh_exops=50000)
#wfn.assign_params(wfn.params + 0.0001 * 2 * (np.random.rand(*wfn.params.shape) - 0.5))
wfn_list.append(wfn1)
print('Wavefunction in System 1: CCSD')


# system 2: h2o
# Initialize wavefunction
wfn2 = AP1roG(nelec2, len(indices_list[1]), params=None, memory='6gb')
#wfn.assign_params(wfn.params + 0.0001 * 2 * (np.random.rand(*wfn.params.shape) - 0.5))
print('Wavefunction in System 2: AP1roG')
wfn_list.append(wfn2)



# Initialize wavefunction
wfn = FixedEmbeddedWavefunction(nelec, [6, 10], nspin, indices_list, wfn_list, memory=None, disjoint=True)
print('Wavefunction: Embedded CCSD and AP1roG')

# Initialize Hamiltonian
ham1 = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
ham0 = RestrictedMolecularHamiltonian(one_int, reduce_to_fock(two_int), update_prev_params=True) 
print('Hamiltonian: RestrictedMolecularHamiltonian')

# Projection space
print('Projection space by excitation')
fill = 'excitation'
nproj = sum(i.nparams for i in wfn_list) * 2 + 1
print(nproj)

# Select parameters that will be optimized
param_selection = [(i, np.ones(i.nparams, dtype=bool)) for i in wfn_list]

# Initialize objective
pyci_ham1 = pyci.hamiltonian(nuc_nuc, ham1.one_int, ham1.two_int)
pyci_ham0 = pyci.hamiltonian(nuc_nuc, ham0.one_int, ham0.two_int)
fanci_wfn = convert_to_fanci(wfn, pyci_ham0, seniority=wfn.seniority, param_selection=param_selection, nproj=nproj, objective_type='projected', norm_det=[(0, 1)])
fanci_wfn.tmpfile = 'checkpoint.npy'
fanci_wfn.step_print = True

# Set energies
integrals = np.zeros(fanci_wfn._nproj, dtype=pyci.c_double)
olps = fanci_wfn.compute_overlap(fanci_wfn.active_params, 'S')[:fanci_wfn._nproj]
fanci_wfn._ci_op(olps, out=integrals)
energy_val = np.sum(integrals * olps) / np.sum(olps ** 2)
print('Initial energy:', energy_val)

# Solve
print('Optimizing wavefunction: solver')
#fanci_results = fanci_wfn.optimize(100, np.hstack([fanci_wfn.active_params, energy_val]),
#                                              mode='lstsq', use_jac=True, xtol=1.0e-15,
#                                              ftol=1.0e-15, gtol=1.0e-15,
#                                              max_nfev=1000*fanci_wfn.nactive)
#fanci_results = fanci_wfn.optimize(np.hstack([fanci_wfn.active_params, energy_val]),
#                                              mode='lstsq', use_jac=True)
fanci_results = solve_fanpt(
        fanci_wfn, pyci_ham0, pyci_ham1, np.hstack([fanci_wfn.active_params, energy_val]), fill=fill,
        energy_active=True, resum=False, ref_sd=0, final_order=1, lambda_i=0.0, lambda_f=1.0, steps=50,
        solver_kwargs={'mode':'lstsq', 'use_jac':True, 'xtol':1.0e-8, 'ftol':1.0e-8, 'gtol':1.0e-5, 'max_nfev':1000*fanci_wfn.nactive, 'verbose':2, 'vtol':1e-5})

results = {}
results['success'] = fanci_results.success
results['params'] = fanci_results.x
results['message'] = fanci_results.message
results['internal'] = fanci_results
results['energy'] = fanci_results.x[-1]

# Results
if results['success']:
    print('Optimization was successful')
else:
    print('Optimization was not successful: {}'.format(results['message']))
print('Final Electronic Energy: {}'.format(results['energy']))
print('Final Total Energy: {}'.format(results['energy'] + nuc_nuc))
