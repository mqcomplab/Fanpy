# PySCF Calculation
import time
from pyscf import gto, scf
import numpy as np
from fanpy.wfn.ci.cisd import CISD
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools.sd_list import sd_list
import fanpy.interface as interface


# Record the start time
start_time = time.time()

energy = {}

print('# PySCF calculation for alpha = 0.0...')
mol = gto.M(atom = [['Be', ['0.0', '0.0', '0.0']], ['H', ['0.0', '-2.54', '0.0']], ['H', ['0.0', '2.54', '0.0']]],
            unit = 'B',
            basis = 'sto-6g')

myhf = scf.HF(mol)
myhf.kernel()

# PySCF interface
pyscf_interface = interface.pyscf.PYSCF(myhf)

print("\n>> >> Fanpy")

   
# Number of electrons
print("Number of Electrons: {}".format(pyscf_interface.nelec))

# Number of spin orbitals
print("Number of Spin Orbitals: {}".format(pyscf_interface.nspinorb))

# Nuclear-nuclear repulsion
print("Nuclear-nuclear repulsion: {}".format(pyscf_interface.energy_nuc))

# Fanpy calculation
print('# Fanpy calculation for alpha = 0.0...')

# Initialize wavefunction
wfn = CISD(pyscf_interface.nelec, pyscf_interface.nspinorb)
wfn.assign_params(wfn.params + 0.0001 * 2 * (np.random.rand(*wfn.params.shape) - 0.5))
print('Wavefunction: CISD')
print(f"Number of parameters: {wfn.nparams}")

# Initialize Hamiltonian
ham = RestrictedMolecularHamiltonian(pyscf_interface.one_int, pyscf_interface.two_int, update_prev_params=True)
print('Hamiltonian: RestrictedMolecularHamiltonian')

# Projection space
exc_orders = [1, 2, 3, 4, 5, 6]
pspace = sd_list(pyscf_interface.nelec, pyscf_interface.nspinorb, num_limit=None, exc_orders=exc_orders, spin=0)
nproj = len(pspace)
print("Projection space (orders of excitations): ", exc_orders)
print("Number of projections (CISD): {:}".format(nproj))

print("\n# New interface!")
# Initialize objective
from fanpy.eqn.projected import ProjectedSchrodinger

objective = ProjectedSchrodinger(wfn, ham, energy_type="compute", pspace=pspace)

# FANPT calc
from fanpy.fanpt import FANPT

fanpt = FANPT(objective, pyscf_interface.energy_nuc, 
        energy_active=False, final_order=1,
              steps = 10, legacy_fanci=False)

results = fanpt.optimize(
    guess_params = wfn.params,
    guess_energy = pyscf_interface.energy_elec,
    mode='lstsq', 
    use_jac=True, 
    xtol=1.0e-8,
    ftol=1.0e-8, 
    gtol=1.0e-8,
    max_nfev=wfn.nparams,
    verbose=2)
# Record the end time
end_time = time.time()

# Calculate the total time taken
execution_time = end_time - start_time

print(f'Total execution time: {execution_time} seconds')
print("### Final energy: {:}".format(results["energy"] + pyscf_interface.energy_nuc))

