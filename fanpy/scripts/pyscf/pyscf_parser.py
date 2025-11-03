import argparse 
import json

parser = argparse.ArgumentParser()
parser.add_argument("--geom", 
                    type=json.loads, 
                    required=True, 
                    help="Geometry of the molecule for pyscf. E.g. `[['H', [0, 0, 0]], ['H', [0, 0, 0.74]]]`")

parser.add_argument("--wfn_type", 
                    type=str, 
                    required=True, 
                    help="Type of wavefunction to use. E.g. 'cisd', 'fci', 'doci', 'ap1rog', 'apig'")

parser.add_argument("--basis", 
                    type=str, 
                    required=True, 
                    help="Basis set to use. E.g. 'sto6g'")

parser.add_argument("--hf_units", 
                    type=str, 
                    default="Angstrom", 
                    help="Units for the geometry in the Hartree-Fock calculation. Default is 'angstrom'.")

parser.add_argument("--optimize_orbs", 
                    type=bool, 
                    default=False, 
                    help="Whether to optimize orbitals in the Hartree-Fock calculation.")

parser.add_argument("--pspace_exc",  
                    type=list, 
                    default=(1, 2), 
                    help="Excitations to include in the pspace. E.g. '(1, 2)'")

parser.add_argument("--objective", 
                    type=str, 
                    default='projected', 
                    help="Objective function to use. E.g. 'least_squares', 'variational', 'one_energy', 'projected'")

parser.add_argument("--solver", 
                    type=str, 
                    default='least_squares', 
                    help="Solver to use. E.g. 'minimize', 'least_squares', 'root'")

parser.add_argument("--wfn_kwargs", 
                    type=str, 
                    default="", 
                    help="Additional keyword arguments for the wavefunction.")

parser.add_argument("--solver_kwargs", 
                    type=str, 
                    default="", 
                    help="Additional keyword arguments for the solver.")

parser.add_argument("--ham_noise", 
                    type=float, 
                    default=0.0, 
                    help="Amount of noise to add to the Hamiltonian matrix elements.")

parser.add_argument("--wfn_noise", 
                    type=float, 
                    default=0.0, 
                    help="Amount of noise to add to the wavefunction parameters.")

parser.add_argument("--load_orbs", 
                    type=str, 
                    default=None, 
                    help="Numpy file of the orbital transformation matrix that will be applied to the initial Hamiltonian")

parser.add_argument("--load_ham", 
                    type=str, 
                    default=None, 
                    help="Numpy file of the Hamiltonian matrix to load directly.")

parser.add_argument("--load_ham_um", 
                    type=str, 
                    default=None, 
                    help="Numpy file of the Hamiltonian parameters that will overwrite the unitary matrix of the initial Hamiltonian.")

parser.add_argument("--load_wfn", 
                    type=str, 
                    default=None, 
                    help="Numpy file of the wavefunction parameters to load directly.")

parser.add_argument("--load_chk", 
                    type=str, 
                    default=None, 
                    help="Checkpoint file to load the wavefunction and Hamiltonian parameters from.")

parser.add_argument("--save_chk", 
                    type=str, 
                    default="", 
                    help="Checkpoint file to save the wavefunction and Hamiltonian parameters to.")

parser.add_argument("--memory", 
                    type=str, 
                    default=None, 
                    help="Amount of memory to allocate.")

parser.add_argument("--constraint", 
                    type=str, 
                    default=None, 
                    help="Constraint to apply during optimization.")

parser.add_argument("--fanpt_kwargs", 
                    type=str,
                    default="",
                    help="Additional keyword arguments for FANPT.")