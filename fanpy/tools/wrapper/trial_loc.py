from functools import reduce
import numpy as np
import scipy.linalg
import os
import re
from pyscf import gto 
from pyscf import scf 
from pyscf import __config__
from pyscf.lo.iao import reference_mol
from fanpy.tools.wrapper.pyscf import convert_gbs_nwchem


xyz_file = "./system.xyz"
basis_file = "../basis.gbs"
system_inds = [0, 1, 2]
nelec = 6

# check xyz file
cwd = os.path.dirname(__file__)
if os.path.isfile(os.path.join(cwd, xyz_file)):
    xyz_file = os.path.join(cwd, xyz_file)
elif not os.path.isfile(xyz_file):  # pragma: no branch
    raise ValueError("Given xyz_file does not exist")

# get coordinates
with open(xyz_file, "r") as f:
    lines = [i.strip() for i in f.readlines()[2:]]
    atoms = ";".join(lines)

# get mol
if os.path.splitext(basis_file)[1] == ".gbs":
    basis = convert_gbs_nwchem(basis_file)
else:
    basis = basis_file
mol = gto.M(
    atom=atoms, basis={i: j for i, j in basis.items() if i + ' ' in atoms}, parse_arg=False,
    unit="bohr"
)
    
# get hf
hf = scf.RHF(mol)
# run hf
hf.scf()
mo_coeff = hf.mo_coeff


MINAO = getattr(__config__, 'lo_iao_minao', 'minao')

pmol = reference_mol(mol, MINAO)
orig_s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)

minao_labels = pmol.ao_labels()
minao_inds = np.array([system_inds[int(re.search(r'^(\d+)\s+', label).group(1))] for label in minao_labels])
print(minao_labels)
print(minao_inds)

coeff_mo_lo = []
for sub_mo_coeff in [mo_coeff[:, hf.mo_occ > 0], mo_coeff[:, hf.mo_occ == 0]]:
    s12 = sub_mo_coeff.T.dot(orig_s12)
    system_s12 = []
    for i in range(max(system_inds) + 1):
        system_s12.append(s12[:, minao_inds == i])

    system_mo_lo_T = [[] for i in range(max(system_inds) + 1)]
    counter = 0
    cum_transform = np.identity(s12.shape[0])
    while counter < sub_mo_coeff.shape[1]:
        system_u = []
        system_s = []
        print('x'*99)
        for s12 in system_s12:
            u, s, vdag = np.linalg.svd(s12)
            system_u.append(u.T)
            system_s.append(s)
            # TEST: negative singular value?
            assert np.all(s > 0)
            print(s12)
            print(s)
        # find which system had the largest singular value
        max_system_ind = np.argmax([max(s) for s in system_s])
        # find largest singular value
        max_sigma_ind = np.argmax(system_s[max_system_ind])
        # add corresponding left singular vector
        system_mo_lo_T[max_system_ind].append(system_u[max_system_ind][max_sigma_ind].dot(cum_transform))
        # update overlap matrix (remove singular vector)
        trunc_transform = np.delete(system_u[max_system_ind], max_sigma_ind, axis=0)
        assert np.allclose(trunc_transform.dot(trunc_transform.T), np.identity(trunc_transform.shape[0]))
        for i in range(len(system_s12)):
            system_s12[i] = trunc_transform.dot(system_s12[i])
        # update transformation matrix
        cum_transform = trunc_transform.dot(cum_transform)
        # increment
        counter += 1

    lo_inds = [[i] * len(rows) for i, rows in enumerate(system_mo_lo_T)]
    lo_inds = np.array([j for i in lo_inds for j in i])
    print(lo_inds)
    system_mo_lo_T = np.vstack([np.vstack(rows) for rows in system_mo_lo_T if rows])
    np.set_printoptions(linewidth=200)
    print(system_mo_lo_T)

    # check orthogonalization
    s1 = mol.intor_symmetric('int1e_ovlp')
    transform = system_mo_lo_T.dot(sub_mo_coeff.T)
    olp = transform.dot(s1).dot(transform.T)
    print(olp)
    assert np.allclose(olp, np.identity(transform.shape[0]))

    # check span
    s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)
    s12 = sub_mo_coeff.T.dot(s12)
    s12 = system_mo_lo_T.dot(s12)
    for i in range(max(system_inds) + 1):
        # how much is inside
        system_s12 = s12[lo_inds == i][:, minao_inds == i]
        print(np.sum(np.diag(system_s12.T.dot(system_s12))))
        # how much is outside
        system_s12 = s12[lo_inds == i][:, minao_inds != i]
        print(np.sum(np.diag(system_s12.T.dot(system_s12))))

    coeff_mo_lo.append(system_mo_lo_T.T)

from pyscf.tools import molden
# visualize
temp = np.zeros((coeff_mo_lo[0].shape[0] + coeff_mo_lo[1].shape[0], coeff_mo_lo[0].shape[1] + coeff_mo_lo[1].shape[1]))
temp[np.where(hf.mo_occ > 0)[0][:, None],  np.where(np.arange(temp.shape[1]) < nelec // 2)[0][None, :]] = coeff_mo_lo[0] 
temp[np.where(hf.mo_occ == 0)[0][:, None], np.where(np.arange(temp.shape[1]) >= nelec // 2)[0][None, :]] = coeff_mo_lo[1] 
coeff_mo_lo = scipy.linalg.block_diag(*coeff_mo_lo)
assert np.allclose(coeff_mo_lo, temp)
t_ab_lo = mo_coeff.dot(coeff_mo_lo)
molden.from_mo(mol, 'trial.molden', t_ab_lo)

