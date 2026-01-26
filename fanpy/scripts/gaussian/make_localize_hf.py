
def make_script(xyzfile, basis, method, system_inds, unit="Bohr", mo_coeff_file=None, filename="calculate.py"):
    """Create a Python script to perform localization and integral calculation.

    Parameters
    ----------
    xyzfile : str
        Path to the XYZ file containing the molecular geometry.
    basis : str
        Basis set name for PySCF. 
    """
    if mo_coeff_file is not None:
        mo_coeff_file = f'"{mo_coeff_file}"'

    with open(filename, 'w') as f:
        f.write(
            f"""from fanpy.tools.wrapper.pyscf_tools import localize
import numpy as np

with open('{xyzfile}', 'r') as f:
    xyz_lines = f.readlines()
xyz_lines = [i for i in xyz_lines if i.strip()]

with open('./system.xyz', 'w') as f:
    f.write(str(len(xyz_lines)))
    f.write('\\n\\n')
    for line in xyz_lines:
        f.write(line)

results = localize("./system.xyz", "{basis}", mo_coeff_file={mo_coeff_file}, unit="{unit}",
                   method="{method}", system_inds={system_inds})

print(f"Nuclear-nuclear repulsion: {{results['nuc_nuc']}}")
print(f"HF Electronic Energy: {{results['hf_energy']}}")
print(f"HF Total Energy: {{results['hf_energy'] + results['nuc_nuc']}}")

np.save("ao_inds.npy", results["ao_inds"])
np.save("oneint.npy", results["one_int"])
np.save("twoint.npy", results["two_int"])
np.save("hf_energies.npy", [results["hf_energy"], results["nuc_nuc"]])
np.save("t_ab_lo.npy", results["t_ab_mo"])
"""
    )
