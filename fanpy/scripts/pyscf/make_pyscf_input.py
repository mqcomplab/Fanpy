def make_pyscf_input(coords, basis, memory=4096, charge=0, spin=0, units='B'):
    """Make a PySCF input file.

    Parameters
    ----------
    coords : str
        Coordinates of the molecule.
    basis : str
        Basis set used for the calculation.
    memory : str
        Amount of memory used by PySCF in MB.
    charge : int
        Charge of the molecule.
    spin : int
        Number of unpaired electrons of the molecule.
        Singlet is 0, doublet is 1, triplet is 2, etc.
    units : str
        Units of coordinates. Default is Bohrs.

    """

    output = "# PySCF calculation: HF//{:}\n".format(basis)
    output += "import pyscf.gto\n"
    output += "import pyscf.scf\n\n"

    output += "mol = pyscf.gto.Mole()\n"
    output += "mol.atom = {:}\n".format(str(coords))

    output += "mol.charge = {:}\n".format(charge)
    output += "mol.spin = {:}\n\n".format(spin)

    output += "mol.unit = '{:}'\n".format(str(units))
    output += "mol.basis = '{:}'\n".format(basis)
    output += "mol.verbose = 4\n"
    output += "mol.max_memory = {:}\n".format(memory)
    output += "mol.build()\n\n"

    output += "# RHF calculation\n"
    output += "mf = pyscf.scf.RHF(mol)\n"
    output += "mf.conv_tol = 1e-12\n"
    output += "mf.max_cycle = 100\n"
    output += "mf.kernel()\n\n"

    output += "if mf.check_convergence:\n"
    output += "    print('> PySCF calculation not converged. Aborting calculation...')\n"
    output += "    raise RuntimeError('> PySCF calculation not converged. Aborting calculation...')\n\n"

    output += "# PySCF interface\n"
    output += "import fanpy.interface.pyscf\n"
    output += "interface = fanpy.interface.pyscf.PYSCF(mf)\n\n"

    return output
