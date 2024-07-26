import numpy as np

# Log Levels
INFO = 4
NOTE = 3


class PYSCF:

    def __init__(self, mf):

        try:
            import pyscf
        except ImportError:
            print("# ERROR: PySCF package not found.")

        # Output settings
        def info(str, *args):
            if self.verbose >= INFO:
                print(str % args)

        def note(str, *args):
            if self.verbose >= NOTE:
                print(str % args)

        if isinstance(mf, pyscf.scf.hf.SCF):
            self.type = "pyscf"
        else:
            raise TypeError("Invalid object: SCF object from PySCF not found.")

        self.verbose = mf.verbose
        info("Importing PySCF objects...")

        # General info
        self.mol = mf.mol

        # Number of electrons
        self.nelec = self.mol.nelectron
        info("> Number of electrons: %i", self.nelec)

        # Number of spin orbitals
        self.mo_coeff = mf.mo_coeff.copy()
        self.nmo = self.mo_coeff.shape[1]
        self.nspinorb = self.nmo * 2
        info("> Number of spin orbitals: %i", self.nspinorb)

        # Hartree-Fock energy
        self.energy_total = self.e_tot
        info("> Hartree-Fock total energy: %.15f", self.energy_total)

        # Electronic energy
        self.energy_elec = self.energy_elec()[0]
        info("> Electronic energy: %.15f", self.energy_elec)

        # Nuclear-nuclear repulsion
        self.energy_nuc = self.mol.energy_nuc()
        info("> Nuclear-nuclear repulsion: %.15f", self.energy_nuc)

        # One-electron integrals
        self.one_int_ao = mf.get_hcore()

        info("> Transforming one-electron integrals from AO to MO basis...")
        self.one_int = np.einsum(
            "ji,jk,kl->il",
            self.mo_coeff,
            self.one_int_ao,
            self.mo_coeff,
            optimize="greedy",
        )

        # Two-electron integrals
        from pyscf import ao2mo

        ## Check if two-electron integrals were calculated incore or outcore
        if isinstance(mf._eri, np.ndarray):
            self.two_int_ao = mf._eri
        else:
            self.two_int_ao = mf.mol

        info("> Transforming two-electron integrals from AO to MO basis...")
        self.two_int = ao2mo.general(
            self.two_int_ao,
            (self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff),
            compact=False,
        )

        ## Convert integrals to Physicist's notation
        self.two_int = self.two_int.reshape(
            self.nmo, self.nmo, self.nmo, self.nmo
        ).transpose(0, 2, 1, 3)

        note("Importing PySCF objects completed.")
