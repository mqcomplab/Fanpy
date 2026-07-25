# This file is to implement the linearized CCD correction based on the paper by Boguslawski & Ayers (2015)

# First Goal: create an implementation for the AP1roG wavefunction that can be generalized to other wavefunctions

# Need to be use like this:
# lccd = LCCD(wfn, ham, pspace)
# corr = lccd.compute_correction()

# Pay attention to the fact that wfn (and ham) are wavefunction (and hamiltonian) that will be used for the correction.
# One should probably reassign the optimized parameters (after pCCD calculation for example) to the wfn (and ham if needed).

### This is the "First" implementation as proposed on the Overleaf document 'LCCD results for AP1roG' ###


import numpy as np
import scipy as sc

from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.tools import slater

class LCCD:
    r""" Linearized Coupled Cluster correction. Currently implemented for AP1roG / pCCD only. "First" implementation.

    .. math::

        \ket{AP1roG-LCCD} = e^{\hat{T}_2} \ket{AP1roG}

    where :math:`\hat{T}_2` is the doubles cluster operator defined as:

    .. math::

        \hat{T}_2 = \sum_{\nu} t_{\nu} \hat{\tau}_{\nu} = \frac{1}{2} \sum_{i,j}^{occ} \sum_{a,b}^{virt} {'}t_{ij}^{ab} \hat{E}_{ai} \hat{E}_{bj}

    .. math::

        \hat{E}_{ai} = a_a^{\dagger}a_i + a_{\bar{a}}^{\dagger} a_{\bar{i}}

    Attributes
    ----------
    wfn : AP1roG
        Wavefunction used to compute the correction. Currently only AP1roG works.
    ham : GeneralizedMolecularHamiltonian
        Hamiltonian used to compute the correction
    pspace : list of int
        Projection space (list of Slater determinants)
    ref_sd : int
        Reference Slated determinant of the wfn instance.
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    a_matrix : np.ndarray
        Matrix A (nparams x nparams)
    b_vector : np.ndarray
        Matrix B (nparams x 1)
    amplitudes : np.ndarray
        Amplitudes after solving the linear problem
    dict_exops_ind : dict of 4-tuple of int to int
        Dictionary of excitation operators (i,j,a,b) where i,j,a,b are spatial orbital indices to the index of the excitation operator
    dict_ind_exops : dict of int to 4-tuple of int
        Dictionary of the indices of the excitation operators to the corresponding excitation operators (i,j,a,b) where i,j,a,b are spatial orbital indices
    nparams : int
        Numbers of parameters (amplitudes)

    Methods
    -------
    __init__(self, wfn, ham, pspace)
        Initialize the LCCD correction.
    generate_exops(self)
        Generate the different excitation operators.
    _generate_sub_exops(self, mu)
        Generate the sub-excitations from a certain excitation mu (mu being the 4-tuple (i,j,a,b)) and remove the pair excitations.
    calculate_a(self)
        Calculate the matrix A.
    calculate_b(self)
        Calculate the matrix B.
    compute_correction(self)
        Compute the LCCD correction to the energy. 

    """

    def __init__(self, wfn, ham, pspace):
        r"""Initialize the LCCD correction instance.

        Parameters
        ----------
        wfn : AP1roG
            Wavefunction used to compute the correction. Can be only AP1roG for now.
        ham : GeneralizedMolecularHamiltonian
            Hamiltonian that defines the system under study.
        pspace : list of int
            List of Slater determinant defining the projection space.
        
        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of AP1roG.
        
        """
        if not isinstance(wfn, AP1roG):
            raise TypeError("Current LCCD implementation only supports AP1roG.")

        self.wfn = wfn
        self.ham = ham
        self.pspace = pspace

        self.ref_sd = wfn.ref_sd
        self.nspatial = wfn.nspatial
        self.nspin = wfn.nspin

        self.generate_exops()
        self.calculate_b()
        self.calculate_a()

    def generate_exops(self):
        r"""Generate the different excitation operators.
        
        Creates the two attributes dict_ind_exops and dict_exops_ind. The excitation operators are defined with 4-tuples (i,j,a,b)
        with i,j,a,b being integers standing for spatial orbitals indices. Reminder that the alpha spin orbital indices and the spatial
        orbital indices are the same.

        """
        total_occ = slater.total_occ(self.ref_sd)
        occ_indices = slater.occ_indices(self.ref_sd)
        vir_indices = slater.vir_indices(self.ref_sd, self.nspin)
        alpha_indices, _ = slater.split_spin(self.ref_sd, self.nspatial)
        occ_alpha_indices = slater.occ_indices(alpha_indices)
        vir_alpha_indices = slater.vir_indices(alpha_indices, self.nspatial)
        
        dict_exops_ind = {}
        param_ind = 0
        for i in occ_alpha_indices:
            for j in occ_alpha_indices:
                for a in vir_alpha_indices:
                    for b in vir_alpha_indices:
                        dict_exops_ind[(i, j, a, b)] = param_ind
                        param_ind += 1 
        
        self.dict_exops_ind = dict_exops_ind
        self.dict_ind_exops = {i: exops for exops,i in dict_exops_ind.items()}

        self.nparams = len(self.dict_exops_ind)

    def _generate_sub_exops(self, mu):
        r"""Generate all the sub-excitations (from a excitation operator mu) created by the double singlet operator:
        
        .. math::

            E_{ia} E_{jb} = a^{\dagger}_a a_i a^{\dagger}_b a_j+ a^{\dagger}_a a_i a^{\dagger}_{\bar{b}} a_{\bar{j}}+ a^{\dagger}_{\bar{a}} a_{\bar{i}} a^{\dagger}_b a_j + a^{\dagger}_{\bar{a}} a_{\bar{i}} a^{\dagger}_{\bar{b}} a_{\bar{j}}
        
        Here, we go from a picture of a spatial-orbital indices excitation (mu) to some spin-orbital indices real excitations.
        We remove here the pair sub-excitations by removing some 4-tuple of the list. 

        Parameters
        ----------
        mu : 4-tuple of int
            Excitation operator from which we compute the sub-excitations.
        
        Returns
        -------
        sub_exops : list of 4-tuple of int
            Sub-excitations (k,l,c,d) created from mu. Here k,l,c,d are spin-orbital indices.

        """
        i, j, a, b = mu
        sub_exops = [(i,j,a,b), (i,j+self.nspatial,a,b+self.nspatial), (i+self.nspatial,j,a+self.nspatial,b), (i+self.nspatial,j+self.nspatial,a+self.nspatial,b+self.nspatial)]
        for sub_exop in sub_exops:
            i, j, a, b = sub_exop
            if i == j+self.nspatial and a == b+self.nspatial:
                sub_exops.remove(sub_exop)
            if j == i+self.nspatial and b == a+self.nspatial:
                sub_exops.remove(sub_exop)

        return sub_exops

    def calculate_b(self):
        r""" Calculate the B matrix elements.
        The B matrix elements are defined as:

        .. math::

            B_\mu = \braket{\mu | \hat{H} | AP1roG}
        
        The implementation proposed is:

        .. math::

            B_\mu = \sum_k \braket{\mu_k | \hat{H} | AP1roG}
        
        where the sum run on all the sub-excitations :math:`\mu_k` from a certain excitation :math:`\mu`.
        
        """
        b = np.zeros((self.nparams))
        for ind_mu, mu in self.dict_ind_exops.items():
            b_mu = 0.0
            for sub_mu_exop in self._generate_sub_exops(mu):
                sub_exc_refsd = slater.excite(self.ref_sd, *sub_mu_exop)
                if sub_exc_refsd == None:
                    continue
                b_mu += self.ham.integrate_sd_wfn(sub_exc_refsd, self.wfn)
            b[ind_mu] = b_mu
        self.b_vector = b

    def calculate_a(self):
        r""" Calculate the A matrix elements.
        The A matrix elements are defined as:

        .. math::

            A_{\mu \nu} = \frac{1}{2}\braket{\mu | [\hat{H}, \hat{\tau _ \nu}] | AP1roG}

        The implementation proposed is:

        .. math::

            A_{\mu \nu} = \frac{1}{2} \left[ \sum_{m\in S} \left( \sum_k \sum_l \left( \braket{\mu_k | \hat{H} \hat{\tau}_{\nu_{l}} | m} \right) \braket{m|AP1roG} \right) - \delta _{\mu \nu} \braket{\Phi_0 | \hat{H} | AP1roG} \right]

        where :math:`\mu_k` are sub-excitations taken from :math:`\mu` and :math:`\nu_l` are sub-excitations taken from :math:`\nu`.
        
        """
        a = np.zeros((self.nparams, self.nparams))
        const = self.ham.integrate_sd_wfn(self.ref_sd, self.wfn)

        for ind_mu, mu in self.dict_ind_exops.items():
            for ind_nu, nu in self.dict_ind_exops.items():
                for m in self.pspace:
                    a_mu_nu = 0.0
                    for sub_mu_exop in self._generate_sub_exops(mu):
                        sub_exc_mu_refsd = slater.excite(self.ref_sd, *sub_mu_exop)
                        for sub_nu_exop in self._generate_sub_exops(nu):
                            sub_exc_nu_m = slater.excite(m, *sub_nu_exop)
                            if sub_exc_mu_refsd == None:
                                continue
                            if sub_exc_nu_m == None:   # If the excitation applied gives zero
                                continue
                            a_mu_nu += self.ham.integrate_sd_sd(sub_exc_mu_refsd, sub_exc_nu_m)
                    a_mu_nu *= self.wfn.get_overlap(m)

                if nu == mu:
                    a_mu_nu -= const
                
                a[ind_mu, ind_nu] = a_mu_nu

        self.a_matrix = 0.5*a

    def compute_correction(self):
        r"""Compute the LCCD correction to the energy.

        .. math::

            E_{corr} = \sum_{ijab} t_{ij}^{ab} (\braket{ij||ab} - \braket{ij|ba})

        where the physicist's notation is used. 
            
        Returns
        -------
        E_corr : float
            LCCD correction to the energy.
        
        """
        amplitudes = sc.linalg.lstsq(self.a_matrix, -self.b_vector, check_finite=True)[0]
        self.amplitudes = amplitudes
        
        E_corr = 0.0
        print(f'amplitudes = {amplitudes}')
        print(f'two-int : {self.ham.two_int}')
        print(self.ham.two_int.size)
        for ind_nu, nu in self.dict_ind_exops.items():
            i, j, a, b = nu
            E_corr += amplitudes[ind_nu] * (2*self.ham.two_int[i, j, a, b] - self.ham.two_int[i, j, b, a])

        return E_corr