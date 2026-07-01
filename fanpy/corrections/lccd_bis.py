import numpy as np
import scipy as sc

from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.tools import slater


class LCCD_bis:
    """
    Attributes:
    wfn : AP1roG
    ham : BaseHamiltonian or ??
    ref_sd : int
        Reference SD from AP1roG
    nspatial : int
    nspin : int
    exops : dict
    nparams : int
    params : np.ndarray with amplitudes

    Methods:
    __init__(self, wfn, ham)
    generate_exops(self)
    """


    def __init__(self, wfn, ham, pspace):

        if not isinstance(wfn, AP1roG):
            raise TypeError("Current LCCD implementation only supports AP1roG.")

        self.wfn = wfn
        self.ham = ham
        self.pspace = pspace

        self.ref_sd = wfn.ref_sd
        self.nspatial = wfn.nspatial
        self.nspin = wfn.nspin


        self.generate_exops()
        print(f'exops : {self.dict_ind_exops}')
        self.calculate_b()
        self.calculate_a()
    

    def generate_exops(self):
        # Get occupied and virtual orbital indices from reference SD
        total_occ = slater.total_occ(self.ref_sd)
        occ_indices = slater.occ_indices(self.ref_sd)
        vir_indices = slater.vir_indices(self.ref_sd, self.nspin)
        alpha_indices, _ = slater.split_spin(self.ref_sd, self.nspatial)
        occ_alpha_indices = slater.occ_indices(alpha_indices)
        vir_alpha_indices = slater.vir_indices(alpha_indices, self.nspatial)
        
        dict_exops_ind = {}
        param_ind = 0
        for i_ind, i in enumerate(occ_indices):        ##########
            for j in occ_alpha_indices:
                for a_ind, a in enumerate(vir_indices):      ######
                    for b in vir_alpha_indices:

                        if self._is_pair_excitation(i, j, a, b):      ###
                            continue                                  ###
                        
                        #Store excitation operator: (i,j,a,b) -> parameter index
                        dict_exops_ind[(i, j, a, b)] = param_ind
                        param_ind += 1 
        
        self.dict_exops_ind = dict_exops_ind
        self.dict_ind_exops = {i: exops for exops,i in dict_exops_ind.items()}
         
        # Initialize cluster amplitude parameters
        self.nparams = len(self.dict_exops_ind)
        self.amplitudes = np.zeros(self.nparams)


    def _is_pair_excitation(self, i, j, a, b):
        if i == self._get_opposite_spin(j) and a == self._get_opposite_spin(b):
            return True
        return False


    def _generate_sub_exops(self, mu):
        """
        generate all the excitations in the double singlet operator :
        E_{ia} E_{jb} = a^{\dagger}_a a_i a^{\dagger}_b a_j+ a^{\dagger}_a a_i a^{\dagger}_{\bar{b}} a_{\bar{j}}+ a^{\dagger}_{\bar{a}} a_{\bar{i}} a^{\dagger}_b a_j + a^{\dagger}_{\bar{a}} a_{\bar{i}} a^{\dagger}_{\bar{b}} a_{\bar{j}}
        """
        i, j, a, b = mu

        sub_exops = [(i,j,a,b), (i,self._get_opposite_spin(j),a,self._get_opposite_spin(b)), (self._get_opposite_spin(i),j,self._get_opposite_spin(a),b), (self._get_opposite_spin(i),self._get_opposite_spin(j),self._get_opposite_spin(a),self._get_opposite_spin(b))]   ####

        for sub_exop in sub_exops:
            i, j, a, b = sub_exop
            if i == self._get_opposite_spin(j) and a == self._get_opposite_spin(b):
                sub_exops.remove(sub_exop)

        return sub_exops


    def calculate_b(self):
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
        """
        \begin{equation}
        \begin{split}
        A_{\mu, \nu} &= \braket{\mu | [\hat{H}, \hat{\tau}_{\nu}]|AP1roG} \\
        &= \braket{\mu | \hat{H} \hat{\tau}_{\nu}|AP1roG} - \braket{\overbrace{\mu | \hat{\tau}_{\nu}}^{\delta_{\nu, \mu} \bra{\Phi_0}} \hat{H} | AP1roG}
        \\
        &= \left( \sum_{m\in S} \left( \sum_{\mu _k} \left( \sum_{\nu _k} \braket{\mu _k | \hat{H} \hat{\tau _{\nu _k}} | m}  \right) \right) * \braket{m | AP1roG} \right) - \delta_{\mu \nu} \braket{\Phi_0 | \hat{H} | AP1roG}
        \end{split}
        \end{equation}
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
                            if sub_exc_nu_m == None:   #if the excitation applied give zero
                                continue
                            a_mu_nu += self.ham.integrate_sd_sd(sub_exc_mu_refsd, sub_exc_nu_m)
                    a_mu_nu *= self.wfn.get_overlap(m)

                if nu == mu:
                    a_mu_nu -= const
                
                a[ind_mu, ind_nu] = a_mu_nu

        self.a_matrix = 0.5*a


    def compute_correction(self):
        amplitudes = sc.linalg.lstsq(self.a_matrix, -self.b_vector, check_finite=True)[0]

        self.amplitudes = amplitudes
        
        E_corr = 0.0
        print(f'amplitudes = {amplitudes}')
        print(f'two-int : {self.ham.two_int}')
        print(self.ham.two_int.size)
        for ind_nu, nu in self.dict_ind_exops.items():
            i, j, a, b = nu

            i, j, a, b = slater.spatial_index(i, self.nspatial), slater.spatial_index(j, self.nspatial), slater.spatial_index(a, self.nspatial), slater.spatial_index(b, self.nspatial) ######

            E_corr += amplitudes[ind_nu] * (2*self.ham.two_int[i, j, a, b] - self.ham.two_int[i, j, b, a])

        return E_corr


    def _get_opposite_spin(self,i):              ####all
        if slater.is_alpha(i, self.nspatial):
            return i+self.nspatial
        else:
            return i-self.nspatial               