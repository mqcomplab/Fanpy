from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
import numpy as np
import random


class RestrictedBoltzmannMachine(BaseWavefunction):
    r"""Restricted Boltzmann Machine (RBM) as a Wavefunction 

    Using the probability distribution representation by RBM as a wavefunction.
   
    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of total spin orbitals (including occupied and virtual, alpha and beta).
    bath : np.array
        Hidden variables (here, spin orbitals) for RBM.
    params : np.array
        Parameters of the RBM without including parameters for sign correction.
    memory : float
        Memory available for the wavefunction.
    orders : np.array
        Orders of interaction considered in the virutal variables i.e. spin orbitals. 
        Interaction term with order = 1 : 
            \sum_i a_i n_i, 
                where a_i : coefficients, n_i : occupation number for spin orbital i.     
        Interaction term with order = 2 : 
            \sum_{i, j} a_{ij} n_i n_j, 
                where a_i : coefficients, 
                    n_i, n_j  : occupation number for spin orbitals i, j, respectively.     

    """
    def __init__(
        self, nelec, nspin, bath=[-1,1], params=None, memory=None, orders=(1)
    ):

        super().__init__(nelec, nspin, memory=memory)
        self.bath = np.array(bath)
        self.orders = np.array(orders)
        self._template_params = None
        #self._template_sign_params = None
        #self.assign_sign_params(sign_params=None)
        self.assign_params(params=params)
        self.cache_wfn_olp = 1.0 # store olp without sign
        self.cache_sign_output = 1.0 # store sign_output 

     
    @property
    def params(self):
        return np.hstack([i.flat for i in self._params])  

    @property
    def nbath(self):
        return (self.bath.size)

    @property
    def nparams(self):
        return (
            np.sum(self.nspin ** self.orders) 
            + self.nbath + (self.nspin * self.nbath) + (self.nspin + 1) 
        )   

    
    @property
    def nsign_params(self):
        # total sign params = sign_params_for_virtual +1 (bias) 
        return (self.nspin + 1) 
  
    @property
    def params_shape(self):
        # excluding sign_params here
        # Coefficient matrix for interaction order = 1 with size (nspin)
        # Coefficient matrix for interaction order = 1 with size (nspin, nspin)      
        # Coefficient matrix for hidden variables with size (nbath)
        # weights matrix of size nbath x nspin
        return ( 
            [((self.nspin, ) * self.orders)] + 
            [(self.nbath, )] + [(self.nspin, self.nbath)] + [(self.nspin + 1, )] 
          
        )

 
    @property
    def template_params(self):
         return self._template_params

    #@property
    #def template_sign_params(self):
    #     return self._template_sign_params
 
    @staticmethod
    def sign_correction(x):
        return np.tanh(x)

    @staticmethod
    def sign_correction_deriv(x):
        return 1 - np.tanh(x) ** 2


    def assign_template_params(self):
        params = []

        for i, param_shape in enumerate(self.params_shape[:-1]):
            random_params = [(random.random() * 0.1) - 0.05 for _ in range(np.prod(param_shape))]
            random_params = np.array(random_params).reshape(param_shape) 
            params.append(random_params) 

            #params.append(np.zeros(param_shape))
            #params.append(np.ones(param_shape) * 22.0)

        random_params = [random.choice([-50,50]) for _ in range(self.nsign_params)]
        sign_params = np.array(random_params) 
        params.append(sign_params)
        
        ''' 
        ### Assigning template params so that only HF will be enabled
        ground_state = slater.ground(self.nelec, self.nspin)
        occ_indices = slater.occ_indices(ground_state)

        as_ = np.zeros(self.nspin)
        as_[occ_indices] = 1.0
        params.append(as_)

        bs_ = np.zeros(self.nbath)
        if self.nbath == self.nspin:
            bs_[occ_indices] = 1.0
        params.append(bs_)
        
        ws_ = np.zeros((self.nspin, self.nbath))
        for i in range(self.nspin):
            for j in range(self.nbath):
                if as_[i] == 1.0 and bs_[j] == 1.0:
                    ws_[i,j] = 1.0  

        params.append(ws_)
        '''
        self._template_params = params

    #def assign_template_sign_params(self):
        #random_params = [(random.random() * -10.0) - 5.0 for _ in range(self.nsign_params)]
        #random_params = random.sample(range(-30, 30), self.nsign_params)
        #random_params = [random.choice([-50,50]) for _ in range(self.nsign_params)]
        #sign_params = np.array(random_params) 

        #random_params = [(random.choice([-10, 10.0])) for _ in range(self.nsign_params)]
        #sign_params = np.array(random_params) 

        #sign_params = np.ones(self.nsign_params) * -10
        #self._template_sign_params = sign_params

    #def assign_sign_params(self, sign_params=None):
    #    if sign_params is None:
    #        if self._template_sign_params is None:
    #            self.assign_template_sign_params()
    #    self.sign_params = self._template_sign_params

  

    def assign_params(self, params=None, add_noise=False):
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self.template_params

        if isinstance(params, np.ndarray):
            structured_params = []
            for param_shape in self.params_shape:
                structured_params.append(params[:np.prod(param_shape)].reshape(*param_shape))
                params = params[np.prod(param_shape):]
            params = structured_params 
        
        self._params = params
        
        # append sign parameters
        #if self._template_sign_params is None:
        #    self.assign_template_sign_params()
        #    self.assign_sign_params()
        #self._params.append(self.sign_params)
 
        #print("self._params:", self._params)

    '''
    def get_overlap(self, sd, deriv=None):
        if deriv is None: 
            return self._olp(sd)
        return self._olp_deriv(sd)[deriv]



    def _olp(self, sd):
        occ_indices = np.array(slater.occ_indices(sd))
        occ_vec = np.ones(self.nspin) * -1.0
        occ_vec[occ_indices] = 1.0
       
         
        #result = 0.0
        #for i in range(self.nspin):
        #    result += self._params[0][i] * occ_vec[i] 
        #    for j in range(self.nbath):
        #        result += self._params[2][i][j] * occ_vec[i] * self.bath[j]
        #for j in range(self.nbath):
        #    result += self._params[1][j] * self.bath[j]

        #result = np.sqrt(np.exp(result))

        a = self._params[0]
        b = self._params[1]
        w = self._params[2]
        sign_params = self._params[3]
        
        sum_ = np.sum(a * occ_vec) + np.sum(b * self.bath) + np.sum(w * np.outer(occ_vec, self.bath))
        numerator = self.nbath * np.exp(sum_)

        ##NOTE: Denominator should be modified since it is the sum over all sds
        ###     which is done in get_overlaps function where the overlap of all
        ###     sds are computed together. 
        denominator = self.nspin * self.nbath * np.exp(sum_)
        result = np.sqrt(numerator/denominator)

        self.cache_wfn_olp = result
        #print("olp", sd, self.cache_wfn_olp)
 

        # Adding sign_correction  
        sign_input = (np.sum(sign_params[:-1] * occ_vec) +
            sign_params[-1])

        sign_result = self.sign_correction(sign_input)
        #print("sign_result", sign_result)
        self.cache_sign_output = sign_result

        return result * sign_result

    def _olp_deriv(self, sd):
        "Derivative with respect to virutal variables/elements of occupation vector"
        occ_indices = np.array(slater.occ_indices(sd))
        occ_vec = np.ones(self.nspin) * -1.0
        occ_vec[occ_indices] = 1.0
       
        output = [] 
        #print("olp_Deriv", sd, self.cache_wfn_olp)
        term = (1/2) * self.cache_wfn_olp
        
        # Derivative with respect to coefficients of virtual variables
        result = term * occ_vec * self.cache_sign_output
        output.append(result)
        
        # Derivative with respect to hidden variables
        result = term * self.bath * self.cache_sign_output
        output.append(result)
        

        # Derivative with respect to weights
        result = 1.0
        for i in range(self.nspin):
            for j in range(self.nbath):
                result = occ_vec[i] * self.bath[j]
                result = term * result * self.cache_sign_output
                output.append(result)
 
        # Derivative with respect to sign_params for virtual 
        result = self.cache_wfn_olp * (1 - self.cache_sign_output ** 2) * occ_vec
        output.append(result)
        

        # Derivative with respect to sign_params for bias 
        result = self.cache_wfn_olp * (1 - self.cache_sign_output ** 2) 
        output.append(result)
        return np.hstack(output)
        

    def get_overlap(self, sd, deriv=None):
        if deriv is None:
            return self._olp(sd)
        #print(deriv)
        return self._olp_deriv(sd)[deriv]
    '''

    def get_overlaps(self, sds, deriv=None):
        if len(sds) == 0:
            return np.array([])
        
        occ_indices = np.array([slater.occ_indices(sd) for sd in sds])
        occ_mask = np.ones((len(sds), self.nspin)) * -1.0
        for i, inds in enumerate(occ_indices):
            occ_mask[i, inds] = 1.0
        
        a = self._params[0]
        b = self._params[1]
        w = self._params[2]
        sign_params = self._params[3]
        print("\na", a, "\nb", b, "\nw", w, "\nsign_params", sign_params, "\n")
        #print("\nocc_mask:", occ_mask)
        print("\nself.bath:", self.bath)
        print("\nsds: ", sds)
       
      
        olp_ = []
        wfn_only_olp = []
        sign_output = []

        for i in range(len(sds)):
            occ_vec = occ_mask[i]
            print("\ni, occ_vec:", i, occ_vec)
            sum_ = np.sum(a * occ_vec) + np.sum(b * self.bath) + np.sum(w * np.outer(occ_vec, self.bath))
            numerator =   np.exp(sum_) 
            # we are using only one list of hidden variables unlike the set of virtual varibles which 
            # constitutes len(sds) number of occupation vectors.  

            wfn_only_olp.append(numerator)  # appending the numerator
        
            sign_input = (np.sum(sign_params[:-1] * occ_vec) + sign_params[-1] )
            sign_result = self.sign_correction(sign_input)
            print("i, sign_input, sign_result:", i, sign_result)
            sign_output.append(sign_result)
            #print("\nsds, sign_correction", sds[i], sign_result, "\n")

         
        wfn_only_olp = np.array(wfn_only_olp)
        partition_func = np.sum(wfn_only_olp) # denominator

        if len(wfn_only_olp) == len(sign_output) == len(sds):            
            f_wfn = np.sqrt(wfn_only_olp / partition_func)
            print("\nroot P", f_wfn)
            olp_ = f_wfn * sign_output
            print("\nfinal olp", olp_)

        if deriv is None:
            return np.array(olp_)

        ### If deriv is not None
        output = []
    
    
        for i in range(len(sds)):
            occ_vec = occ_mask[i]
            # Derivative with respect to coefficients of virtual variables
            result = 1/2 * olp_[i] * occ_vec 
            sd_i = np.hstack((np.array([]), result))
        
            # Derivative with respect to hidden variables
            result = 1/2 * olp_[i] * self.bath
            sd_i = np.hstack((sd_i, result))

            # Derivative with respect to weights
            result = 1.0
            for j in range(self.nspin):
                for k in range(self.nbath):
                    nj_hk = occ_vec[j] * self.bath[k]
                    result = 1/2 * olp_[i] * nj_hk
                    sd_i = np.hstack((sd_i, result))
 
            # Derivative with respect to sign_params for virtual 
            result = f_wfn[i] * (1 - sign_output[i] ** 2) * occ_vec
            sd_i = np.hstack((sd_i, result))

            # Derivative with respect to sign_params for bias 
            result = f_wfn[i] * (1 - sign_output[i] ** 2) 
            sd_i = np.hstack((sd_i, result))
            
            output.append(list(sd_i))
        output = np.array(output)
        #print("\na: ", a, "\nb: ",b, "\nw: ", w, "\nsign_params", sign_params) 
        #print('\n\nlen(sds)', len(sds), '\n')
        #print("\noverlap", olp_)
        print("\nolp derivative", output)
        return  output[:, deriv]
