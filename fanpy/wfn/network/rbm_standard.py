'Implementation of Standard RBM expression'
from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
import numpy as np
import random
import sys
np.set_printoptions(threshold=sys.maxsize)

class RestrictedBoltzmannMachine(BaseWavefunction):
    r"""Restricted Boltzmann Machine (RBM) as a Wavefunction 
    Expression
    ----------
     f(\vec{a}, \vec{b}, \textbf{w}, \vec{c}, d, \vec{n}) = \sqrt{\frac{exp(\sum_i a_i n_i + \sum_j b_j h_j + \sum_{ij} w_{ij} n_i h_j)}{\sum_{\{n\}}exp(\sum_i a_i n_i + \sum_j b_j h_j + \sum_{ij} w_{ij} n_i h_j)}} tanh(\sum_k c_k n_k + d)
 

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
        self, nelec, nspin, bath=[0,1], params=None, memory=None, orders=(1)
    ):

        super().__init__(nelec, nspin, memory=memory)
        self.bath = np.array(bath)
        self.orders = np.array(orders)
        self._template_params = None
        self.assign_params(params=params)

     
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
        # Coefficient matrix for interaction order = 1 with size (nspin)
        # Coefficient matrix for interaction order = 2 with size (nspin, nspin)      
        # Coefficient matrix for hidden variables with size (nbath)
        # weights matrix of size nspin x nbath
        # coefficient matrix for sign parameter of size (nspin + 1)
        return ( 
            [((self.nspin, ) * self.orders)] + 
            [(self.nbath, )] + [(self.nspin, self.nbath)] + [(self.nspin + 1, )] 
          
        )

 
    @property
    def template_params(self):
         return self._template_params

 
    @staticmethod
    def sign_correction(x):
        '''
        Sign correction function to introduce + or - sign to the exp() ansatz of RBM.
        '''
        return np.tanh(x)

    @staticmethod
    def sign_correction_deriv(x):
        return 1 - np.tanh(x) ** 2


    def assign_template_params(self):
        '''
        The template parameters are set to a default range of [-0.02, 0.02).
        '''
        params = []
        random.seed(10)
        for i, param_shape in enumerate(self.params_shape[:-1]):
            random_params = [(random.random() * 0.04) - 0.02 for _ in range(np.prod(param_shape))]
            random_params = np.array(random_params).reshape(param_shape) 
            params.append(random_params) 


        random_params = [(random.random() * 0.04) - 0.02 for _ in range(self.nsign_params)]
        sign_params = np.array(random_params) 
        params.append(sign_params)
        self._template_params = params


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
        a = self._params[0]
        b = self._params[1]
        w = self._params[2]
        sign_params = self._params[3]
        #print("\na = np.",repr(a),"\nb=np.",repr(b),"\nw=np.",repr(w),"\nsign_params=np",repr(sign_params), "\n")
        

    def get_overlaps(self, sds, deriv=None):
        '''
        Function to calculate the overlap of |RBM> with given set of Slater determinants. 
     
        Input
        -----
        sds: list
            List of Slater determinants with respect to which overlap of RBM wavefunction needs to be calculated.
        deriv: (None, list)
            If deriv is None meaning that only overlap is required or else list of indices of parameters is provided
            with respect to which derivative is required.  
        '''
        if len(sds) == 0:
            return np.array([])
        
        occ_indices = np.array([slater.occ_indices(sd) for sd in sds])
        occ_mask = np.zeros((len(sds), self.nspin)) 
        for i, inds in enumerate(occ_indices):
            occ_mask[i, inds] = 1.0
        
        a = self._params[0]
        b = self._params[1]
        w = self._params[2]
        sign_params = self._params[3]
        #print("\na = np.",repr(a),"\nb=np.",repr(b),"\nw=np.",repr(w),"\nsign_params=np",repr(sign_params), "\n")
        #print("occ_mask = np.",repr(occ_mask))
        #print("h = np.",repr(self.bath))
        #print("sds = ",repr(sds))
       
      
        olp_ = []
        numerator = []
        sign_output = []
        exp_n = [] #numerator * occ_vec, for each sds
        exp_h = [] #numerator * self.bath, for each sds
        exp_nh = [] #numerator * outer_product(occ_vec, self.bath), for each sds

        for i in range(len(sds)):
            occ_vec = np.array(occ_mask[i])
            #print("\ni, occ_vec:", i, occ_vec)
            sum_1 = np.sum(a * occ_vec) + np.sum(b * self.bath)
            outer_oh = np.einsum('i,j->ij', occ_vec, self.bath)
            sum_oh = (w.ravel() * outer_oh.ravel())
            #print("w.ravel():", w.ravel())
            #print("outer_oh.ravel():", outer_oh.ravel())
            #print("\nsum_oh:", sum_oh)
            sum_oh = np.sum(sum_oh)
            numerr =   np.exp(sum_1 + sum_oh) 
            # we are using only one list of hidden variables unlike the set of virtual varibles which 
            # constitutes len(sds) number of occupation vectors.  

            numerator.append(numerr)  # appending the numerator
            exp_n.append(numerr * occ_vec)
            exp_h.append(numerr * self.bath)
            exp_nh.append(numerr * outer_oh)
        
            sign_input = (np.sum(sign_params[:-1] * occ_vec) + sign_params[-1] )
            sign_result = self.sign_correction(sign_input)
            #print("i, sign_input, sign_result:", i, sign_result)
            sign_output.append(sign_result)
            #print("\nsds, sign_correction", sds[i], sign_result, "\n")

         
        numerator = np.array(numerator)
        partition_func = np.sum(numerator) # denominator
        sign_output = np.array(sign_output)
        exp_n = np.array(exp_n)
        exp_h = np.array(exp_h)
        exp_nh = np.array(exp_nh)
        #print("\nnumerator = np.",repr(numerator))
        #print("\npartition_func: ", partition_func)
        #print("\nexp_n = np.",repr(exp_n))
        #print("\nexp_h = np.",repr(exp_h))
        #print("\nexp_nh = np.",repr(exp_nh))
        #print("\nsign_output = np.",repr(sign_output))

        if len(numerator) == len(sign_output) == len(sds):            
            f_wfn = np.sqrt(numerator / partition_func)
            #print("\nroot P", f_wfn)
            olp_ = f_wfn * sign_output
            #print("\nfinal olp", olp_)

        if deriv is None:
            return np.array(olp_)

        ### If deriv is not None
        output = []
    
    
        for i in range(len(sds)):
            occ_vec = occ_mask[i]
            #print(f"\n----PRINTING derivatives info for sd={sds[i]}------\n")
            #if i in [0,1,2]:
            # Derivative with respect to coefficients of virtual variables
                #print("\nnp.sum(exp_n, axis=0):", np.sum(exp_n, axis=0))
                #print("\npartition_func: ", partition_func)
                #print("occ_vec: ", occ_vec)
                #print("partition_func * occ_vec: ", partition_func * occ_vec)
            term = (partition_func * occ_vec - np.sum(exp_n, axis=0)) / partition_func
            result = 1.0/2.0 * olp_[i] * term
            sd_i = np.hstack((np.array([]), result))
            #if i in [0,1,2]:
                #print("\nwrt a =", repr(result))
        
            # Derivative with respect to hidden variables
            term = (partition_func * self.bath - np.sum(exp_h,axis=0)) / partition_func
            result = 1.0/2.0 * olp_[i] * term
            sd_i = np.hstack((sd_i, result))
            #if i in [0,1,2]:
                #print("\nwrt b =", repr(result))

            # Derivative with respect to weights
            result = 1.0
            outer_oh = np.einsum('i,j->ij', occ_vec, self.bath)
            term = (partition_func * outer_oh - np.sum(exp_nh, axis=0)) / partition_func
            result = 1.0/2.0 * olp_[i] * term.ravel()
            sd_i = np.hstack((sd_i, result))
            #if i in [0,1,2]:
                #print("\nwrt w =", repr(result))
 
            # Derivative with respect to sign_params for virtual 
            result = f_wfn[i] * (1 - sign_output[i] ** 2) * occ_vec
            sd_i = np.hstack((sd_i, result))
            #if i in [0,1,2]:
                #print("\nwrt d =", repr(result))

            # Derivative with respect to sign_params for bias 
            result = f_wfn[i] * (1 - sign_output[i] ** 2) 
            sd_i = np.hstack((sd_i, result))
            #if i in [0,1,2]:
                #print("\nwrt e =",repr(result))
                #print(f"\nWHOLE array of derivatives for sd={sds[i]}:\n", sd_i)
            
            output.append(list(sd_i))
        output = np.array(output)
        #print("\na: ", a, "\nb: ",b, "\nw: ", w, "\nsign_params", sign_params) 
        #print('\n\nlen(sds)', len(sds), '\n')
        #print("\noverlap", olp_)
        #print("\nolp derivative", output)
        #print(deriv)
        return  output[:, deriv]
