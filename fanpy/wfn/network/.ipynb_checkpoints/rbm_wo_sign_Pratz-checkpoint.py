from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
import numpy as np
import random


class RestrictedBoltzmannMachine(BaseWavefunction):
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
            + self.nbath + (self.nspin * self.nbath) 
        )   

    
  
    @property
    def params_shape(self):
        # Coefficient matrix for interaction order = 1 with size (nspin)
        # Coefficient matrix for interaction order = 1 with size (nspin, nspin)      
        # Coefficient matrix for hidden variables with size (nbath)
        # weights matrix of size nbath x nspin
        return ( 
            [((self.nspin, ) * self.orders)] + 
            [(self.nbath, )] + [(self.nspin, self.nbath)] 
          
        )

 
    @property
    def template_params(self):
         return self._template_params

 


    def assign_template_params(self):
        params = []

        for i, param_shape in enumerate(self.params_shape):
            random_params = [(random.random() * 0.32) - 0.16 for _ in range(np.prod(param_shape))]
            random_params = np.array(random_params).reshape(param_shape) 
            params.append(random_params) 
            #params.append(np.zeros(param_shape))
  
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
        


    def get_overlap(self, sd, deriv=None):
        if deriv is None: 
            return self._olp(sd)
        return self._olp_deriv(sd)[deriv]



    def _olp(self, sd):
        occ_indices = np.array(slater.occ_indices(sd))
        occ_vec = np.zeros(self.nspin)
        occ_vec[occ_indices] = 1.0
       
        ''' 
        result = 0.0
        for i in range(self.nspin):
            result += self._params[0][i] * occ_vec[i] 
            for j in range(self.nbath):
                result += self._params[2][i][j] * occ_vec[i] * self.bath[j]
        for j in range(self.nbath):
            result += self._params[1][j] * self.bath[j]

        result = np.sqrt(np.exp(result))
        '''
        a = self._params[0]
        b = self._params[1]
        w = self._params[2]
        
        sum_ = np.sum(a * occ_vec) + np.sum(b * self.bath) + np.sum(w * np.outer(occ_vec, self.bath))
        result = np.sqrt(np.exp(sum_))

 
        return result 

    def _olp_deriv(self, sd):
        "Derivative with respect to virutal variables/elements of occupation vector"
        occ_indices = np.array(slater.occ_indices(sd))
        occ_vec = np.zeros(self.nspin)
        occ_vec[occ_indices] = 1.0
       
        output = [] 
        #print("olp_Deriv", sd, self.cache_wfn_olp)
        term = (1/2) * self._olp(sd)
        
        # Derivative with respect to coefficients of virtual variables
        result = occ_vec * term
        output.append(result)
        
        # Derivative with respect to hidden variables
        result = term * self.bath 
        output.append(result)
        

        # Derivative with respect to weights
        result = 1.0
        for i in range(self.nspin):
            for j in range(self.nbath):
                result = occ_vec[i] * self.bath[j]
                result = term * result
                output.append(result)
 
        return np.hstack(output)
        

    def get_overlap(self, sd, deriv=None):
        if deriv is None:
            return self._olp(sd)
        #print(deriv)
        return self._olp_deriv(sd)[deriv]
     

