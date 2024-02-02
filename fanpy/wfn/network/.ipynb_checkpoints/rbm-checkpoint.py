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

 
    @staticmethod
    def sign_correction(x):
        return np.tanh(x)

    @staticmethod
    def sign_correction_deriv(x):
        return 1 - np.tanh(x) ** 2


    def assign_template_params(self):
        params = []

        for i, param_shape in enumerate(self.params_shape[:-1]):
            #random_params = [(random.random() * 1.0) - 0.5 for _ in range(np.prod(param_shape))]
            #random_params = [(random.random() * 5.0) - 2.5 for _ in range(np.prod(param_shape))]
        #    random_params = np.array(random_params).reshape(param_shape) 
        #    params.append(random_params) 

            #params.append(np.zeros(param_shape))
            params.append(np.ones(param_shape) * -47)

        #random_params = [(random.random() * 1.0) - 0.5 for _ in range(self.nsign_params)]
        #random_params = [random.choice([-50,50]) for _ in range(self.nsign_params)]
        #sign_params = np.array(random_params) 
        #params.append(sign_params)
        params.append(np.ones(param_shape) * -47)
        '''
        a = [0.41267033,  0.08177125, -0.36411596, -0.51247041,  0.04015036,  0.10848756,
            -0.14865894,  0.23084796, -0.0118374,  -0.61422139, -0.49070759,  0.28048365] 
        b = [-0.15253534, -0.00074402, -0.08189891, -0.25649102, -0.16409671,  0.15844061,
            0.27836968, 0.40460482, 0.33343838, -0.01789289, -0.27384057,  0.25290587,
            -0.30622803,  0.35365532, -0.43490145, -0.3809053,   0.02262462, -0.36373953,
            0.03318439,  0.38259665,  0.34487478,  0.34061942, -0.47706264, -0.27897505] 
        w = [[-1.52012198e-01,	0.560253061,	0.416830754,	0.273875252,
            -0.882570119,	-0.000120887033,	-0.615481073,	0.0735643818,
            -0.497256432,	0.547121856,	-0.901060842,	0.117413726,
            0.260016341,	0.975882998,	-0.441168824,	0.834876429,
            0.484923183,	0.0238339015,	0.136166596,	-0.127389986,
            -0.0446637556,	-0.0156208648,	0.111472793,	7.88037713e-01],
            [-2.11703604e-01,	0.291221648,	0.244709904,	-0.0573494919,
            -0.645313759,	0.2473628,	0.124475918,	0.0205614309,
            -0.51951951,	-0.242304104,	-0.305915596,	-0.266811057,
            0.138950819,	0.500676312,	0.214732752,	-0.0316237786,
            0.588826428,	-0.328801488,	0.0893274753,	0.0415667284,
            0.223929089,	0.73377,	0.464153196,	1.21338654e-01],
            [0.15623113,	-0.113359892,	0.262886634,	-0.111425529,
            -0.243518936,	-0.119995314,	0.227095191,	-0.307755512,
            0.00381957034,	-0.215128246,	0.450696034,	0.459910123,
            -0.57843946,	-0.226824317,	0.437393366,	0.508180736,
            0.511706339,	0.446307043,	-0.180610762,	0.356913074,
            -0.112221389,	-0.153898815,	-0.0502656894,	2.61511038e-01],
            [0.244254325,	0.198826908,	-0.422511215,	0.169816901,
            0.453388986,	-0.367018537,	-0.028457944,	-0.277336819,
            0.392695298,	0.39159588,	-0.184747212,	0.290387972,
            -0.0939686623,	0.287538497,	0.129714736,	0.0495721931,
            -0.479221457,	0.583701109,	0.0863227975,	0.0348314212,
            0.384058184,	-0.036374591,	-0.124148696,	-2.49957422e-01],
            [0.266488367,	0.256141284,	0.0655417651,	-0.307502307,
            0.26630622,	-0.64112878,	0.0915395323,	0.034660039,
            -0.15498981,	-0.503018229,	-0.300492979,	-0.176826312,
            -0.175999472,	-0.0353132026,	-0.0449104024,	-0.112780248,
            -0.539976312,	-0.053746837,	-0.443046286,	-0.084663693,
            0.262229837,	0.0650815409,	0.0526217538,	-2.59379717e-02],
            [0.647509033,	-0.434242323,	-0.423901321,	-0.384025907,
            0.800317477,	-0.853385518,	0.0793306829,	-0.0525160445,
            0.483032554,	-0.0183510719,	-0.127100461,	-0.11445338,
            -0.0847498667,	-0.0198040666,	0.179434542,	-0.757392718,
            -0.0281444174,	0.0245192749,	0.00480685955,	0.704253991,
            0.160425262,	-0.507358665,	-0.245157865,	-4.32908448e-02],
            [-3.35510579e-01,	-0.179646294,	0.565798219,	-0.123456726,
            -0.349433845,	0.564562186,	-0.0114851127,	-0.406240594,
            -0.158240141,	-0.26916864,	-0.426826029,	0.615418201,
            0.107788213,	0.341119923,	-0.29707372,	0.351203255,
            0.627123159,	-0.361081883,	0.600439054,	0.245112293,
            0.214952213,	0.597194323,	-0.0283719687,	6.06797326e-02],
            [-6.00722866e-01,	0.57741802,	0.669457474,	0.163839984,
            -0.520446034,	0.725939345,	-0.0280198189,	-0.0482944202,
            0.210179725,	-0.0250741106,	-0.256722732,	-0.0804914346,
            0.409978968,	0.594202604,	-0.211297326,	0.639520621,
            -0.051369373,	-0.392919773,	0.663808987,	0.194817365,
            0.000932507496,	0.0371936078,	0.168887275,	3.99216210e-01],
            [-2.26322602e-01,	-0.0925277235,	0.243906402,	-0.537195688,
            0.603811914,	-0.0975906711,	0.190802849,	0.434459969,
            -0.230018961,	0.0701283178,	0.0669676842,	-0.406125976,
            0.0839169994,	-0.316685868,	0.312751958,	0.365155274,
            -0.278641944,	0.333150183,	-0.40015614,	0.250279307,
            -0.521919978,	-0.310816674,	-0.0595876294,	-1.40066544e-01],
            [0.450191695,	-0.0677572692,	0.00774768186,	-0.00121015435,
            0.333850641,	0.0130129732,	0.152529642,	0.00380323562,
            0.176697243,	-0.398462607,	-0.401195485,	0.153051007,
            0.123762796,	-0.440276674,	0.198372696,	0.271763144,
            -0.038415401,	0.316855338,	-0.19883297,	0.141037492,
            0.136734534,	-0.296478717,	-0.12381174,	2.42005468e-01],
            [0.1986506,	0.0811603371,	-0.0410177041,	-0.34425245,
            0.275305576,	-0.25253397,	0.144652135,	0.372131644,
            -0.0611203893,	-0.0200351683,	0.430060736,	-0.325177182,
            0.380071257,	0.184084184,	-0.064157051,	0.10852327,
            0.249186079,	0.129687273,	-0.661085268,	0.165543195,
            0.292270669,	0.0365244128,	-0.399665714,	-8.52140519e-02],
            [-1.17390783e-01,	-0.064224451,	-0.574994399,	-0.599042974,
            -0.0984594771,	-0.22100967,	-0.126894409,	0.102138306,
            -0.103842177,	-0.440874574,	0.428616941,	-0.254519073,
            -0.133405378,	-0.566203834,	0.539127723,	-0.312524249,
            -0.336385179,	-0.197828745,	-0.00384904895,	-0.211963437,
            0.0897058343,	-0.0933456228,	-0.407209834,	-2.72832547e-04]]
        params.append(np.array(a))
        params.append(np.array(b))
        params.append(np.array(w))
        sign_params = [0.33230418,  0.40506995, -0.38564207,  0.04764798, -0.16420564, -0.1943137,
             0.09072551,  0.19199804,  0.04735237, -0.34819376, -0.53823759, -0.29617865, -0.33808151] 
        params.append(np.array(sign_params))
        ''' 
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
            #print("\ni, occ_vec:", i, occ_vec)
            sum_ = np.sum(a * occ_vec) + np.sum(b * self.bath) + np.sum(w * np.outer(occ_vec, self.bath))
            numerator =   np.exp(sum_) 
            # we are using only one list of hidden variables unlike the set of virtual varibles which 
            # constitutes len(sds) number of occupation vectors.  

            wfn_only_olp.append(numerator)  # appending the numerator
        
            sign_input = (np.sum(sign_params[:-1] * occ_vec) + sign_params[-1] )
            sign_result = self.sign_correction(sign_input)
            #print("i, sign_input, sign_result:", i, sign_result)
            sign_output.append(sign_result)
            #print("\nsds, sign_correction", sds[i], sign_result, "\n")

         
        wfn_only_olp = np.array(wfn_only_olp)
        partition_func = np.sum(wfn_only_olp) # denominator

        if len(wfn_only_olp) == len(sign_output) == len(sds):            
            f_wfn = np.sqrt(wfn_only_olp / partition_func)
            #print("\nroot P", f_wfn)
            olp_ = f_wfn * sign_output
            #print("\nfinal olp", olp_)

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
