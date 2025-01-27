"""RBM with triple order as Paul tried."""

from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
import numpy as np
import random


class RestrictedBoltzmannMachine(BaseWavefunction):
    r"""Restricted Boltzmann Machine (RBM) type wavefunction  where only order=3 interactions
    between virtual orbitals are considered and no hidden layer is considered. Below is the math
    expression:

    f(\vec{a}, \vec{b}, c, \vec{n}) = \sqrt{\frac{exp(\sum_{ijk} a_{ijk} n_i n_j n_k)}{\sum_{\{n\}     exp(\sum_{ijk} a_{ijk} n_i n_j n_k)}}} tanh(\sum_k b_k n_k + c)

    , where \vec{n} is a occupation number vector.

    Using the probability distribution representation by RBM as a wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of total spin orbitals (including occupied and virtual, alpha and beta).
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

    def __init__(self, nelec, nspin, params=None, memory=None, orders=(3)):

        super().__init__(nelec, nspin, memory=memory)
        self.orders = np.array(orders)
        self._template_params = None
        self.assign_params(params=params)

    @property
    def params(self):
        return np.hstack([i.flat for i in self._params])

    @property
    def nparams(self):
        # return (int((self.nspin ** 2) * (self.nspin - 1) / 2) + (self.nspin +1)
        return np.sum(self.nspin**self.orders) + (self.nspin + 1)

    @property
    def nsign_params(self):
        # total sign params = sign_params_for_virtual +1 (bias)
        return self.nspin + 1

    @property
    def params_shape(self):
        # Coefficient matrix for interaction order = 3 with size (nspin, nspin, nspin)
        return [(self.nspin, self.nspin, self.nspin)] + [(self.nspin + 1,)]

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
        # print(self.orders, self.params_shape)

        for i, param_shape in enumerate(self.params_shape[:-1]):
            random_params = [(random.random() * 0.04) - 0.02 for _ in range(np.prod(param_shape))]
            random_params = np.array(random_params).reshape(param_shape)
            params.append(random_params)

        random_params = [(random.random() * 0.04) - 0.02 for _ in range(self.nsign_params)]
        params.append(np.array(random_params))

        self._template_params = params

    def assign_params(self, params=None, add_noise=False):
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self.template_params

        if isinstance(params, np.ndarray):
            structured_params = []
            for param_shape in self.params_shape:
                structured_params.append(params[: np.prod(param_shape)].reshape(*param_shape))
                params = params[np.prod(param_shape) :]
            params = structured_params

        self._params = params

    def get_overlaps(self, sds, deriv=None):
        if len(sds) == 0:
            return np.array([])

        occ_indices = np.array([slater.occ_indices(sd) for sd in sds])
        occ_mask = np.zeros((len(sds), self.nspin))
        for i, inds in enumerate(occ_indices):
            occ_mask[i, inds] = 1.0

        a = self._params[0]
        # a = a[np.triu_indices(self.nspin, k=1)].flatten()
        sign_params = self._params[-1]
        # print("\na", a, "\nb", b, "\nw", w, "\nsign_params", sign_params, "\n")
        # print("\nocc_mask:", occ_mask)
        # print("\nself.params:", self._params)
        # print("\nparams.shape a:", self._params[0].shape)
        # print("\nparams a:", a)
        # print("\nsign_params c, d:", self._params[-1])
        # print("\nsds: ", sds)

        olp_ = []
        wfn_only_olp = []
        sign_output = []
        exp_ninjnk = []

        for i in range(len(sds)):
            occ_vec = occ_mask[i]
            # print("\ni, occ_vec:", i, occ_vec)
            ninjnk = np.einsum("i,j,k->ijk", occ_vec, occ_vec, occ_vec)
            # ninjnk = ninjnk[np.triu_indices(self.nspin, k=1)].flatten()

            # Doing element-wise multiplication of arrays a and ninjnk and
            # taking sum of all resulting elements
            # That means, here we are considering all possible permutations in interactions
            # For example, for a 2x2 matrix, interactions ij and ji both are involved.

            # NOTE: If we want to avoid this doubling, we can use np.triu_indices
            # to extract elements of upper triangular matrix without including diagonal.
            # This idea needs to be thoroughly tested for a 3D array.

            sum_ = np.sum(a * ninjnk)
            numerator = np.exp(sum_)
            exp_ninjnk.append(numerator * ninjnk)

            wfn_only_olp.append(numerator)  # appending the numerator

            sign_input = np.sum(sign_params[:-1] * occ_vec) + sign_params[-1]
            sign_result = self.sign_correction(sign_input)
            # print("i, sign_input, sign_result:", i, sign_result)
            sign_output.append(sign_result)
            # print("\nsds, sign_correction", sds[i], sign_result, "\n")

        wfn_only_olp = np.array(wfn_only_olp)
        sign_output = np.array(sign_output)
        exp_ninjnk = np.array(exp_ninjnk)

        # denominator
        partition_func = np.sum(wfn_only_olp)

        if len(wfn_only_olp) == len(sign_output) == len(sds):
            f_wfn = np.sqrt(wfn_only_olp / partition_func)
            # print("\nroot P", f_wfn)
            olp_ = f_wfn * sign_output
            # print("\nfinal olp", olp_)

        if deriv is None:
            return np.array(olp_)

        ### If deriv is not None
        output = []

        for l in range(len(sds)):
            occ_vec = np.array(occ_mask[l])
            ninjnk = np.einsum("i,j,k->ijk", occ_vec, occ_vec, occ_vec)
            # ninjnk = ninjnk[np.triu_indices(self.nspin, k=1)].flatten()
            term = (partition_func * ninjnk - np.sum(exp_ninjnk, axis=0)) / partition_func
            df_da = 1.0 / 2.0 * olp_[l] * term
            # print(df_da.shape)

            sd_i = df_da.ravel()
            # print('len(sd_i)', len(sd_i))

            sign_input = np.sum(sign_params[:-1] * occ_vec) + sign_params[-1]
            sign_deriv = self.sign_correction_deriv(sign_input)

            # Derivative with respect to sign_params for virtual
            result = f_wfn[l] * sign_deriv * occ_vec
            sd_i = np.hstack((sd_i, result))

            # Derivative with respect to sign_params for bias
            result = f_wfn[l] * sign_deriv
            sd_i = np.hstack((sd_i, result))

            output.append(list(sd_i))
        output = np.array(output)
        # print("\na: ", a, "\nb: ",b, "\nw: ", w, "\nsign_params", sign_params)
        # print('\n\nlen(sds)', len(sds), '\n')
        # print("\noverlap", olp_)
        # print("\nolp derivative", output)
        return output[:, deriv]
