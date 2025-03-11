"""Restricted Boltzmann Machine wavefunction."""

from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction

import numpy as np


class RestrictedBoltzmannMachine(BaseWavefunction):
    def __init__(self, nelec, nspin, nbath, params=None, memory=None, num_layers=1, orders=(1, 2)):
        super().__init__(nelec, nspin, memory=memory)
        self.nbath = nbath
        self.orders = np.array(orders)
        self.num_layers = num_layers
        self._template_params = None
        self._template_sign_params = None
        # self.enable_cache(include_derivative=True)
        self.assign_sign_params(sign_params=None)
        self.assign_params(params=params)
        self.forward_cache_lin = []
        self.forward_cache_act = []
        # self.probable_sds = {}
        # self.olp_threshold = 42

    @property
    def params(self):
        return np.hstack([i.flat for i in self._params])

    # @params.setter
    # def params(self, val):
    #     self.assign_params(val)

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        # NOTE: first layer is K x B
        return np.sum(self.nbath * self.nspin**self.orders) + (self.num_layers - 1) * (
            self.nbath**2
        )  # + (self.nspin + 1) # including sign_params which are the size of npsin, and +1 for bias

        # NOTE: first layer is K x K and K x B
        # return (
        #     np.sum(self.nspin * self.nspin ** self.orders) +
        #     self.nspin * self.nbath +
        #     (self.num_layers - 1) * (self.nbath ** 2)
        # )

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        Notes
        -----
        Instance must have attribut `model`.

        """
        # NOTE: first layer is K x B
        return [(self.nbath, *(self.nspin,) * order) for order in self.orders] + (self.num_layers - 1) * [
            (self.nbath, self.nbath)
        ]
        # NOTE: first layer is K x K and K x B
        # return (
        #     [(self.nspin, *(self.nspin, ) * order) for order in self.orders] +
        #     [(self.nspin, self.nbath)] +
        #     (self.num_layers - 1) * [(self.nbath, self.nbath)]
        # )

    @property
    def template_sign_params(self):

        return self._template_sign_params

    @property
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Returns
        -------
        template_params : np.ndarray
            Default parameters of the wavefunction.

        Notes
        -----
        May depend on params_shape and other attributes/properties.

        """
        return self._template_params

    def assign_template_sign_params(self):
        # sign_params = np.ones(self.nspin + 1) * -0.001 #* 10  # +1 for bias
        sign_params = np.random.uniform(-1, -1, size=(self.nspin + 1))  # +1 for bias
        self._template_sign_params = sign_params

    def assign_template_params(self):
        r"""Assign the initial guess for the HF ground state wavefunction.

        Since the template parameters are calculated/approximated, they are computed and stored away
        rather than generating each one on the fly.

        Raises
        ------
        ValueError
            If any of the layers of the model has more than one type of weights. For example, bias
            is not allowed.
            If the number of units `K` in the final hidden layer is greater than
            :math:`1 + (K-N)N + \binom{K-N}{2} \binom{N}{2}`.

        Notes
        -----
        The template parameters can only be created for networks without bias and sufficiently large
        final hidden layer. Additionally, the produced parameters may not be a good initial guess
        for the HF ground state.

        """
        params = []
        # scale = 1 / np.tanh(1)
        scale = 1 / self.activation(0)

        for i, param_shape in enumerate(self.params_shape):
            params.append(np.zeros(param_shape))

            # For one layer input with different subsequent activation
            # if i < len(self.orders):
            #     params.append(np.zeros(param_shape))
            # else:
            #     params.append(np.eye(*param_shape) * scale)

            # For two layer input
            # elif i == self.orders:
            #     params.append(np.eye(param_shape))
            # else:
            #     params.append(np.eye(*param_shape) * scalle)

        # self.output_scale = scale
        self.output_scale = 0.5
        self._template_params = params

    def assign_sign_params(self, sign_params=None):
        if sign_params is None:
            if self._template_sign_params is None:
                self.assign_template_sign_params()
        self.sign_params = self._template_sign_params

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
        add_noise : bool
            Flag to add noise to the given parameters.

        Raises
        ------
        TypeError
            If `params` is not a numpy array.
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`.
            If `params` has complex data type and wavefunction has float data type.
        ValueError
            If `params` does not have the same shape as the template_params.

        Notes
        -----
        Depends on dtype, template_params, and nparams.

        """
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self.template_params
        if isinstance(params, np.ndarray):
            structured_params = []
            for param_shape in self.params_shape:
                structured_params.append(
                    params[: np.prod(param_shape)].reshape(*param_shape)
                )  # It's appending arrays in the list structured_params
                params = params[np.prod(param_shape) :]
            params = structured_params

        # store parameters
        self._params = params

        # append sign parameters
        if self._template_sign_params is None:
            self.assign_template_sign_params()
            self.assign_sign_params()
        self._params.append(self.sign_params)

        # print("self._params: ", self._params[2].shape)
        # print("\nself._params", self._params[0].shape, self._params[1].shape, self._params[2].shape, self._params[3].shape)

        # super().assign_params(params=params, add_noise=add_noise)

        self.clear_cache()

    @staticmethod
    def activation(x):
        # return np.tanh(x)
        return 1 + np.exp(x)
        # return 2 * np.cosh(x)

    @staticmethod
    def activation_deriv(x):
        # return 1 - np.tanh(x) ** 2
        return np.exp(x)
        # return 2 * np.sinh(x)

    @staticmethod
    def sign_correction(x):
        return np.tanh(x)

    @staticmethod
    def sign_correction_deriv(x):
        return 1 - np.tanh(x) ** 2

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally.

        Notes
        -----
        Overlaps and their derivatives are not cached.

        """
        # breakpoint()
        # if no derivatization
        if deriv is None:
            return self._olp(sd)
        return self._olp_deriv(sd)[deriv]

    def _olp(self, sd):
        output = np.prod(self._olp_helper(sd) * self.output_scale) * self._olp_sign_helper(sd)

        # if abs(output) > self.olp_threshold:
        #     self.probable_sds[sd] = output

        return output

    def _olp_sign_helper(self, sd):
        occ_indices = np.array(slater.occ_indices(sd))
        occ_mask = np.zeros(self.nspin)
        occ_mask[occ_indices] = 1
        sign_output = self.sign_correction(np.sum(self.sign_params[:-1] * occ_mask) + self.sign_params[-1])

        return sign_output

    def _olp_helper(self, sd, cache=False):
        """Return output of the network (before the product layer)."""
        occ_indices = np.array(slater.occ_indices(sd))
        occ_mask = np.zeros(self.nspin, dtype=bool)
        occ_mask[occ_indices] = True
        # output[np.array(slater.occ_indices(sd))] = 1

        if cache:
            self.forward_cache_act = []
            self.forward_cache_lin = []

        output = np.zeros(self.params_shape[0][0])
        # first layer
        for order, layer_params in zip(self.orders, self._params[:-1]):  # excluding the sign_params
            # NOTE: if you want to use a different operator, you would change the corresponding
            # input here
            # get input vector/tensor
            input_layer = np.zeros((self.nspin,) * order)
            mask = occ_mask
            for _ in range(order - 1):
                mask = occ_mask & np.expand_dims(mask, -1)
            input_layer[mask] = 1
            if cache:
                # FIXME: large memory usage
                # store
                self.forward_cache_act.append(input_layer.flatten())

            # contraction
            # NOTE: this assumes that the inputs are 0 or 1
            # for _ in range(order):
            #     layer_params = np.sum(layer_params[:, ..., occ_indices], axis=-1)
            # if cache:
            #     self.forward_cache_lin.append(layer_params)
            # output += layer_params

            # OR more generally (if inputs are not 0 or 1)
            next_layer = np.sum(layer_params * input_layer, axis=tuple(np.arange(1, layer_params.ndim)))
            output += next_layer

        if cache:
            for _ in self.orders:
                self.forward_cache_lin.append(output)

        output = self.activation(output)

        for layer_params in self._params[len(self.orders) : -1]:  # -1 to exclude sign_params
            if cache:
                self.forward_cache_act.append(output)
            output = layer_params.dot(output)
            if cache:
                self.forward_cache_lin.append(output)
            output = self.activation(output)
        return output

    def _olp_deriv(self, sd):
        grads = []
        # FIXME: hard coded
        dy_da_prev = np.identity(self.nbath)

        # load forward cache
        vals = self._olp_helper(sd, cache=True)

        for i, layer_params in enumerate(self._params[::-1]):
            i = len(self._params) - 1 - i
            dy_da_curr = dy_da_prev
            a_prev = self.forward_cache_act[i]
            z_curr = self.forward_cache_lin[i]
            w_curr = self._params[i]

            # z_curr_i = \sum_j w_curr_ij a_prev_j
            # a_curr_j = sigma(z_curr_i)
            da_curr_dz_curr = self.activation_deriv(z_curr)
            dy_dz_curr = dy_da_curr * da_curr_dz_curr

            dz_curr_dw_curr = a_prev[None, None, :]
            dy_dw_curr = dy_dz_curr[:, :, None] * dz_curr_dw_curr

            if i >= len(self.orders):
                dz_curr_da_prev = w_curr
                dy_da_prev = dy_dz_curr.dot(dz_curr_da_prev)

            grads.append(dy_dw_curr * self.output_scale)

        grads = grads[::-1]

        # get product of the network output that have not been derivatived
        indices = np.arange(self.nbath)
        indices = np.array([np.roll(indices, -i - 1, axis=0)[:-1] for i in indices])
        other_vals = np.prod(vals[indices] * self.output_scale, axis=1)

        output = []
        for grads_layer in grads[:-1]:
            output.append(np.sum(other_vals[:, None, None] * grads_layer, axis=0).flat)
        output.append(np.sum(other_vals[:, None] * grads[-1], axis=0).flatten())
        return np.hstack(output)

    def get_overlaps(self, sds, deriv=None):
        if len(sds) == 0:
            return np.array([])
        occ_indices = np.array([slater.occ_indices(sd) for sd in sds])

        # print("\nself._params.shape: ", self.params_shape)
        # print("\nself.nparams: ", self.nparams)

        # array of sign_params for sds
        # sign_params_sds = np.ones((len(sds), self.nspin + 1)) * -0.001 #* 10 #creating sign_params on fly using np.ones
        sign_params_sds = np.array([self._params[-1] for sd in sds])
        occ_mask_sds = np.zeros((len(sds), self.nspin))
        for i, inds in enumerate(occ_indices):
            occ_mask_sds[i, inds] = 1

        sign_output = self.sign_correction(
            (sign_params_sds[:, :-1] * occ_mask_sds).sum(axis=1) + sign_params_sds[:, -1]
        )

        vals = np.zeros((len(sds), self.nbath))
        vals = vals.T

        self.forward_cache_act = []
        self.forward_cache_lin = []

        occ_mask = np.zeros((len(sds), self.nspin), dtype=bool)
        # FIXME: breaks if inconsistent particle number
        # occ_mask[np.arange(len(sds))[:, None], occ_indices] = True
        # FIXME: but less efficient
        for i, inds in enumerate(occ_indices):
            occ_mask[i, inds] = True
        for order, layer_params in zip(self.orders, self._params[:-1]):  # -1 for excluding the sign_params
            # NOTE: if you want to use a different operator, you would change the corresponding
            # input here
            # get input vector/tensor
            input_layer = np.zeros((len(sds),) + (self.nspin,) * order)
            mask = occ_mask
            for i in range(order - 1):
                occ_mask = np.expand_dims(occ_mask, 1)
                mask = np.expand_dims(mask, -1) & occ_mask
            # NOTE: mask has the shape: number of sds * nspin * nspin * ...
            input_layer[mask] = 1
            # FIXME: larger memory usage
            self.forward_cache_act.append(input_layer.reshape(input_layer.shape[0], -1).T)

            # contraction
            # NOTE: layer_params has shape: nbath * nspin * nspin * ...
            # 1st dimension will be that of bath states
            # 2nd dimension will be that of the sds
            next_layer = np.sum(
                layer_params[:, None] * input_layer[None],
                axis=tuple(np.arange(2, layer_params.ndim + 1)),
            )
            vals += next_layer

        for _ in self.orders:
            self.forward_cache_lin.append(vals)

        vals = self.activation(vals)

        for layer_params in self._params[len(self.orders) : -1]:  # -1 excluding sign_params
            self.forward_cache_act.append(vals)
            vals = layer_params.dot(vals)
            self.forward_cache_lin.append(vals)
            vals = self.activation(vals)

        if deriv is None:
            output = np.prod(vals * self.output_scale, axis=0) * sign_output  # adding sign_correction
            return output

        # NOTE: Following code is for the case when deriv != None
        grads = []
        # last index is for the slater determinants
        dy_da_prev = np.identity(self.nbath)[:, :, None]

        # print("self._params[1::-1]:", self._params[1::-1])
        # starting self._params from 1 to exclude sign_params at index 0
        # for i, layer_params in enumerate(self._params[1::-1]):
        for i, layer_params in enumerate(self._params[self.num_layers :: -1]):
            i = len(self._params[:-1]) - 1 - i
            dy_da_curr = dy_da_prev
            # print("index i:", i)
            a_prev = self.forward_cache_act[i]
            z_curr = self.forward_cache_lin[i]
            w_curr = self._params[i]

            # z_curr_i = \sum_j w_curr_ij a_prev_j
            # a_curr_j = sigma(z_curr_i)
            da_curr_dz_curr = self.activation_deriv(z_curr)
            dy_dz_curr = dy_da_curr * da_curr_dz_curr

            dz_curr_dw_curr = a_prev[None, None, :, :]
            dy_dw_curr = dy_dz_curr[:, :, None, :] * dz_curr_dw_curr

            if i >= len(self.orders):
                dz_curr_da_prev = w_curr
                dy_da_prev = np.tensordot(dy_dz_curr, dz_curr_da_prev, axes=(1, 0))
                dy_da_prev = np.swapaxes(dy_da_prev, 1, 2)

            grads.append(dy_dw_curr * self.output_scale)
        grads = grads[::-1]

        # get product of the network output that have not been derivatived
        indices = np.arange(self.nbath)
        indices = np.array([np.roll(indices, -i - 1, axis=0)[:-1] for i in indices])
        other_vals = np.prod(vals[indices] * self.output_scale, axis=1)

        output = []
        for grads_layer in grads[:-1]:
            output.append(
                np.moveaxis(np.sum(other_vals[:, None, None] * grads_layer, axis=0), 2, 0).reshape(len(sds), -1)
            )
        output.append(np.moveaxis(np.sum(other_vals[:, None, None, :] * grads[-1], axis=0), 2, 0).reshape(len(sds), -1))

        # print("\n\noutput:", output[0].shape, output[1].shape, "\n", output)

        # Adding sign_correction to derivative
        # NOTE: F_new = F_old * sign
        #     dF_new = [dF_old * sign, F_old * dsign]
        #            = [df_sign, f_disgn]

        # df_sign = np.hstack(output)[:, deriv[:-(self.nspin+1)]] * sign_output[:, np.newaxis]
        # sign_output_deriv = self.sign_correction_deriv(
        #                        (sign_params_sds[:, :-1] * occ_mask_sds).sum(axis=1) + sign_params_sds[:,-1])

        # f_dsign = np.prod(vals * self.output_scale, axis=0) * sign_output_deriv
        # f_dsign = f_dsign[:, np.newaxis] * occ_mask_sds #Multiplying by occupation vector
        # f_dsign = np.column_stack((f_dsign, np.zeros(len(sds)))) #Stacking column of zeros for derivative w.r.t sign_bias

        # Final output matrix for derivatives wrt wfn_params, sign_params and sign_bias
        # new_output = np.c_[df_sign, f_dsign]

        # print("\nlen(sds): ", len(sds))
        # print("\noutput w/o sign for deriv=None: ", np.prod(vals * self.output_scale, axis=0) , "\n" )#adding sign_correction
        # print("\nSign_output: ",len(sign_output), "\n", sign_output, "\n")
        # print("\nSign_output_deriv: ",len(sign_output_deriv), "\n", sign_output_deriv, "\n")
        # print("\nnew_output: ", new_output, "\n")

        # return  new_output
        return np.hstack(output)[:, deriv]

    def normalize(self, pspace=None):
        if pspace is not None:
            norm = sum(self.get_overlap(sd) ** 2 for sd in pspace)
            print(norm, sorted([abs(self.get_overlap(sd)) for sd in pspace], reverse=True)[:5], "norm")
        else:
            norm = sum(self.get_overlap(sd) ** 2 for sd in self.pspace_norm)
            print(norm, sorted([abs(self.get_overlap(sd)) for sd in self.pspace_norm], reverse=True)[:5], "norm")
        # norm = np.sum([olp ** 2 for olp in self.probable_sds.values()])
        # norm = max(self.probable_sds.values()) ** 2
        self.output_scale *= norm ** (-0.5 / self.params_shape[-1][0])
        self.clear_cache()
