"""Copied from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb
"""
from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction

import numpy as np


class NumpyNetwork(BaseWavefunction):
    def __init__(self, nelec, nspin, params=None, memory=None, num_layers=2):
        super().__init__(nelec, nspin, memory=memory)
        self.num_layers = num_layers
        self._template_params = None
        self.enable_cache(include_derivative=True)
        self.assign_params(params=params)
        self.forward_cache_lin = []
        self.forward_cache_act = []
        self.probable_sds = {}
        self.olp_threshold = 42

    @property
    def params(self):
        return np.hstack([i.flat for i in self._params])

    @params.setter
    def params(self, val):
        self.assign_params(val)

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return (self.num_layers - 1) * (self.nspin ** 2) + self.nspin * self.nelec

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
        return (self.num_layers - 1) * [(self.nspin, self.nspin)] + [(self.nelec, self.nspin)]

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

    def assign_template_params(self):
        r"""Assign the intial guess for the HF ground state wavefunction.

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
        scale = 1 / np.tanh(1)
        if self.num_layers > 1:
            params.append(np.identity(self.nspin))
            for _ in range(self.num_layers - 2):
                params.append(np.identity(self.nspin) * scale)

        output_weights = np.zeros((self.nelec, self.nspin))
        npair = self.nelec // 2
        output_weights[np.arange(npair), np.arange(npair)] = 1
        output_weights[
            np.arange(npair, self.nelec), np.arange(self.nspatial, self.nspatial + npair)
        ] = 1
        if self.num_layers > 1:
            output_weights *= scale
        params.append(output_weights)

        self.output_scale = scale
        self._template_params = params

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
            for i, j in self.params_shape:
                structured_params.append(params[:i * j].reshape(i, j))
                params = params[i * j:]
            params = structured_params

        # store parameters
        self._params = params
        # super().assign_params(params=params, add_noise=add_noise)

        self.clear_cache()

    @staticmethod
    def activation(x):
        return np.tanh(x)

    @staticmethod
    def activation_deriv(x):
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
        # if no derivatization
        if deriv is None:
            return self._olp(sd)
        return self._olp_deriv(sd)

    def _olp(self, sd):
        output = np.prod(self._olp_helper(sd) * self.output_scale)

        if abs(output) > self.olp_threshold:
            self.probable_sds[sd] = output

        return output

    def _olp_helper(self, sd, cache=False):
        output = np.zeros(self.nspin)
        output[np.array(slater.occ_indices(sd))] = 1

        if cache:
            self.forward_cache_act = []
            self.forward_cache_lin = []
        for layer_params in self._params:
            if cache:
                self.forward_cache_act.append(output)
            output = layer_params.dot(output)
            if cache:
                self.forward_cache_lin.append(output)
            output = np.tanh(output)
        return output

    def _olp_deriv(self, sd):
        grads = []
        dy_da_prev = np.identity(self.nelec)

        # load forward cache
        vals = self._olp_helper(sd, cache=True)

        for i, layer_params in enumerate(self._params[::-1]):
            i = self.num_layers - 1 - i
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

            dz_curr_da_prev = w_curr
            dy_da_prev = dy_dz_curr.dot(dz_curr_da_prev)

            grads.append(dy_dw_curr * self.output_scale)
        grads = grads[::-1]

        output = np.zeros(self.nparams)

        # get product of the network output that have not been derivatived
        indices = np.arange(self.nelec)
        indices = np.array([np.roll(indices, - i - 1, axis=0)[:-1] for i in indices])
        other_vals = np.prod(vals[indices] * self.output_scale, axis=1)

        output = []
        for grads_layer in grads[:-1]:
            output.append(np.sum(other_vals[:, None, None] * grads_layer, axis=0).flat)
        output.append(np.sum(other_vals[:, None] * grads[-1], axis=0).flat)
        return np.hstack(output)

    def get_overlaps(self, sds, deriv=None):
        if len(sds) == 0:
            return np.array([])
        vals = np.zeros((len(sds), self.nspin))
        vals[
            np.arange(len(sds))[:, None], np.array([slater.occ_indices(sd) for sd in sds])
        ] = 1
        vals = vals.T

        self.forward_cache_act = []
        self.forward_cache_lin = []
        for layer_params in self._params:
            self.forward_cache_act.append(vals)
            vals = layer_params.dot(vals)
            self.forward_cache_lin.append(vals)
            vals = np.tanh(vals)

        if deriv is None:
            return np.prod(vals * self.output_scale, axis=0)

        grads = []
        dy_da_prev = np.identity(self.nelec)[:, :, None]

        for i, layer_params in enumerate(self._params[::-1]):
            i = self.num_layers - 1 - i
            dy_da_curr = dy_da_prev
            a_prev = self.forward_cache_act[i]
            z_curr = self.forward_cache_lin[i]
            w_curr = self._params[i]

            # z_curr_i = \sum_j w_curr_ij a_prev_j
            # a_curr_j = sigma(z_curr_i)
            da_curr_dz_curr = self.activation_deriv(z_curr)
            dy_dz_curr = dy_da_curr * da_curr_dz_curr

            dz_curr_dw_curr = a_prev[None, None, :, :]
            dy_dw_curr = dy_dz_curr[:, :, None, :] * dz_curr_dw_curr

            dz_curr_da_prev = w_curr
            dy_da_prev = np.tensordot(dy_dz_curr, dz_curr_da_prev, axes=(1, 0))
            dy_da_prev = np.swapaxes(dy_da_prev, 1, 2)

            grads.append(dy_dw_curr * self.output_scale)
        grads = grads[::-1]

        output = np.zeros((len(sds), self.nparams))

        # get product of the network output that have not been derivatived
        indices = np.arange(self.nelec)
        indices = np.array([np.roll(indices, - i - 1, axis=0)[:-1] for i in indices])
        other_vals = np.prod(vals[indices] * self.output_scale, axis=1)

        output = []
        for grads_layer in grads[:-1]:
            output.append(
                np.rollaxis(
                    np.sum(other_vals[:, None, None] * grads_layer, axis=0), 2
                ).reshape(len(sds), -1)
            )
        output.append(
            np.rollaxis(np.sum(other_vals[:, None, None, :] * grads[-1], axis=0), 2).reshape(len(sds), -1)
        )
        return np.hstack(output)

    def normalize(self, pspace=None):
        if pspace is not None:
            norm = sum(self.get_overlap(sd)**2 for sd in pspace)
            print(
                norm,
                sorted([abs(self.get_overlap(sd)) for sd in pspace], reverse=True)[:5],
                'norm'
            )
        else:
            norm = sum(self.get_overlap(sd)**2 for sd in self.pspace_norm)
            print(
                norm,
                sorted([abs(self.get_overlap(sd)) for sd in self.pspace_norm], reverse=True)[:5],
                'norm'
            )
        # norm = np.sum([olp ** 2 for olp in self.probable_sds.values()])
        # norm = max(self.probable_sds.values()) ** 2
        self.output_scale *= norm ** (-0.5 / self.params_shape[-1][0])
        self.clear_cache()

    def update_pspace_norm(self, refwfn=None):
        if refwfn:
            self.pspace_norm = refwfn
        else:
            # self.pspace_norm = list(self.probable_sds.keys())
            self.pspace_norm.add(max(self.probable_sds.keys(), key=lambda x: self.probable_sds[x]))
            # max_sd = max(self.probable_sds.keys(), key=lambda x: self.probable_sds[x])
            # if self.probable_sds[max_sd] > self.get_overlap(self.probable_sds.pop()):
            #     self.pspace_norm = [max_sd]
        print('Adapt normalization pspace')
        print(
            sum(self.get_overlap(sd)**2 for sd in self.pspace_norm),
            len(self.probable_sds), len(self.pspace_norm), max(self.probable_sds.values()),
            'norm_pspace'
        )
