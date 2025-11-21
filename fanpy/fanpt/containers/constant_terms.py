r"""Class that generates and contains the constant terms of the FANPT system of equations."""

from math import factorial
import numpy as np

from fanpy.fanpt.containers.base import FANPTContainer


class FANPTConstantTerms:
    r"""Generates and contains the constant terms of the FANPT system of equations.

    If the order is 1:
    -dG_n/dl

    If the energy is not an active parameter:
    -N * sum_k {d^2G_n/dldp_k * d^(N-1)p_k/dl^(N-1)}

    If the energy is an active parameter:
    linear_term = t1 + t2
    t1 = -N * sum_k {d^2G_n/dldp_k * d^(N-1)p_k/dl^(N-1)}
    t2 = - sum_k{d^2G_n/dEdp_k * sum_m{C(N,m) * d^mE/dl^m * d^(N-m)p_k/dl^(N-m)}}

    If `quasi_approximation_order == 3`, additional second- and third-derivative
    contributions are included for orders 2 and 3 (when energy is active).

    Notation
    --------
    N: order
    sum_k: sum over all active wfn parameters.
    sum_m: sum from 1 to N-1.
    C(N,m): binomial coefficient.

    Attributes
    ----------
    fanpt_container : FANPTContainer
        Object containing the FANPT matrices and vectors.
    order : int
        Order of the current FANPT iteration.
    previous_responses : np.ndarray
        Previous responses of the FANPT calculations up to order = order - 1.
        Shape: (order-1, nactive)
    quasi_approximation_order : int
        Either 2 or 3. Controls inclusion of higher-derivative terms.

    Methods
    -------
    __init__(self, fanpt_container, order=1, previous_responses=None, quasi_approximation_order=2)
        Initialize the constant terms.
    assign_fanpt_container(self, fanpt_container):
        Assign the FANPT Container
    assign_order(self, order):
        Assign the order
    assign_previous_responses(self, previous_responses):
        Assign the previous responses
    assign_quasi_approximation_order(self, qao):
        Assign the quasi approximation order
    gen_constant_terms(self)
    """

    def __init__(self, fanpt_container, order=1, previous_responses=None, quasi_approximation_order=2):
        r"""Initialize the constant terms.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            Object containing the FANPT matrices and vectors.
        order : int, optional
            Order of the current FANPT iteration (default: 1).
        previous_responses : np.ndarray or None, optional
            Previous responses of the FANPT calculations up to order = order - 1.
            If provided, expected shape is (order - 1, nactive). Default: None.
        quasi_approximation_order : int, optional
            2 or 3. If 3, include extra second-/third-derivative contributions
            (when energy is active) as described below. Default: 2.
        """
        self.assign_fanpt_container(fanpt_container=fanpt_container)
        self.assign_order(order=order)
        self.assign_quasi_approximation_order(quasi_approximation_order)
        self.assign_previous_responses(previous_responses=previous_responses)
        self.gen_constant_terms()

    def assign_fanpt_container(self, fanpt_container):
        r"""Assign the FANPT container.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            Container with the matrices/vectors required to perform the FANPT calculation.

        Raises
        ------
        TypeError
            If fanpt_container is not a child of FANPTContainer.
        """
        if not isinstance(fanpt_container, FANPTContainer):
            raise TypeError("fanpt_container must be a child of FANPTContainer")
        self.fanpt_container = fanpt_container

    def assign_order(self, order):
        r"""Assign the order.

        Parameters
        ----------
        order : int
            Order of the current FANPT iteration.

        Raises
        ------
        TypeError
            If order is not an int.
        ValueError
            If order is negative.
        """
        if not isinstance(order, int):
            raise TypeError("order must be an integer.")
        if order < 0:
            raise ValueError("order must be non-negative")
        self.order = order

    def assign_quasi_approximation_order(self, qao):
        r"""Assign and validate the quasi approximation order.

        Parameters
        ----------
        qao : int
            Allowed values: 2 or 3.

        Raises
        ------
        ValueError
            If qao is not 2 or 3.
        """
        if qao is None:
            qao = 2

        if qao not in (2, 3):
            raise ValueError("quasi_approximation_order must be 2 or 3.")
        self.quasi_approximation_order = qao

    def assign_previous_responses(self, previous_responses):
        r"""Assign the previous responses.

        Parameters
        ----------
        previous_responses : np.ndarray
            Previous responses of the FANPT calculations up to order = order -1.

        Raises
        ------
        TypeError
            If previous_responses is not a numpy array.
            If the elements of previous_responses are not numpy arrays.
        ValueError
            If previous_responses is None and order is not 1.
            If the shape of previous_responses is not equal to (order - 1, nactive).
        """
        if self.order == 1:
            self.previous_responses = previous_responses
        else:
            if not isinstance(previous_responses, np.ndarray):
                raise TypeError("previous_responses must be a numpy array.")
            if not all([isinstance(response, np.ndarray) for response in previous_responses]):
                raise TypeError("The elements of previous_responses must be numpy arrays.")
            if previous_responses.shape != (self.order - 1, self.fanpt_container.nactive):
                raise ValueError(
                    "The shape of previous_responses must be ({}, {}).".format(
                        self.order - 1, self.fanpt_container.nactive
                    )
                )
            self.previous_responses = previous_responses

    def gen_constant_terms(self):
        r"""Generate the constant terms.

        Returns
        -------
        constant_terms : np.ndarray
            Constant terms of the FANPT linear system of equations.
            Shape: (nequation,)
        """
        if self.order == 1:
            constant_terms = -self.fanpt_container.d_g_lambda

        else:
            if self.fanpt_container.active_energy:
                # Base (order-agnostic) active-energy contributions
                r_vector = np.zeros(self.fanpt_container.nactive - 1)
                for o in range(1, self.order):
                    comb = factorial(self.order) / (factorial(o) * factorial(self.order - o))
                    r_vector += (
                        comb
                        * self.previous_responses[o - 1][-1]           # d^o E / dl^o
                        * self.previous_responses[self.order - o - 1][:-1]  # d^(N-o) p / dl^(N-o)
                    )

                constant_terms = (
                    -self.order
                    * np.dot(self.fanpt_container.d2_g_lambda_wfnparams,
                             self.previous_responses[-1][:-1])
                    - np.dot(self.fanpt_container.d2_g_e_wfnparams, r_vector)
                )

                # Extra contributions only if quasi_approximation_order == 3
                if self.quasi_approximation_order == 3:
                    if self.order == 2:
                        # Second-derivative contribution wrt p_k, p_l
                        constant_terms -= np.einsum(
                            "mkl,k,l->m",
                            self.fanpt_container.d2_g_wfnparams2,
                            self.previous_responses[0][:-1],
                            self.previous_responses[0][:-1],
                        )

                    elif self.order == 3:
                        # Second-derivative contribution wrt p_k, p_l
                        constant_terms -= 3.0 * np.einsum(
                            "mkl,k,l->m",
                            self.fanpt_container.d2_g_wfnparams2,
                            self.previous_responses[1][:-1],
                            self.previous_responses[0][:-1],
                        )
                        # Third-derivative contribution wrt energy, p_k, p_l
                        constant_terms -= (
                            3.0
                            * self.previous_responses[0][-1]
                            * np.einsum(
                                "mkl,k,l->m",
                                self.fanpt_container.d3_g_e_wfnparams2,
                                self.previous_responses[1][:-1],
                                self.previous_responses[0][:-1],
                            )
                        )
                        # Third-derivative contribution wrt lambda, p_k, p_l
                        constant_terms -= 3.0 * np.einsum(
                            "mkl,k,l->m",
                            self.fanpt_container.d3_g_lambda_wfnparams2,
                            self.previous_responses[0][:-1],
                            self.previous_responses[0][:-1],
                        )

            else:
                # Energy not active: unchanged
                constant_terms = -self.order * np.dot(
                    self.fanpt_container.d2_g_lambda_wfnparams,
                    self.previous_responses[-1],
                )

        self.constant_terms = constant_terms
        return constant_terms
