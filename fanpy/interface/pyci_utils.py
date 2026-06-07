""" Adapter for Fanpy constraints"""

import numpy as np

from fanpy.eqn.base import BaseSchrodinger


class ConstraintAdapter:
    """ Constraints in Fanpy only take the wavefunction parameters as inputs, however PyCI passes the wfn parameters and the Energy.
    This class serves as an adapter for the energy and normalization constraints in Fanpy. 
    """

    def __init__(self, constraint):
        """ Constructor for Constraint Adapter 

        Parameters
        ----------
        constraint : BaseSchrodinger
            Constraint for an objective. E.g.: normalization or energy constraint. 
        """
        
        if not isinstance(constraint, BaseSchrodinger):
            raise TypeError("Constraint must be a child of the BaseSchrodinger class.")
        self.constraint = constraint

    def objective(self, x):
        """ Calculate the value for the constraint. This truncates x and passes it to the objective of the Fanpy constraint.

        Parameters
        ----------
        x : np.ndarray
            Input data for the constraint. x[:-1] corresponds to wfn parameters and x[-1] is the energy parameter. 

        Returns
        -------
        obj_value : float
            objective value from the constraint. 
        """

        return self.constraint.objective(x[:-1])
    
    def gradient(self, x):
        """ Calculate the gradient for the constraint. This truncates x and passes it to the objective of the Fanpy constraint.

        Parameters
        ----------
        x : np.ndarray
            Input data for the constraint. x[:-1] corresponds to wfn parameters and x[-1] is the energy parameter. 

        Returns
        -------
        grad : np.ndarray
            gradient of the constraint. This is padded with a 0 for the derivative w.r.t. the energy
        """

        wfn_grad = self.constraint.gradient(x[:-1]) # dim 1 x wfn params 
        e_grad = np.zeros(1) # gradient w.r.t. E is 0, since it does not enter the Fanpy constraint 
        grad = np.hstack((wfn_grad, e_grad))
        return grad


    