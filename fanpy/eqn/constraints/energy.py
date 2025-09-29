from collections import deque
import numpy as np
from fanpy.eqn.energy_oneside import EnergyOneSideProjection


class EnergyConstraint(EnergyOneSideProjection):
    """Energy constraint.
    This class implements an energy constraint based on a reference energy.
    The objective function is the difference between the calculated energy and the reference energy.
    The prupose of this "constraint" is to keep a downward pressure on the energy, and avoid getting stuck in local minima.
    

    Attributes
    ----------
    ref_energy : float

        Reference energy.
    energy_diff_history : deque
        History of energy differences.
    queue_size : int
        Number of energy differences to keep in history.
    base : float
        Base for logarithm to determine if energy difference is changing significantly.
    min_diff : float
        Minimum energy difference to consider for adjusting reference energy.
    energy_variable : Variable
        Variable representing the energy, if any.
    simple : bool
        If True, the objective function is simply the energy difference without any adjustments.
    
    Methods
    -------
    objective(params)
        Calculate the objective function value based on the energy difference and adjust reference energy if needed.
    """

    def __init__(
        self,
        wfn,
        ham,
        tmpfile="",
        param_selection=None,
        refwfn=None,
        ref_energy=-100.0,
        queue_size=4,
        base=np.e,
        min_diff=1e-1,
        simple=False,
    ):
        """Initialize EnergyConstraint instance.

        Parameters
        ----------
        wfn : Wavefunction
            Wavefunction to be optimized.
        ham : Hamiltonian
            Hamiltonian of the system.
        tmpfile : str, optional
            Temporary file for storing intermediate results. Default is "".
        param_selection : list, optional
            List of parameter indices to optimize. Default is None, which means all parameters are optimized.
        refwfn : Wavefunction, optional
            Reference wavefunction for overlap calculation. Default is None.
        ref_energy : float, optional
            Reference energy for the constraint. Default is -100.0.
        queue_size : int, optional
            Number of energy differences to keep in history. Default is 4.
        base : float, optional
            Base for logarithm to determine if energy difference is changing significantly. Default is np.e.
        min_diff : float, optional
            Minimum energy difference to consider for adjusting reference energy. Default is 1e-1.
        simple : bool, optional
            If True, the objective function is simply the energy difference without any adjustments. Default is False
        """

        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection, refwfn=refwfn)
        self.assign_refwfn(refwfn)
        self.ref_energy = ref_energy
        self.energy_diff_history = deque([])
        self.queue_size = queue_size
        self.base = base
        self.min_diff = min_diff
        self.simple = simple

    def objective(self, params):
        """Calculate the difference between calculated energy and reference energy. The reference energy is adjusted if needed.
        The reference energy is adjusted if:

        1. The calculated energy is lower than the reference energy.
        2. The energy difference does not change significantly over a number of calls.

        Parameters
        ----------
        params : np.ndarray
            Parameters for the wavefunction.

        Returns
        -------
        float
            Difference between calculated energy and reference energy.
        """

        energy = super().objective(params)
        energy_diff = energy - self.ref_energy
        if self.simple:
            return energy_diff

        self.energy_diff_history.append(energy_diff)
        if len(self.energy_diff_history) > self.queue_size:
            self.energy_diff_history.popleft()

        # if calculated energy is lower than the reference, bring down reference energy
        if energy_diff <= 0:
            print("Energy lower than reference. Adjusting reference energy: {}" "".format(self.ref_energy))
            self.ref_energy += self.base * energy_diff
            return energy_diff

        if len(self.energy_diff_history) != self.queue_size or any(i <= 0 for i in self.energy_diff_history):
            return energy_diff

        energy_diff_order = np.log(self.energy_diff_history) / np.log(self.base)
        if energy_diff_order[0] < np.log(self.min_diff) / np.log(self.base):
            return energy_diff
        # if energy difference does not change significantly with "many" calls, adjust ref_energy
        # if energy differences are all within one order of magnitude of each other
        # keep adjusting reference until the energy difference is not within one order of magnitude
        if np.all(
            np.logical_and(
                energy_diff_order[0] - 1 < energy_diff_order,
                energy_diff_order < energy_diff_order[0] + 1,
            )
        ):
            # bring reference closer (i.e. decrease energy difference)
            self.ref_energy = energy - self.base ** (energy_diff_order[0] - 1)
            print(
                "Changes to energy is much smaller than the reference energy. "
                "Adjusting reference energy: {}".format(self.ref_energy)
            )
            self.energy_diff_history = deque([])

        return energy_diff
