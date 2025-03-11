from collections import deque
import numpy as np
from fanpy.eqn.energy_oneside import EnergyOneSideProjection


class EnergyConstraint(EnergyOneSideProjection):
    # def __init__(self, wfn, ham, tmpfile="", param_selection=None, refwfn=None, ref_energy=-100.0):
    #     super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection)
    #     self.assign_refwfn(refwfn)
    #     self.ref_energy = ref_energy

    # def objective(self, params):
    #     energy = super().objective(params)
    #     energy_diff = energy - self.ref_energy

    #     # if calculated energy is lower than the reference, bring down reference energy
    #     if energy_diff <= 0:
    #         print("Computed energy below reference. Adjusting reference down.")
    #         self.ref_energy -= 2 * abs(energy_diff)

    #     return energy_diff
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
        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection, refwfn=refwfn)
        self.assign_refwfn(refwfn)
        self.ref_energy = ref_energy
        self.energy_diff_history = deque([])
        self.queue_size = queue_size
        self.base = base
        self.min_diff = min_diff
        self.energy_variable = None
        self.simple = simple

    def objective(self, params):
        if self.energy_variable:
            energy = self.energy_variable.params[0]
        else:
            energy = super().objective(params)
        energy_diff = energy - self.ref_energy
        if self.simple:
            return energy_diff

        # if calculated energy is lower than the reference, bring down reference energy
        if energy_diff <= 0:
            print("Energy lower than reference. Adjusting reference energy: {}" "".format(self.ref_energy))
            self.ref_energy += self.base * energy_diff
            return energy_diff

        self.energy_diff_history.append(energy_diff)
        if len(self.energy_diff_history) > self.queue_size:
            self.energy_diff_history.popleft()

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

        # if all(abs(energy - i) < 1e-3 for i in self.energy_history):
        #     self.ref_energy = energy - 1e-3

        return energy_diff
