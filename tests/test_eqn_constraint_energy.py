from fanpy.eqn.constraints.energy import EnergyConstraint
from fanpy.wfn.base import BaseWavefunction
from fanpy.ham.base import BaseHamiltonian
import numpy as np


class DummyWavefunction(BaseWavefunction):
    """A dummy wavefunction for testing purposes."""

    def __init__(self, nspin, nelec):
        super().__init__(nspin, nelec)
        self.params = np.array([0.0])

    def assign_params(self, params):
        self.params = params

    def get_overlap(self, sd, deriv=False):
        return 1.0 if not deriv else (1.0, np.array([0.0]))

class DummyHamiltonian(BaseHamiltonian):
    """A dummy Hamiltonian for testing purposes."""

    def __init__(self, nspin):
        self._nspin = nspin
    
    @property
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        return self._nspin
    def integrate_sd_wfn(self, sd, wfn, deriv=False):
        return -0.5 if not deriv else (-0.5, np.array([0.0]))
    

def test_energy_constraint_simple():
    wfn = DummyWavefunction(4, 8)
    ham = DummyHamiltonian(8)
    const = EnergyConstraint(wfn, ham, ref_energy=-1.0, simple=True)
    assert const.objective(wfn.params) == 0.5 # difference is 0.5 (-0.5 - -1.0)
    const.ref_energy = 0.0
    assert const.objective(wfn.params) == -0.5 
    assert const.ref_energy == 0.0 # ref energy should not change in simple mode