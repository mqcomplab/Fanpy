from fanpy.eqn.constraints.energy import EnergyConstraint
from fanpy.wfn.base import BaseWavefunction
from fanpy.ham.base import BaseHamiltonian
import numpy as np


class DummyWavefunction(BaseWavefunction):
    """A dummy wavefunction for testing purposes."""

    def __init__(self, nelec, nspin):
        super().__init__(nelec, nspin)
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
    """Test the energy constraint in simple mode. The reference energy does not change."""
    wfn = DummyWavefunction(4, 8)
    ham = DummyHamiltonian(8)
    const = EnergyConstraint(wfn, ham, ref_energy=-1.0, simple=True)
    # E_calc = -0.5, E_ref = -1.0, so difference is 0.5 = (-0.5 - -1.0)
    assert const.objective(wfn.params) == 0.5
    const.ref_energy = 0.0
    assert const.objective(wfn.params) == -0.5 
    assert const.ref_energy == 0.0 # ref energy should not change in simple mode

def test_energy_constraint_dynamic():
    """Test the energy constraint in dynamic mode. The reference energy changes 
    if the calculated energy is lower than the reference or if the energy 
    difference does not change significantly.    
    """
    wfn = DummyWavefunction(4, 8)
    ham = DummyHamiltonian(8)
    # change ref E if diff < 0
    const = EnergyConstraint(wfn, ham, ref_energy=0.0, base=10, simple=False)
    assert np.allclose(const.objective(wfn.params), -0.5) # check Ediff
    assert np.allclose(const.ref_energy, -5.0) # E_ref,n+1 = E_ref,n + base * diff 
    assert len(const.energy_diff_history) == 1 # check if ediff history is updated

    # change ref E if diff does not change significantly
    const = EnergyConstraint(wfn, ham, ref_energy=-1.0, base=10, simple=False)
    for _ in range(4):
        const.objective(wfn.params)
    assert -1.0 != const.ref_energy # ref energy should change
    assert np.allclose(const.ref_energy, -0.55) # E_ref,n+1 = E - diff / base
    # check if ediff history is reset
    assert len(const.energy_diff_history) == 0
