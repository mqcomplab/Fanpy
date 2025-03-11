"""Variational energy that corresponds to the Schrodinger equation."""

from fanpy.eqn.energy_twoside import EnergyTwoSideProjection
from fanpy.tools import sd_list


class EnergyVariational(EnergyTwoSideProjection):
    r"""Variational energy that corresponds to the Schrodinger equation.

    .. math::

        E &= \frac{
          \left< \Psi \right|
            \sum_{\mathbf{m} \in S}
            \left| \mathbf{m} \middle> \middle< \mathbf{m} \right|
          \hat{H}
            \sum_{\mathbf{n} \in S}
            \left| \mathbf{n} \middle> \middle< \mathbf{n} \right|
          \left| \Psi \right>
        }{
          \left< \Psi \right|
            \sum_{\mathbf{m} \in S}
            \left| \mathbf{m} \middle> \middle< \mathbf{m} \right|
          \left| \Psi \right>
        }\\
        &= \frac{
          \sum_{\mathbf{m}, \mathbf{n} \in S}
          \left< \Psi \middle| \mathbf{m} \right>
          \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>
          \left< \mathbf{n} \middle| \Psi \right>
        }{
          \sum_{\mathbf{m} \in S}
          \left< \Psi \middle| \mathbf{m} \right>^2
        }

    where :math:`S` is the projection spaces.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : BaseHamiltonian
        Hamiltonian that defines the system under study.
    indices_component_params : ComponentParameterIndices
        Indices of the component parameters that are active in the objective.
    step_print : bool
        Option to print relevant information when the objective is evaluated.
    step_save : bool
        Option to save parameters when the objective is evaluated.
    tmpfile : str
        Name of the file that will store the parameters used by the objective method.
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    pspace : {list/tuple of int}
        Projection space applied to the wavefunction..

    Properties
    ----------
    indices_objective_params : dict
        Indices of the (active) objective parameters that corresponds to each component.
    all_params : np.ndarray
        All of the parameters associated with the objective.
    active_params : np.ndarray
        Parameters that are selected for optimization.
    active_nparams : int
        Number of active parameters in the objective.
    num_eqns : int
        Number of equations in the objective.

    Methods
    -------
    __init__(self, wfn, ham, param_selection=None, optimize_orbitals=False, tmpfile="", pspace=None)
        Initialize the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or hamiltonian.
    save_params(self)
        Save all of the parameters in the `param_selection` to the temporary file.
    wrapped_get_overlap(self, sd, deriv=False)
        Wrap `get_overlap` to be derivatized with respect to the (active) parameters of the
        objective.
    wrapped_integrate_sd_wfn(self, sd, deriv=False)
        Wrap `integrate_sd_wfn` to be derivatized wrt the (active) parameters of the objective.
    wrapped_integrate_sd_sd(self, sd1, sd2, deriv=False)
        Wrap `integrate_sd_sd` to be derivatized wrt the (active) parameters of the objective.
    get_energy_one_proj(self, refwfn, deriv=False)
        Return the energy with respect to a reference wavefunction.
    get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=False)
        Return the energy after projecting out both sides.
    objective(self, params)
        Return the energy after inserting projection operators.
    gradient(self, params)
        Return the gradient of the energy after inserting projection operators.

    """

    # pylint: disable=W0223
    def __init__(
        self,
        wfn,
        ham,
        param_selection=None,
        optimize_orbitals=False,
        step_print=True,
        step_save=True,
        tmpfile="",
        pspace=None,
    ):
        r"""Initialize the objective instance.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction.
        ham : BaseHamiltonian
            Hamiltonian that defines the system under study.
        param_selection : tuple/list of 2-tuple/list
            Selection of the parameters that will be used in the objective.
            First element of each entry is a component of the objective: a wavefunction,
            Hamiltonian, or `ParamContainer` instance.
            Second element of each entry is a numpy index array (boolean or indices) that will
            select the parameters from each component that will be used in the objective.
            Default selects the wavefunction parameters.
        optimize_orbitals : bool
            Option to optimize orbitals.
            If Hamiltonian parameters are not selected, all of the orbital optimization parameters
            are optimized.
            If Hamiltonian parameters are selected, then only optimize the selected parameters.
            Default is no orbital optimization.
        step_print : bool
            Option to print relevant information when the objective is evaluated.
            Default is True.
        step_save : bool
            Option to save parameters with every evaluation of the objective.
            Default is True
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.
        pspace : {tuple/list of int, None}
            Projection space in terms of which the wavefunction is expressed.
            By default, the largest projection space is used.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If tmpfile is not a string.
        ValueError
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.

        """
        super().__init__(
            wfn,
            ham,
            param_selection=param_selection,
            optimize_orbitals=optimize_orbitals,
            step_print=step_print,
            step_save=step_save,
            tmpfile=tmpfile,
        )
        self.assign_pspace(pspace)

    def assign_pspace(self, pspace=None):
        r"""Assign the projection space.

        Parameters
        ----------
        pspace : {tuple/list of int, None}
            Projection space with respect to which the wavefunction is expressed.
            By default, the largest space is used.

        Raises
        ------
        TypeError
            If projection space is not a list or a tuple.
            If state in projection space is not given as a Slater determinant or a CI wavefunction.
        ValueError
            If given state in projection space does not have the same number of electrons as the
            wavefunction.
            If given state in projection space does not have the same number of spin orbitals as the
            wavefunction.

        """
        if pspace is None:
            pspace = sd_list.sd_list(self.wfn.nelec, self.wfn.nspin, spin=self.wfn.spin, seniority=self.wfn.seniority)
        super().assign_pspaces(pspace, pspace, pspace)
