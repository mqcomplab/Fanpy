"""Script for running calculations."""
from fanpy.scripts.gaussian.make_script import make_script
from fanpy.scripts.utils import parser


def run_calc(
    nelec,
    one_int_file,
    two_int_file,
    wfn_type,
    nuc_nuc=None,
    optimize_orbs=False,
    pspace_exc=None,
    objective=None,
    solver=None,
    solver_kwargs=None,
    wfn_kwargs=None,
    ham_noise=None,
    wfn_noise=None,
    load_orbs=None,
    load_ham=None,
    load_ham_um=None,
    load_wfn=None,
    save_chk="",
    memory=None,
):
    """Script for running some basic calculations.

    Parameters
    ----------
    nelec : int
        Number of electrons.
    one_int_file : str
        Path to the one electron integrals (for restricted orbitals).
        One electron integrals should be stored as a numpy array of dimension (nspin/2, nspin/2).
    two_int_file : str
        Path to the two electron integrals (for restricted orbitals).
        Two electron integrals should be stored as a numpy array of dimension
        (nspin/2, nspin/2, nspin/2, nspin/2).
    wfn_type : str
        Type of wavefunction.
        One of `ci_pairs`, `cisd`, `fci`, `doci`, `mps`, `determinant-ratio`, `ap1rog`, `apr2g`,
        `apig`, `apsetg`, or `apg`.
    nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    optimize_orbs : bool
        If True, orbitals are optimized.
        If False, orbitals are not optimized.
        By default, orbitals are not optimized.
        Not compatible with solvers that require a gradient (everything except cma).
    pspace_exc : list of int
        Orders of excitations that will be used to build the projection space.
    objective : str
        Form of the Schrodinger equation that will be solved.
        Use `projected` to solve the Schrodinger equation as a system of equations.
        Use `least_squares` to solve the Schrodinger equation as a squared sum of the system of
        equations.
        Use `variational` to solve the energy variationally.
        Use `one_energy` to solve the energy projected on one side..
        Must be one of `projected`, `least_squares`, `variational`, or `one_energy`.
    solver : str
        Solver that will be used to solve the Schrodinger equation.
        Keyword `cma` uses Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).
        Keyword `diag` results in diagonalizing the CI matrix.
        Keyword `minimize` uses the BFGS algorithm.
        Keyword `least_squares` uses the Trust Region Reflective Algorithm.
        Keyword `root` uses the MINPACK hybrd routine.
        Must be one of `cma`, `diag`, `least_squares`, or `root`.
        Must be compatible with the objective.
    solver_kwargs : {str, None}
        Keyword arguments for the solver.
    wfn_kwargs : {str, None}
        Keyword arguments for the wavefunction.
    ham_noise : float
        Scale of the noise to be applied to the Hamiltonian parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    wfn_noise : bool
        Scale of the noise to be applied to the wavefunction parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    load_orbs : str
        Numpy file of the orbital transformation matrix that will be applied to the initial
        Hamiltonian.
        If the initial Hamiltonian parameters are provided, the orbitals will be transformed
        afterwards.
    load_ham : str
        Numpy file of the Hamiltonian parameters that will overwrite the parameters of the initial
        Hamiltonian.
    load_ham_um : str
        Numpy file of the Hamiltonian parameters that will overwrite the unitary matrix of the
        initial Hamiltonian.
    load_wfn : str
        Numpy file of the wavefunction parameters that will overwrite the parameters of the initial
        wavefunction.
    save_chk : str
        Name of the Numpy file that will store the chkpoint of the objective.
    memory : None
        Memory available to run calculations.

    """
    # make script
    script = make_script(
        nelec,
        one_int_file,
        two_int_file,
        wfn_type,
        nuc_nuc=nuc_nuc,
        optimize_orbs=optimize_orbs,
        pspace_exc=pspace_exc,
        objective=objective,
        solver=solver,
        solver_kwargs=solver_kwargs,
        wfn_kwargs=wfn_kwargs,
        ham_noise=ham_noise,
        wfn_noise=wfn_noise,
        load_orbs=load_orbs,
        load_ham=load_ham,
        load_ham_um=load_ham_um,
        load_wfn=load_wfn,
        save_chk=save_chk,
        filename=-1,
        memory=memory,
    )
    # run script
    # NOTE: Since the script is entirely generated from make_script, it should be more difficult to
    # inject something into the executable. (hopefully)
    exec(script)  # nosec: B102 # pylint: disable=W0122


def main():
    """Run script for run_calc using arguments obtained via argparse."""
    parser.description = "Optimize a wavefunction and/or Hamiltonian."
    args = parser.parse_args()
    run_calc(
        args.nelec,
        args.one_int_file,
        args.two_int_file,
        args.wfn_type,
        nuc_nuc=args.nuc_nuc,
        optimize_orbs=args.optimize_orbs,
        pspace_exc=args.pspace_exc,
        objective=args.objective,
        solver=args.solver,
        solver_kwargs=args.solver_kwargs,
        wfn_kwargs=args.wfn_kwargs,
        ham_noise=args.ham_noise,
        wfn_noise=args.wfn_noise,
        load_orbs=args.load_orbs,
        load_ham=args.load_ham,
        load_ham_um=args.load_ham_um,
        load_wfn=args.load_wfn,
        save_chk=args.save_chk,
        memory=args.memory,
    )
