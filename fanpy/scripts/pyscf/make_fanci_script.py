"""Code generating script."""
import os
import textwrap

from fanpy.scripts.utils import check_inputs
from fanpy.scripts.pyscf.pyscf_parser import parser
from fanpy.scripts.wavefunction_info import get_wfn_info

def make_script(  # pylint: disable=R1710,R0912,R0915
    geom,
    wfn_type,
    basis,
    hf_units="Angstrom",
    optimize_orbs=False,
    pspace_exc=(1, 2),
    objective="projected",
    solver="least_squares",
    solver_kwargs=None,
    wfn_kwargs=None,
    ham_noise=0.0,
    wfn_noise=0.0,
    load_orbs=None,
    load_ham=None,
    load_ham_um=None,
    load_wfn=None,
    load_chk=None,
    save_chk="",
    filename=None,
    memory=None,
    constraint=None,
    fanpt_kwargs=None):

    """Make a script for running calculations.

    Parameters
    ----------
    wfn_type : str
        Type of wavefunction.
        One of `ci_pairs`, `cisd`, `fci`, `doci`, `mps`, `determinant-ratio`, `ap1rog`, `apr2g`,
        `apig`, `apsetg`, or `apg`.
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
    filename : str
        Name of the script
        By default, the script is printed.
        If `-1` is given, then the script is returned as a string.
        Otherwise, the given string is treated as the name of the file.
    memory : None
        Memory available to run calculations.

    fanpt_kwargs : str, optional
        Keyword arguments for FANPT.
    """
    # check inputs
    # TODO: the inputs should be checked in the parser
    # check_inputs(
    #     wfn_type,
    #     pspace_exc,
    #     objective,
    #     solver,
    #     optimize_orbs=optimize_orbs,
    #     load_orbs=load_orbs,
    #     load_ham=load_ham,
    #     load_ham_um=load_ham_um,
    #     load_wfn=load_wfn,
    #     save_chk=save_chk,
    #     filename=filename if filename != -1 else None,
    #     memory=memory,
    #     solver_kwargs=solver_kwargs,
    #     wfn_kwargs=wfn_kwargs,
    #     ham_noise=ham_noise,
    #     wfn_noise=wfn_noise,
    # )

    imports = ["numpy as np", "os", "sys", "pyci", "fanpy.interface as interface"]
    from_imports = [('pyscf', "gto, scf"), ("fanpy.interface.pyscf", "PYSCF")]

    wfn_info = get_wfn_info(wfn_type)
    import_line, wfn_name, wfn_kwargs = wfn_info(wfn_kwargs)
    from_imports.append(import_line)

    if wfn_name in ["DOCI", "CIPairs"] and not optimize_orbs:
        from_imports.append(("fanpy.ham.senzero", "SeniorityZeroHamiltonian"))
        ham_name = "SeniorityZeroHamiltonian"
    else:
        from_imports.append(("fanpy.ham.restricted_chemical", "RestrictedMolecularHamiltonian"))
        ham_name = "RestrictedMolecularHamiltonian"

    from_imports.append(("fanpy.tools.sd_list", "sd_list"))

    if objective == "projected":
        from_imports.append(("fanpy.eqn.projected", "ProjectedSchrodinger"))
    else:
        raise ValueError("Unsupported objective. The PyCI interface only supports the projected objective only.")

    # set constaint to norm for fanpt
    # the pyci interface adds the norm constraint in default cases
    # thus we need to add it to the fanpy objective as well
    # TODO: create a more robust approach for handling constraints for FanPT
    if solver == "fanpt" and constraint is None:
        print("Setting constraint to 'norm' for FanPT.")
        constraint = 'norm'

    if constraint == 'norm':
        from_imports.append(("fanpy.eqn.constraints.norm", "NormConstraint"))
    elif constraint == 'energy':
        from_imports.append(("fanpy.eqn.constraints.energy", "EnergyConstraint"))

    if solver == "least_squares":
        if solver_kwargs is None:
            solver_kwargs = (
                "xtol=5.0e-7, ftol=1.0e-9, gtol=5.0e-7, "
                "max_nfev=objective.active_nparams, verbose=2"
            )
        solver_kwargs = ", ".join(["mode='lstsq', use_jac=True", solver_kwargs])
    elif solver == "root":  # pragma: no branch
        if solver_kwargs is None:
            solver_kwargs = "method='hybr', options={'xtol': 1.0e-9}"
        solver_kwargs = ", ".join(["mode='root', use_jac=True", solver_kwargs])
    # todo: re-enable these once PyCI interface supports one energy objectives
    # elif solver == "cma":
    #     if solver_kwargs is None:
    #         solver_kwargs = (
    #             "sigma0=0.01, options={'ftarget': None, 'timeout': np.inf, "
    #             "'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 1}"
    #         )
    #     solver_kwargs = ", ".join(["mode='cma', use_jac=False", solver_kwargs])
    # elif solver == "minimize":
    #     if solver_kwargs is None:
    #         solver_kwargs = "method='BFGS', options={'gtol': 5e-7, 'disp':True}"
    #     solver_kwargs = ", ".join(["mode='bfgs', use_jac=True", solver_kwargs])
    elif solver == "fanpt":
        from_imports.append(("fanpy.fanpt.fanpt", "FANPT"))
        if solver_kwargs is None:
            solver_kwargs = (
                "mode='lstsq', use_jac=True, xtol=1.0e-8, ftol=1.0e-8, "
                "gtol=1.0e-5, max_nfev=objective.active_nparams, verbose=2"
            )
        if fanpt_kwargs is None:
            fanpt_kwargs = (
                "energy_active=True, resum=False, ref_sd=0, final_order=1, "
                "lambda_i=0.0, lambda_f=1.0, steps=50"
            )
    else:
        raise ValueError("Unsupported solver")

    if memory is not None:
        memory = "'{}'".format(memory)

    output = ""
    for i in imports:
        output += "import {}\n".format(i)
    for key, val in from_imports:
        output += "from {} import {}\n".format(key, val)

    output += "\n\n"

    # PySCF Calculation
    # todo: this is not needed if ham parameters are loaded
    output += "print('# PySCF calculation')\n"
    output += f"mol = gto.M(atom = [{geom[0]}, \n"
    for atom in geom[1:-1]:
        output += f"                      {atom}, \n"
    output += f"                      {geom[-1]}], \n"
    output += f"            unit = '{hf_units}', \n"
    output += f"            basis = '{basis}') \n"
    output += "# Number of electrons\n"
    output += "\n"
    output += "myhf = scf.HF(mol) \n"
    output += "myhf.kernel() \n"
    output += "\n"
    output += "new_guess, _, stable, _ = myhf.stability(return_status=True) \n"
    output += "\n"
    output += "if not stable: \n"
    output += "    print('HF stability failed, trying Newton method...') \n"
    output += "    myhf = myhf.newton().run(new_guess, myhf.mo_occ) \n"
    output += "    _, _, stable, _ =  myhf.stability(return_status=True) \n"
    output += "\n"
    output += "if not stable: \n"
    output += "    raise RuntimeError('HF stability failed') \n"
    output += "\n"
    output += "pyscf_interface = PYSCF(myhf) \n"
    output += "\n"

    # Fanpy calculation    
    output += "print('>> >> Fanpy') \n"
    output += "\n"

    output += "# Number of electrons\n"
    output += "print('Number of Electrons: {}'.format(pyscf_interface.nelec))\n"
    output += "\n"

    output += "# Number of spin orbitals\n"
    output += "print('Number of Spin Orbitals: {}'.format(pyscf_interface.nspinorb))\n"
    output += "\n"

    output += "# Nuclear-nuclear repulsion\n"
    output += "print('Nuclear-nuclear repulsion: {}'.format(pyscf_interface.energy_nuc))\n"
    output += "\n"

    if load_wfn is not None:
        output += "# Load wavefunction parameters\n"
        output += "wfn_params_file = '{}'\n".format(load_wfn)
        output += "wfn_params = np.load(wfn_params_file)\n"
        output += "print('Load wavefunction parameters: {}'"
        output += ".format(os.path.abspath(wfn_params_file)))\n"
        output += "\n"
        wfn_params = "wfn_params"
    else:
        wfn_params = "None"

    output += "# Initialize wavefunction\n"
    wfn_init1 = "wfn = {}(".format(wfn_name)
    wfn_init2 = "pyscf_interface.nelec, pyscf_interface.nspinorb, params={}, memory={}, {})\n".format(wfn_params, memory, wfn_kwargs)
    output += "\n".join(
        textwrap.wrap(wfn_init1 + wfn_init2, width=100, subsequent_indent=" " * len(wfn_init1))
    )
    output += "\n"
    if wfn_noise not in [0, None]:
        output += (
            "wfn.assign_params(wfn.params + "
            "{} * 2 * (np.random.rand(*wfn.params.shape) - 0.5))\n".format(wfn_noise)
        )
    output += "print('Wavefunction: {}')\n".format(wfn_name)
    output += "\n"

    output += "# Initialize Hamiltonian\n"

    if solver != "fanpt":
        ham_init1 = "ham = {}(".format(ham_name)
        ham_init2 = "pyscf_interface.one_int, pyscf_interface.two_int"
        if solver == 'cma':
            ham_init2 += ')\n'
        else:
            ham_init2 += ', update_prev_params=True)\n'
        output += "\n".join(
            textwrap.wrap(ham_init1 + ham_init2, width=100, subsequent_indent=" " * len(ham_init1))
        )
    else:
        ham_final = "ham = {}(pyscf_interface.one_int, pyscf_interface.two_int)".format(ham_name)
        output += "\n".join(
            textwrap.wrap(ham_final, width=100, subsequent_indent=" " * len(ham_final))
        )
        output += "\n"

    if load_ham_um is not None:
        output += "# Load unitary matrix of the Hamiltonian\n"
        output += "ham_um_file = '{}'\n".format(load_ham_um)
        output += "ham_um = np.load(ham_um_file)\n"
        output += "print('Load unitary matrix of the Hamiltonian: {}'"
        output += ".format(os.path.abspath(ham_um_file)))\n"
        if ham_name == "UnrestrictedMolecularHamiltonian":  # pragma: no cover
            output += "ham._prev_unitary_alpha = ham_um[0]\n"
            output += "ham._prev_unitary_beta = ham_um[1]\n"
        else:
            output += "ham._prev_unitary = ham_um\n"
        output += "\n"

    if load_ham is not None:
        output += "# Load Hamiltonian parameters (orbitals)\n"
        output += "ham_params_file = '{}'\n".format(load_ham)
        output += "ham_params = np.load(ham_params_file)\n"
        output += "print('Load Hamiltonian parameters: {}'"
        output += ".format(os.path.abspath(ham_params_file)))\n"
        if load_ham_um:
            output += "ham._prev_params = ham_params\n"
        output += "ham.assign_params(ham_params)\n"
        output += "\n"

    output += "\n"
    if ham_noise not in [0, None]:
        output += (
            "ham.assign_params(ham.params + "
            "{} * 2 * (np.random.rand(*ham.params.shape) - 0.5))\n".format(ham_noise)
        )
    output += "print('Hamiltonian: {}')\n".format(ham_name)
    output += "\n"

    if load_chk:
        if False:
            output += "# Load checkpoint\n"
            output += "chk_point_file = '{}'\n".format(load_chk)
            output += "chk_point = np.load(chk_point_file)\n"
            if objective in ["projected", "system_qmc", "least_squares", "one_energy_system", "projected_stochastic"]:
                output += "if chk_point.size == objective.params.size - 1 and objective.energy_type == 'variable':\n"
                output += '    objective.assign_params(np.hstack([chk_point, 0]))\n'
                output += "elif chk_point.size - 1 == objective.params.size and objective.energy_type != 'variable':\n"
                output += '    objective.assign_params(chk_point[:-1])\n'
                output += 'else:\n'
                output += "    objective.assign_params(chk_point)\n"
            else:
                output += "objective.assign_params(chk_point)\n"
            output += "print('Load checkpoint file: {}'.format(os.path.abspath(chk_point_file)))\n"
            output += "\n"
            # check for unitary matrix
            output += '# Load checkpoint hamiltonian unitary matrix\n'
            output += "ham_params = chk_point[wfn.nparams:]\n"
            output += "load_chk_um = '{}_um{}'.format(*os.path.splitext(chk_point_file))\n"
            output += "if os.path.isfile(load_chk_um):\n"
            output += "    ham._prev_params = ham_params.copy()\n"
            output += "    ham._prev_unitary = np.load(load_chk_um)\n"
            output += "ham.assign_params(ham_params)\n\n"
        else:
            output += "# Load checkpoint\n"
            output += "import os\n"
            output += "dirname, chk_point_file = os.path.split('{}')\n".format(load_chk)
            output += "chk_point_file, ext = os.path.splitext(chk_point_file)\n"
            dirname, chk_point_file = os.path.split(load_chk)
            chk_point_file, ext = os.path.splitext(chk_point_file)
            if os.path.isfile(os.path.join(dirname, chk_point_file + '_' + wfn_name + ext)):
                output += "wfn.assign_params(np.load(os.path.join(dirname, chk_point_file + '_' + wfn.__class__.__name__ + ext)))\n"
            else:
                output += "wfn.assign_params(np.load(os.path.join(dirname, chk_point_file + '_wfn' + ext)))\n"
            if os.path.isfile(os.path.join(dirname, chk_point_file + '_' + ham_name + ext)):
                output += "ham.assign_params(np.load(os.path.join(dirname, chk_point_file + '_' + ham.__class__.__name__ + ext)))\n"
            if os.path.isfile(os.path.join(dirname, chk_point_file + '_' + ham_name + '_prev' + ext)):
                output += "ham._prev_params = np.load(os.path.join(dirname, chk_point_file + '_' + ham.__class__.__name__ + '_prev' + ext))\n"
            if os.path.isfile(os.path.join(dirname, chk_point_file + '_' + ham_name + '_um' + ext)):
                output += "ham._prev_unitary = np.load(os.path.join(dirname, chk_point_file + '_' + ham.__class__.__name__ + '_um' + ext))\n"
            output += "ham.assign_params(ham.params)\n\n"

    if pspace_exc is None:  # pragma: no cover
        pspace = "[1, 2]"
    else:
        pspace = str([int(i) for i in pspace_exc])
    output += "# Projection space\n"
    if solver == "root":
        output += f"pspace = sd_list(pyscf_interface.nelec, pyscf_interface.nspinorb, num_limit=len(wfn.params), exc_orders={pspace_exc}, spin=None)"
    else:
        output += f"pspace = sd_list(pyscf_interface.nelec, pyscf_interface.nspinorb, num_limit=None, exc_orders={pspace_exc}, spin=None)"
    output += "\n"
    output += "print('Projection space (orders of excitations): {}')\n".format(pspace)
    output += "\n"

    output += "# Select parameters that will be optimized\n"
    if optimize_orbs:
        raise ValueError("Orbital optimization not supported.")
    output += "param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool))]\n"
    output += "\n"

    if objective in ['projected', 'least_squares']:
        if constraint == 'norm':
            output += "# Set up constraints\n"
            output += "norm = NormConstraint(wfn, refwfn=pspace, param_selection=param_selection)\n"
            output += "weights = np.ones(len(pspace) + 1)\n"
            output += "weights[-1] = 100\n\n"
        elif constraint == 'energy':
            output += "# Set up constraints\n"
            output += "energy = EnergyConstraint(wfn, ham, param_selection=param_selection, refwfn=pspace,\n"
            output += "                          ref_energy=-100, queue_size=4, min_diff=1e-2, simple=True)\n"
            output += "weights = np.ones(len(pspace) + 1)\n"
            output += "weights[-1] = 100\n\n"
        else:
            output += '# Set up weights\n'
            output += "weights = np.ones(len(pspace))\n\n"

    output += "# Initialize objective\n"
    if objective == "projected":
        if solver == 'trustregion':
            objective1 = "objective = ProjectedSchrodinger("
            if wfn_type != 'ap1rog':
                objective2 = (
                    "wfn, ham, param_selection=param_selection, "
                    "pspace=pspace, refwfn=pspace, energy_type='compute', "
                    "energy=None, constraints=[], eqn_weights=weights)\n"
                )
            else:
                objective2 = (
                    "wfn, ham, param_selection=param_selection, "
                    "pspace=pspace, refwfn=[pspace[0]], energy_type='compute', "
                    "energy=None, constraints=[], eqn_weights=weights)\n"
                )
        else:
            objective1 = "objective = ProjectedSchrodinger("
            objective2 = (
                "wfn, ham, param_selection=param_selection, "
                "pspace=pspace, refwfn={}, energy_type='compute', "
                "energy=0.0, constraints=[{}], eqn_weights=weights)\n".format(
                    'pspace' if wfn_type != 'ap1rog' else 'None', constraint if constraint else ''
                )
            )
    elif objective == "least_squares":
        objective1 = "objective = LeastSquaresEquations("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "pspace=pspace, refwfn={}, energy_type='variable', "
            "energy=0.0, constraints=[{}], eqn_weights=weights)\n".format(
                'pspace' if wfn_type != 'ap1rog' else 'None', constraint if constraint else ''
            )
        )
    # todo: re-enable these once PyCI interface supports one energy objectives
    # elif objective == "variational":
    #     objective1 = "objective = EnergyTwoSideProjection("
    #     objective2 = (
    #         "wfn, ham, param_selection=param_selection, "
    #         "pspace_l=pspace, pspace_r=pspace, pspace_n=pspace)\n"
    #     )
    # elif objective == "one_energy":  # pragma: no branch
    #     objective1 = "objective = EnergyOneSideProjection("
    #     objective2 = (
    #         "wfn, ham, param_selection=param_selection, "
    #         "refwfn=pspace)\n"
        # )
    output += "\n".join(
        textwrap.wrap(objective1 + objective2, width=100, subsequent_indent=" " * len(objective1))
    )

    output += "\n"
    output += "objective.tmpfile = '{}'".format(save_chk)
    output += "\n\n"
    if objective == 'projected':
        output += 'objective.print_energy = False\n'
    if objective == 'least_squares':
        output += 'objective.print_energy = True\n'
    if solver != 'cma' and objective in ['one_energy', 'one_energy_system']:
        output += "objective.print_energy = True\n\n"
    if constraint == 'energy':
        output += 'objective.adaptive_weights = True\n'
        output += 'objective.num_count = 10\n'
        output += 'objective.decrease_factor = 5\n\n'
    if solver == 'trustregion':
        if wfn_type == 'ap1rog':
            output += "objective.adapt_type = []\n"
        else:
            # output += "objective.adapt_type = ['norm', 'energy']\n"
            output += "objective.adapt_type = []\n"
        output += "wfn.olp_threshold = 0.001\n"
        output += "objective.weight_type = 'ones'\n"
        output += "objective.sample_size = len(pspace)\n"
        output += "wfn.pspace_norm = objective.refwfn\n"

    if wfn_type in ['apg', 'apig', 'apsetg', 'apg2', 'apg3', 'apg4', 'apg5', 'apg6', 'apg7', 'doci']:
        output += "# Normalize\n"
        output += "wfn.normalize(pspace)\n\n"

    # load energy
    if objective in ["projected", "system_qmc", "least_squares", "one_energy_system"] and solver != 'trustregion':
        output += "# Set energies\n"
        output += "energy_val = objective.get_energy_one_proj(pspace)\n"
        output += "print('Initial energy:', energy_val)\n"
        output += "if objective.energy_type != 'compute':\n"
        output += "    objective.energy.params = np.array([energy_val])\n\n"
        if constraint == 'energy':
            output += "# Set energy constraint\n"
            output += "energy.ref_energy = energy_val - 15\n\n"
    
    # set up interface
    if solver == "fanpt":
        output += "fanpt = FANPT(objective, pyscf_interface.energy_nuc, {})\n".format(fanpt_kwargs)
    else:
        output += 'pyci_interface = interface.pyci.PYCI(objective, pyscf_interface.energy_nuc) \n'

    output += "# Solve\n"
    if solver == "fanpt":
        results1 = "results = fanpt.optimize(wfn.params, energy_val,"
        results2 = "{})\n".format(solver_kwargs)
    elif objective == "projected_stochastic":
        results1 = "results = pyci_interface.objective.optimize_stochastic("
        results2 = "100, np.hstack([objective.active_params, energy_val]), {})\n".format(solver_kwargs)
    else:
        results1 = "results = pyci_interface.objective.optimize("
        results2 = "np.hstack([objective.active_params, energy_val]), {})\n".format(solver_kwargs)
    output += "print('Optimizing wavefunction: solver')\n"
    output += "\n".join(
        textwrap.wrap(results1 + results2, width=100, subsequent_indent=" " * len(results1))
    )

    output += "\n"
    # output += "results = {}\n"
    # if solver != "cma":
    #     output += "results['success'] = fanci_results.success\n"
    #     output += "results['params'] = fanci_results.x\n"
    #     output += "results['message'] = fanci_results.message\n"
    #     output += "results['internal'] = fanci_results\n"
    #     if solver == "minimize":
    #         output += "results['energy'] = fanci_results.fun\n"
    #     else:
    #         output += "results['energy'] = fanci_results.x[-1]\n"
    #     output += "\n"
    # else:
    #     output += "results['success'] = fanci_results[-3] != {}\n"
    #     output += "results['params'] = fanci_results[0]\n"
    #     output += "results['function'] = fanci_results[1]\n"
    #     output += "results['energy'] = fanci_results[1]\n"
    #     output += "if results['success']:\n"
    #     output += "    results['message'] = 'Following termination conditions are satisfied:' + ''.join(\n"
    #     output += "        ' {0}: {1},'.format(key, val) for key, val in fanci_results[-3].items()\n"
    #     output += "    )\n"
    #     output += "    results['message'] = results['message'][:-1] + '.' \n"
    #     output += "else:\n"
    #     output += "    results['message'] = 'Optimization did not succeed.'\n"
    #     output += "results['internal'] = fanci_results\n"

    output += "# Results\n"
    output += "if results['success']:\n"
    output += "    print('Optimization was successful')\n"
    output += "else:\n"
    output += "    print('Optimization was not successful: {}'.format(results['message']))\n"
    output += "print('Final Electronic Energy: {}'.format(results['energy']))\n"
    output += "print('Final Total Energy: {}'.format(results['energy'] + pyscf_interface.energy_nuc))\n"
    # output += "print('Residuals: {}'.format(results['residuals']))\n"

    if filename is None:  # pragma: no cover
        print(output)
    # NOTE: number was used instead of string (eg. 'return') to prevent problems arising from
    #       accidentally using the reserved string/keyword.
    elif filename == -1:
        return output
    else:
        with open(filename, "a") as f:  # pylint: disable=C0103
            f.write(output)

def main():  # pragma: no cover
    """Run script for run_calc using arguments obtained via argparse."""
    parser.description = "Optimize a wavefunction and/or Hamiltonian."
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        required=False,
        help="Name of the file that contains the output of the script.",
    )
    args = parser.parse_args()
    # format pspace excitations
    args.pspace_exc = args.pspace_exc.strip("[]")
    if args.pspace_exc is not None:
        args.pspace_exc = [int(x) for x in args.pspace_exc.split(",")]
    make_script(
        args.geom,
        args.wfn_type,
        args.basis,
        hf_units=args.hf_units,
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
        load_chk=args.load_chk,
        save_chk=args.save_chk,
        filename=args.filename,
        memory=args.memory,
        constraint=args.constraint,
        fanpt_kwargs=args.fanpt_kwargs
    )
