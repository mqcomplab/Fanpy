"""Code generating script."""
import os
import textwrap

from fanpy.scripts.utils import check_inputs, parser

def make_script(  # pylint: disable=R1710,R0912,R0915
    wfn_type,
    optimize_orbs=False,
    pspace_exc=(1, 2),
    nproj=None,
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
    constraint=None):

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

    """
    # check inputs
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

    imports = ["numpy as np", "os", "sys", "pyci"]
    from_imports = [("fanpy.wfn.utils", "convert_to_fanci")]

    wfn_type = wfn_type.lower()
    if wfn_type == "ci_pairs":
        from_imports.append(("fanpy.wfn.ci.ci_pairs", "CIPairs"))
        wfn_name = "CIPairs"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "cisd":
        from_imports.append(("fanpy.wfn.ci.cisd", "CISD"))
        wfn_name = "CISD"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "fci":
        from_imports.append(("fanpy.wfn.ci.fci", "FCI"))
        wfn_name = "FCI"
        if wfn_kwargs is None:
            wfn_kwargs = "spin=None"
    elif wfn_type == "hci":
        from_imports.append(("fanpy.wfn.ci.hci", "hCI"))
        wfn_name = "hCI"
        if wfn_kwargs is None:
            wfn_kwargs = "alpha1=0.5, alpha2=0.25, hierarchy=1"
    elif wfn_type == "doci":
        from_imports.append(("fanpy.wfn.ci.doci", "DOCI"))
        wfn_name = "DOCI"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "mps":
        from_imports.append(("fanpy.wfn.network.mps", "MatrixProductState"))
        wfn_name = "MatrixProductState"
        if wfn_kwargs is None:
            wfn_kwargs = "dimension=None"
    elif wfn_type == "determinant-ratio":
        from_imports.append(("fanpy.wfn.quasiparticle.det_ratio", "DeterminantRatio"))
        wfn_name = "DeterminantRatio"
        if wfn_kwargs is None:
            wfn_kwargs = "numerator_mask=None"
    elif wfn_type == "ap1rog":
        from_imports.append(("fanpy.wfn.geminal.ap1rog", "AP1roG"))
        wfn_name = "AP1roG"
        if wfn_kwargs is None:
            wfn_kwargs = "ref_sd=None, ngem=None"
    elif wfn_type == "apr2g":
        from_imports.append(("fanpy.wfn.geminal.apr2g", "APr2G"))
        wfn_name = "APr2G"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apig":
        from_imports.append(("fanpy.wfn.geminal.apig", "APIG"))
        wfn_name = "APIG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apsetg":
        from_imports.append(("fanpy.wfn.geminal.apsetg", "BasicAPsetG"))
        wfn_name = "BasicAPsetG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apg":  # pragma: no branch
        from_imports.append(("fanpy.wfn.geminal.apg", "APG"))
        wfn_name = "APG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "network":
        from_imports.append(("fanpy.upgrades.numpy_network", "NumpyNetwork"))
        wfn_name = "NumpyNetwork"
        if wfn_kwargs is None:
            wfn_kwargs = "num_layers=2"
    elif wfn_type == "rbm":
        from_imports.append(("fanpy.wfn.network.rbm", "RestrictedBoltzmannMachine"))
        wfn_name = "RestrictedBoltzmannMachine"
        if wfn_kwargs is None:
            wfn_kwargs = "nbath=nspin, num_layers=1, orders=(1, 2)"

    elif wfn_type == "basecc":
        from_imports.append(("fanpy.wfn.cc.base", "BaseCC"))
        wfn_name = "BaseCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "standardcc":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "generalizedcc":
        from_imports.append(("fanpy.wfn.cc.generalized_cc", "GeneralizedCC"))
        wfn_name = "GeneralizedCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "senioritycc":
        from_imports.append(("fanpy.wfn.cc.seniority", "SeniorityCC"))
        wfn_name = "SeniorityCC"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "pccd":
        from_imports.append(("fanpy.wfn.cc.pccd_ap1rog", "PCCD"))
        wfn_name = "PCCD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ap1rogsd":
        from_imports.append(("fanpy.wfn.cc.ap1rog_generalized", "AP1roGSDGeneralized"))
        wfn_name = "AP1roGSDGeneralized"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ap1rogsd_spin":
        from_imports.append(("fanpy.wfn.cc.ap1rog_spin", "AP1roGSDSpin"))
        wfn_name = "AP1roGSDSpin"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apsetgd":
        from_imports.append(("fanpy.wfn.cc.apset1rog_d", "APset1roGD"))
        wfn_name = "APset1roGD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apsetgsd":
        from_imports.append(("fanpy.wfn.cc.apset1rog_sd", "APset1roGSD"))
        wfn_name = "APset1roGSD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apg1rod":
        from_imports.append(("fanpy.wfn.cc.apg1ro_d", "APG1roD"))
        wfn_name = "APG1roD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "apg1rosd":
        from_imports.append(("fanpy.wfn.cc.apg1ro_sd", "APG1roSD"))
        wfn_name = "APG1roSD"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdsen0":
        from_imports.append(("fanpy.wfn.cc.ccsd_sen0", "CCSDsen0"))
        wfn_name = "CCSDsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdqsen0":
        from_imports.append(("fanpy.wfn.cc.ccsdq_sen0", "CCSDQsen0"))
        wfn_name = "CCSDQsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdtqsen0":
        from_imports.append(("fanpy.wfn.cc.ccsdtq_sen0", "CCSDTQsen0"))
        wfn_name = "CCSDTQsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccsdtsen2qsen0":
        from_imports.append(("fanpy.wfn.cc.ccsdt_sen2_q_sen0", "CCSDTsen2Qsen0"))
        wfn_name = "CCSDTsen2Qsen0"
        if wfn_kwargs is None:
            wfn_kwargs = "ranks=None, indices=None, refwfn=None, exop_combinations=None"
    elif wfn_type == "ccs":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
        wfn_kwargs = f"ranks=[1], {wfn_kwargs}"
    elif wfn_type == "ccsd":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
        wfn_kwargs = f"ranks=[1, 2], {wfn_kwargs}"
    elif wfn_type == "ccsdt":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
        wfn_kwargs = f"ranks=[1, 2, 3], {wfn_kwargs}"
    elif wfn_type == "ccsdtq":
        from_imports.append(("fanpy.wfn.cc.standard_cc", "StandardCC"))
        wfn_name = "StandardCC"
        if wfn_kwargs is None:
            wfn_kwargs = "indices=None, refwfn=None, exop_combinations=None"
        wfn_kwargs = f"ranks=[1, 2, 3, 4], {wfn_kwargs}"

    if wfn_name in ["DOCI", "CIPairs"] and not optimize_orbs:
        from_imports.append(("fanpy.ham.senzero", "SeniorityZeroHamiltonian"))
        ham_name = "SeniorityZeroHamiltonian"
    else:
        from_imports.append(("fanpy.ham.restricted_chemical", "RestrictedMolecularHamiltonian"))
        ham_name = "RestrictedMolecularHamiltonian"

    from_imports.append(("fanpy.tools.sd_list", "sd_list"))

    if solver == "least_squares":
        if solver_kwargs is None:
            solver_kwargs = (
                "xtol=5.0e-7, ftol=1.0e-9, gtol=5.0e-7, "
                "max_nfev=fanci_wfn.nactive, verbose=2"
            )
        solver_kwargs = ", ".join(["mode='lstsq', use_jac=True", solver_kwargs])
    elif solver == "root":  # pragma: no branch
        if solver_kwargs is None:
            solver_kwargs = "method='hybr', options={'xtol': 1.0e-9}"
        solver_kwargs = ", ".join(["mode='root', use_jac=True", solver_kwargs])
    elif solver == "cma":
        if solver_kwargs is None:
            solver_kwargs = (
                "sigma0=0.01, options={'ftarget': None, 'timeout': np.inf, "
                "'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 1}"
            )
        solver_kwargs = ", ".join(["mode='cma', use_jac=False", solver_kwargs])
    elif solver == "minimize":
        if solver_kwargs is None:
            solver_kwargs = "method='BFGS', options={'gtol': 5e-7, 'disp':True}"
        solver_kwargs = ", ".join(["mode='bfgs', use_jac=True", solver_kwargs])
    elif solver == "fanpt":
        from_imports.append(("fanci.fanpt_wrapper", "reduce_to_fock, solve_fanpt"))
        if solver_kwargs is None:
            solver_kwargs = (
                "fill=fill, energy_active=True, resum=False, ref_sd=0, final_order=1, "
                "lambda_i=0.0, lambda_f=1.0, steps=50, "
                "solver_kwargs={'mode':'lstsq', 'use_jac':True, 'xtol':1.0e-8, 'ftol':1.0e-8, "
                "'gtol':1.0e-5, 'max_nfev':fanci_wfn.nactive, 'verbose':2, 'vtol':1e-5}"
            )
    else:
        raise ValueError("Unsupported solver")

    if nproj == 0:
        from_imports.append(("scipy.special", "comb"))

    if memory is not None:
        memory = "'{}'".format(memory)

    output = ""
    for i in imports:
        output += "import {}\n".format(i)
    for key, val in from_imports:
        output += "from {} import {}\n".format(key, val)
    output += "from fanpy.upgrades import speedup_sign\n"
    if "apg" in wfn_type or wfn_type in ['ap1rog', 'apig']:
        output += "import fanpy.upgrades.speedup_apg\n"
        # output += "import fanpy.upgrades.speedup_objective\n"
    if 'ci' in wfn_type or wfn_type == 'network':
        # output += "import fanpy.upgrades.speedup_objective\n"
        pass

    output += "\n\n"

    output += "# Number of electrons\n"
    output += "print('Number of Electrons: {}'.format(interface.nelec))\n"
    output += "\n"

    output += "# Number of spin orbitals\n"
    output += "print('Number of Spin Orbitals: {}'.format(interface.nspinorb))\n"
    output += "\n"

    output += "# Nuclear-nuclear repulsion\n"
    output += "print('Nuclear-nuclear repulsion: {}'.format(interface.energy_nuc))\n"
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
    wfn_init2 = "interface.nelec, interface.nspinorb, params={}, memory={}, {})\n".format(wfn_params, memory, wfn_kwargs)
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
        ham_init2 = "interface.one_int, interface.two_int"
        if solver == 'cma':
            ham_init2 += ')\n'
        else:
            ham_init2 += ', update_prev_params=True)\n'
        output += "\n".join(
            textwrap.wrap(ham_init1 + ham_init2, width=100, subsequent_indent=" " * len(ham_init1))
        )
    else:
        ham_final = "ham = {}(interface.one_int, interface.two_int)".format(ham_name)
        output += "\n".join(
            textwrap.wrap(ham_final, width=100, subsequent_indent=" " * len(ham_final))
        )
        output += "\n"
        ham_init = "fock = {}(interface.one_int, reduce_to_fock(interface.two_int))".format(ham_name)
        output += "\n".join(
            textwrap.wrap(ham_init, width=100, subsequent_indent=" " * len(ham_init))
        )
        output += "\n"
        output += "print('Hamiltonian: Fock Hamiltonian to {}')\n".format(ham_name)

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

    output += "# Projection space\n"
    output += "print('Projection space by excitation')\n"
    output += "fill = 'excitation'\n"
    if nproj == 0:
        output += "nproj = int(comb(interface.nspinorb // 2, interface.nelec - interface.nelec // 2) * comb(interface.nspinorb // 2, interface.nelec // 2))\n"
    elif nproj < 0:
        output += f"nproj = int(wfn.nparams * {-nproj}) + 1\n"
    else:
        output += f"nproj = {nproj}\n"
    output += "\n"

    output += "# Select parameters that will be optimized\n"
    if optimize_orbs:
        raise ValueError("Orbital optimization not supported.")
    output += "param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool))]\n"
    output += "\n"

    output += "# Initialize objective\n"
    if solver != "fanpt":
        output += "pyci_ham = pyci.hamiltonian(interface.energy_nuc, ham.one_int, ham.two_int)\n"
    else:
        output += "pyci_ham = pyci.hamiltonian(interface.energy_nuc, ham.one_int, ham.two_int)\n"
        output += "pyci_fock = pyci.hamiltonian(interface.energy_nuc, fock.one_int, fock.two_int)\n"
    seniority = 'wfn.seniority' if wfn_type != 'pccd' else '0'
    if solver == "fanpt":
        output += "\n".join(
                textwrap.wrap(
                    f"fanci_wfn = convert_to_fanci(wfn, pyci_fock, seniority={seniority}, "
                    "param_selection=param_selection, nproj=nproj, objective_type='projected', "
                    "norm_det=[(0, 1)])",
                    width=100, subsequent_indent=" " * len("fanci_wfn = convert_to_fanci(")
                )
            )
        output += "\n"
    elif objective in ["projected_stochastic", "projected"]:
        output += f"fanci_wfn = convert_to_fanci(wfn, pyci_ham, seniority={seniority}, param_selection=param_selection, nproj=nproj, objective_type='projected')\n"
    elif objective in ["energy", "one_energy", "variational"]:
        output += f"fanci_wfn = convert_to_fanci(wfn, pyci_ham, seniority={seniority}, param_selection=param_selection, nproj=nproj, objective_type='energy')\n"
    output += "fanci_wfn.tmpfile = '{}'\n".format(save_chk)
    output += "fanci_wfn.step_print = True\n"
    output += "\n"

    # output += "# Normalize\n"
    # output += "wfn.normalize(pspace)\n\n"

    # load energy
    output += "# Set energies\n"
    output += "integrals = np.zeros(fanci_wfn._nproj, dtype=pyci.c_double)\n"
    output += "olps = fanci_wfn.compute_overlap(fanci_wfn.active_params, 'S')[:fanci_wfn._nproj]\n"
    output += "fanci_wfn._ci_op(olps, out=integrals)\n"
    output += "energy_val = np.sum(integrals * olps) / np.sum(olps ** 2)\n"
    output += "print('Initial energy:', energy_val)\n"
    output += "\n"

    output += "# Solve\n"
    if solver == "fanpt":
        results1 = "fanci_results = solve_fanpt("
        results2 = "fanci_wfn, pyci_fock, pyci_ham, np.hstack([fanci_wfn.active_params, energy_val]), {})\n".format(solver_kwargs)
    elif objective == "projected_stochastic":
        results1 = "fanci_results = fanci_wfn.optimize_stochastic("
        results2 = "100, np.hstack([fanci_wfn.active_params, energy_val]), {})\n".format(solver_kwargs)
    else:
        results1 = "fanci_results = fanci_wfn.optimize("
        results2 = "np.hstack([fanci_wfn.active_params, energy_val]), {})\n".format(solver_kwargs)
    output += "print('Optimizing wavefunction: solver')\n"
    output += "\n".join(
        textwrap.wrap(results1 + results2, width=100, subsequent_indent=" " * len(results1))
    )
    output += "\n"
    output += "results = {}\n"
    if solver != "cma":
        output += "results['success'] = fanci_results.success\n"
        output += "results['params'] = fanci_results.x\n"
        output += "results['message'] = fanci_results.message\n"
        output += "results['internal'] = fanci_results\n"
        if solver == "minimize":
            output += "results['energy'] = fanci_results.fun\n"
        else:
            output += "results['energy'] = fanci_results.x[-1]\n"
        output += "\n"
    else:
        output += "results['success'] = fanci_results[-3] != {}\n"
        output += "results['params'] = fanci_results[0]\n"
        output += "results['function'] = fanci_results[1]\n"
        output += "results['energy'] = fanci_results[1]\n"
        output += "if results['success']:\n"
        output += "    results['message'] = 'Following termination conditions are satisfied:' + ''.join(\n"
        output += "        ' {0}: {1},'.format(key, val) for key, val in fanci_results[-3].items()\n"
        output += "    )\n"
        output += "    results['message'] = results['message'][:-1] + '.' \n"
        output += "else:\n"
        output += "    results['message'] = 'Optimization did not succeed.'\n"
        output += "results['internal'] = fanci_results\n"

    output += "# Results\n"
    output += "if results['success']:\n"
    output += "    print('Optimization was successful')\n"
    output += "else:\n"
    output += "    print('Optimization was not successful: {}'.format(results['message']))\n"
    output += "print('Final Electronic Energy: {}'.format(results['energy']))\n"
    output += "print('Final Total Energy: {}'.format(results['energy'] + interface.energy_nuc))\n"
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
    make_script(
        args.wfn_type,
        optimize_orbs=args.optimize_orbs,
        pspace_exc=args.pspace_exc,
        nproj=args.nproj,
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
    )
