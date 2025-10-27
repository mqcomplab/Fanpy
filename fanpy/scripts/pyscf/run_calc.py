import os
import re
import glob
import subprocess
from fanpy.scripts.pyscf.make_input import make_pyscf_input

def write_pyscf_py(input_path: str, coords: str, basis='sto-3g', memory='4GB', charge=0, spin=0, units='B'):
    """Write the PySCF calculation setup for the directories that match the given pattern.

    Parameters
    ----------
    input_path : str
        Input file path.
    coords : str
        Coordinates of the molecule.
    memory : str
        Amount of memory available to run the calculation.
        Default is 4GB.
    charge : int
        Total charge of the molecule.
        Default is 0.
    spin : int
        Spin (2n) of the molecule.
        Default is 0.

    """

    memory = memory.upper()
    if memory[-2:] == 'GB':
        memory = str(int(float(memory[:-2]) * 1024))
    elif memory[-2:] == 'MB':
        memory = str(int(float(memory[:-2])))
    else:
        raise ValueError('Memory must be given as a MB or GB (e.g. 1024MB, 1GB)')

    # make directory if it does not exist
    cwd = os.getcwd()
    input_path = os.path.join(cwd, input_path)
    if not os.path.isdir(input_path):
        os.makedirs(input_path, exist_ok=True)

    os.chdir(input_path)

    # get com content
    com_content = make_pyscf_input(coords, basis=basis, memory=memory, charge=charge, spin=spin, units=units)

    # make com file
    filename = 'calculate.py'
    with open(os.path.join(input_path, filename), 'w')as f:
        f.write(com_content)

    os.chdir(cwd)

def make_wfn_dirs(pattern: str, wfn_name: str, num_runs: int, rep_dirname_prefix: str=""):
    """Make directories for running the wavefunction calculations.

    Parameters
    ----------
    pattern : str
        Pattern for the directories on which the new directories will be created.
    wfn_name : str
        Name of the wavefunction.
    num_runs : int
        Number of calculations that will be run.
    rep_dirname_prefix : str
        Prefix to the directory names (of the repeated runs) to be created.
        Note that underscore will be added to separate index of run.

    """
    parent = pattern
    newdir = os.path.join(parent, wfn_name)
    if not os.path.isdir(newdir):
        os.makedirs(newdir, exist_ok=True)

    for i in range(num_runs):
        try:
            os.makedirs(os.path.join(newdir, f"{rep_dirname_prefix}_{i}"), exist_ok=True)
        except FileExistsError:
            pass

def write_wfn_py(pattern: str, wfn_type: str, geom: list, basis: str, optimize_orbs: bool=False,
                 pspace_exc=None, objective=None, solver=None,
                 ham_noise=None, wfn_noise=None,
                 solver_kwargs=None, wfn_kwargs=None,
                 load_orbs=None, load_ham=None, load_wfn=None, load_chk=None, load_prev=False,
                 memory=None, filename=None, ncores=1, exclude=None, fanpy_only=False):

    """Make a script for running calculations.

    Parameters
    ----------
    wfn_type : str
        Type of wavefunction.
        One of `
        "ci_pairs", "cisd", "doci", "fci",
        "mps",
        "determinant-ratio",
        "ap1rog", "apr2g", "apig", "apsetg", "apg",
        "rbm",
        "basecc", "standardcc", "generalizedcc", "senioritycc", "pccd", "ccsd", "ccsdt", "ccsdtq",
        "ap1rogsd", "ap1rogsd_spin", "apsetgd", "apsetgsd", "apg1rod", "apg1rosd",
        "ccsdsen0", "ccsdqsen0", "ccsdtqsen0", "ccsdtsen2qsen0".
    geom : list
        List of atomic coordinates for PySCF.
    basis : str
        Basis set for PySCF.
    optimize_orbs : bool
        If True, orbitals are optimized.
        If False, orbitals are not optimized.
        By default, orbitals are not optimized.
        Not compatible with faster fanpy (i.e. `old_fanpy=False`)
    pspace_exc : list of int
        Orders of excitations that will be used to build the projection space.
        Default is first, second, third, and fourth order excitations of the HF ground state.
        Used for slower fanpy (i.e. `old_fanpy=True`)
    objective : str
        Form of the Schrodinger equation that will be solved.
        Use `system` to solve the Schrodinger equation as a system of equations.
        Use `least_squares` to solve the Schrodinger equation as a squared sum of the system of
        equations.
        Use "one_energy" to minimize the energy projected on one sided.
        Use "variatioinal" to minimize the energy projected on both sided.
        Must be one of "system", "least_squares", "one_energy", and "variational".
        By default, the Schrodinger equation is solved as system of equations.
        "least_squares" is not supported in faster fanpy (i.e. `old_fanpy=False`)
    solver : str
        Solver that will be used to solve the Schrodinger equation.
        Keyword `cma` uses Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).
        Keyword `diag` results in diagonalizing the CI matrix.
        Keyword `minimize` uses the BFGS algorithm.
        Keyword `least_squares` uses the Trust Region Reflective Algorithm.
        Keyword `root` uses the MINPACK hybrd routine.
        Keyword `fanpt` uses FANPT solver to make little steps between Fock and given Hamiltonian,
        while optimizing with `least_squares` in between.
        Must be one of `cma`, `diag`, `least_squares`, or `root`.
        Must be compatible with the objective.
    ham_noise : float
        Scale of the noise to be applied to the Hamiltonian parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    wfn_noise : bool
        Scale of the noise to be applied to the wavefunction parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    solver_kwargs : str
        String to be added as arguments and keyword arguments when running the solver.
        To be added after `solver(objective, `
        Default settings are used if not provided.
    wfn_kwargs : str
        String to be added as arguments and keyword arguments when instantiating the
        wavefunction.
        To be added after `Wavefunction(nelec, nspin, params=params, memory=memory, `
        Default settings are used if not provided.
    load_orbs : str
        Numpy file of the orbital transformation matrix that will be applied to the initial
        Hamiltonian.
        If the initial Hamiltonian parameters are provided, the orbitals will be transformed
        afterwards.
    load_ham : str
        Numpy file of the Hamiltonian parameters that will overwrite the parameters of the initial
        Hamiltonian.
    load_wfn : str
        Numpy file of the wavefunction parameters that will overwrite the parameters of the initial
        wavefunction.
    load_chk : str
        Numpy file of the checkpoint for the optimization.
    memory : str
        Memory available to run the calculation.
        e.g. '2gb'
        Default assumes no restrictions on memory usage
    filename : str
        Filename to save the generated script file.
        Default just prints it out to stdout.
    fanpy_only : bool
        Use fanpy only, this is slower (but probably more robust).
        Default uses faster fanpy with the PyCI interface.
        Some features are not avaialble on new fanpy.

    """
    cwd = os.getcwd()

    if optimize_orbs:
        optimize_orbs = ['--optimize_orbs']
    else:
        optimize_orbs = []

    if pspace_exc is None:
        pspace_exc = [1, 2, 3, 4]

    if objective is None:
        objective = 'variational'

    if solver is None:
        solver = 'cma'
    if solver == 'cma' and solver_kwargs is None:
        solver_kwargs = ("sigma0=0.01, options={'ftarget': None, 'timeout': np.inf, "
                         "'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0}")

    load_files = []
    if load_orbs:
        load_files += ['--load_orbs', load_orbs]
    if load_ham:
        load_files += ['--load_ham', load_ham]
    if load_wfn:
        load_files += ['--load_wfn', load_wfn]
    if load_chk:
        load_files += ['--load_chk', load_chk]

    if memory is None:
        memory = []
    else:
        memory = ['--memory', memory]

    kwargs = []
    if wfn_kwargs is not None:
        kwargs += ['--wfn_kwargs', wfn_kwargs]
    if solver_kwargs is not None:
        kwargs += ['--solver_kwargs', solver_kwargs]
    if ham_noise is not None:
        kwargs += ['--ham_noise', str(ham_noise)]
    if wfn_noise is not None:
        kwargs += ['--wfn_noise', str(wfn_noise)]

    for parent in glob.glob(pattern):
        if not os.path.isdir(parent):
            continue
        if exclude and any(i in os.path.abspath(parent) for i in exclude):
            continue

        os.chdir(parent)

        if filename is None:
            filename = 'calculate.py'

        if load_prev:
            curr_sys = os.path.split(os.path.abspath("../../"))[1]
            re_curr_sys = re.search("(\w+_\w+_)(\d+)(_[\w-]+)", curr_sys)
            curr_index = int(re_curr_sys.group(2))
            prev_sys = re_curr_sys.group(1) + str(curr_index - 1) + re_curr_sys.group(3)
            prev_path = os.path.abspath(os.path.join("../../../", prev_sys, *parent.split(os.path.sep)[-3:]))
            load_ham = os.path.join(prev_path, "0", "hamiltonian.npy")
            load_wfn = os.path.join(prev_path, "0", "wavefunction.npy")
            kwargs += ["--load_ham", load_ham, "--load_wfn", load_wfn]

        save_chk = 'checkpoint.npy'

        # convert geom list into json string with no space
        geom_str = str(geom).replace(' ', '')
        # convert to json style for fanpy
        geom_str = geom_str.replace("'", '"')

        subprocess.run(['fanpy_make_pyscf_script' if fanpy_only else 'fanpy_make_fanci_pyscf_script',
                        *optimize_orbs, '--wfn_type', wfn_type,
                        '--geom', geom_str, 
                        '--pspace_exc', str(pspace_exc),
                        '--basis', basis,
                        '--objective', objective,
                        '--solver', solver, *kwargs,
                        *load_files,
                        '--save_chk', save_chk,
                        '--filename', filename, *memory,
                        ])

        os.chdir(cwd)

def run_calcs(pattern: str, time=None, memory=None, ncores=1, outfile='outfile', results_out="results.out", exclude=None, calc_range=None, arg=None):
    """Run the calculations for the selected files/directories.

    Parameters
    ----------
    pattern : str
        Pattern for selecting the files.

    Notes
    -----
    Can only execute at the base directory.

    """
    cwd = os.getcwd()

    if time is not None and memory is not None:
        time = time.lower()
        if time[-1] == 'd':
            time = int(time[:-1]) * 24 * 60
        elif time[-1] == 'h':
            time = int(time[:-1]) * 60
        elif time[-1] == 'm':
            time = int(time[:-1])
        else:
            raise ValueError('Time must be given in minutes, hours, or days (e.g. 1440m, 24h, 1d).')
        memory = memory.upper()
        if memory[-2:] == 'GB':
            memory = str(int(float(memory[:-2]) * 1024)) + 'MB'
        elif memory[-2:] not in ['MB']:
            raise ValueError('Memory must be given as a MB or GB (e.g. 1024MB, 1GB)')
        print(memory)
        submit_job = True
    elif time is None and memory is None:
        submit_job = False
    else:
        raise ValueError('You cannot provide only one of the time and memory.')

    for filename in glob.glob(pattern):
        if os.path.commonpath([cwd, os.path.abspath(filename)]) != cwd:
            continue
        filename = os.path.abspath(filename)[len(cwd)+1:]
        print(filename)
        if exclude and any(i in filename for i in exclude):
            continue

        database, system, basis, *wfn = filename.split(os.sep)
        if os.path.isdir(filename):
            os.chdir(filename)
        else:
            dirname, filename = os.path.split(filename)
            os.chdir(dirname)

        print(database, system, basis, wfn)
        if os.path.splitext(filename)[1] == '.py':
            with open('results.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('cwd=$PWD\n')
                if calc_range:
                    f.write(f'for i in {{{calc_range[0]}..{calc_range[1]}}}; do\n')
                else:
                    f.write('for i in */; do\n')
                f.write('    cd $i\n')
                f.write(f'    python -u ../calculate.py > {results_out}\n')
                f.write('    cd $cwd\n')
                f.write('done\n')
        elif os.path.isfile('./calculate.py'):
            with open('results.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'python -u ./calculate.py > {results_out}\n')
        else:
            os.chdir(cwd)
            continue
        subprocess.run(['chmod', 'u+x', 'results.sh'])
        command = ['./results.sh']

        print(os.getcwd())
        if submit_job:
            subprocess.run(['sbatch', f'--time={time}', f'--output={outfile}', f'--mem={memory}',
                            '--account=rmirandaquintana', f'--cpus-per-task={ncores}', *command])
        else:
            subprocess.run(command)

        # change directory
        os.chdir(cwd)
