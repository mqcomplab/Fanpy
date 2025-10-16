r"""Class that contains the elements required to perform a FANPT calculation with explicit E."""

import numpy as np

from fanpy.fanpt.containers.base import FANPTContainer


class FANPTContainerEParam(FANPTContainer):
    r"""Container for the matrices and vectors required ot perform a FANPT calculation.

    We assume that the equations to be solved have the following structure:

    A block of nproj equations obtained by projecting the Schrödinger equation in a space
    of nproj Slater determinants:

    G_1 = <1|ham(l)|psi(l)> - E * <1|psi(l)> = 0
    G_2 = <2|ham(l)|psi(l)> - E * <2|psi(l)> = 0
    ....
    G_nproj = <nproj|ham(l)|psi(l)> - E * <nproj|psi(l)> = 0

    We also have constraint equations (at least one, used to impose intermediate normalization).
    It is assumed that the constraint equations only depend on the wavefunction parameters, p_k,
    and are independent of the energy, E, and lambda, l. This implies that while all the vector
    have nequation elements and all the matrices have nequation rows, except for the coefficient
    matrix (dG/dp_k), only the first nproj elements of the vectors and the first nproj rows of
    the matrices are non-zero.

    Attributes
    ----------
    fanci_objective : FanCI instance
        FanCI wavefunction.
    params : np.ndarray
        Wavefunction parameters and energy at for the given lambda value.
    ham0 : pyci.hamiltonian
        PyCI Hamiltonian of the ideal system.
    ham1 : pyci.hamiltonian
        PyCI Hamiltonian of the real system.
    l : float
        Lambda value.
    ref_sd : int
        Index of the Slater determinant used to impose intermediate normalization.
        <n[ref_sd]|Psi(l)> = 1.
    inorm : bool
        Indicates whether we will work with intermediate normalization or not.
    ham : pyci.hamiltonian
        PyCI Hamiltonian for the given value of lambda.
        ham = l * ham1 + (1 - l) * ham0
    f_pot : pyci.hamiltonian
        PyCI Hamiltonian corresponding to the fluctuation potential.
        f_pot = ham1 - ham0
    wfn_params : np.ndarray
        Wavefunction parameters.
    energy : float
        Energy for the current value of lambda.
    active_energy : bool
        Indicates if the energy will be varied in the calculations.
        It is False either when the energy is frozen in a E-param calculation
        or in any E-free calculation.
    ham_ci_op : pyci.sparse_op
        PyCI sparse operator corresponding to the perturbed Hamiltonian.
    f_pot_ci_op : pyci.sparse_op
        PyCI sparse operator corresponding to the fluctuation potential.
    ovlp_s : np.ndarray
        Overlaps of the wavefunction with the determinants in the "S" space.
    d_ovlp_s : np.ndarray
        Derivatives of the overlaps of the wavefunction with the determinants in the "S" space
        with respect to the active wavefunction parameters.
    d_g_lambda : np.ndarray
        Derivative of the FANPT equations with respect to the lambda parameter.
        numpy array with shape (self.nequations,).
    d2_g_lambda_wfnparams : np.ndarray
        Derivative of the FANPT equations with respect to lambda and the wavefunction
        parameters.
        numpy array with shape (self.nequations, len(self.wfn_params_active)).
    d2_g_e_wfnparams : np.ndarray
        Derivative of the FANPT equations with respect to the energy and the wavefunction
        parameters.
        numpy array with shape (self.nequations, len(self.wfn_params_active)).
    c_matrix : np.ndarray
        Coefficient matrix of the FANPT system of equations.
        numpy array with shape (self.nequations, len(self.nactive)).

    Properties
    ----------
    nactive : int
        Number of active parameters.
    nequation : int
        Number of equations.
    nproj : int
        Number of determinants in the projection ("P") space.

    Methods
    -------
    __init__(self, fanci_objective, params, ham0, ham1, l=0, ref_sd=0)
        Initialize the FANPT container.
    der_g_lambda(self)
        Derivative of the FANPT equations with respect to the lambda parameter.
    der2_g_lambda_wfnparams(self)
        Derivative of the FANPT equations with respect to lambda and the wavefunction parameters.
    der2_g_e_wfnparams(self)
        Derivative of the FANPT equations with respect to the energy and the wavefunction
        parameters.
    gen_coeff_matrix(self)
        Generate the coefficient matrix of the linear FANPT system of equations.
    """

    def __init__(
        self,
        fanci_interface,
        params,
        ham0,
        ham1,
        l=0,
        ref_sd=0,
        inorm=False,
        norm_det=None,
        ham_ci_op=None,
        f_pot_ci_op=None,
        ovlp_s=None,
        d_ovlp_s=None,
        dd_ovlp_s=None,
    ):
        r"""Initialize the FANPT container.

        Parameters
        ----------
        fanci_interface : PYCI interface instance
            PYCI interface to FanCI wavefunction.
        params : np.ndarray
            Wavefunction parameters and energy at for the given lambda value.
        ham0 : pyci.hamiltonian
            PyCI Hamiltonian of the ideal system.
        ham1 : pyci.hamiltonian
            PyCI Hamiltonian of the real system.
        l : float
            Lambda value.
        ref_sd : int
            Index of the Slater determinant used to impose intermediate normalization.
            <n[ref_sd]|Psi(l)> = 1.
        ham_ci_op : {pyci.sparse_op, None}
            PyCI sparse operator of the perturbed Hamiltonian.
        f_pot_ci_op : {pyci.sparse_op, None}
            PyCI sparse operator of the fluctuation potential.
        ovlp_s : {np.ndarray, None}
            Overlaps in the "S" projection space.
        d_ovlp_s : {np.ndarray, None}
            Derivatives of the overlaps in the "S" projection space.
        """
        super().__init__(
            fanci_interface,
            params,
            ham0,
            ham1,
            l,
            ref_sd,
            inorm,
            norm_det,
            ham_ci_op,
            f_pot_ci_op,
            ovlp_s,
            d_ovlp_s,
            dd_ovlp_s,
        )
        self.der2_g_e_wfnparams()
        self.der2_g_wfnparams2()
        self.der3_g_lambda_wfnparams2()
        self.der3_g_e_wfnparams2()
    def der_g_lambda(self):
        r"""Derivative of the FANPT equations with respect to the lambda parameter.

        dG/dl = <n|f_pot|psi(l)>

        Generates
        ---------
        d_g_lambda : np.ndarray
            Derivative of the FANPT equations with respect to the lambda parameter.
            numpy array with shape (self.nequations,).
        """
        f = np.zeros(self.nequation)
        f_proj = f[: self.nproj]
        self.f_pot_ci_op(self.ovlp_s, out=f_proj)
        self.d_g_lambda = f

    def der2_g_lambda_wfnparams(self):
        r"""Derivative of the FANPT equations with respect to lambda and the wavefunction parameters.

        d^2G/dldp_k = <n|f_pot|dpsi(l)/dp_k>

        Generates
        ---------
        d2_g_lambda_wfnparams : np.ndarray
            Derivative of the FANPT equations with respect to lambda and the wavefunction
            parameters.
            numpy array with shape (self.nequations, len(self.wfn_params_active)).
        """
        if self.active_energy:
            ncolumns = self.nactive - 1
        else:
            ncolumns = self.nactive
        f = np.zeros((self.nequation, ncolumns), order="F")
        f_proj = f[: self.nproj]
        for f_proj_col, d_ovlp_col in zip(f_proj.transpose(), self.d_ovlp_s.transpose()):
            self.f_pot_ci_op(d_ovlp_col, out=f_proj_col)
        self.d2_g_lambda_wfnparams = f

    def der3_g_lambda_wfnparams2(self):
        r"""Compute the third derivative of the FANPT equations with respect to two wavefunction 
        parameters and the lambda perturbation parameter.
    
        This evaluates:
        d^3G / dp_k dλ dp_l = sum_n <m|V|n> d^2 f_n / dp_k dp_l
    
        - For each projection equation m
        - For each pair of active wavefunction parameters (k, l)
        - Stores the result in a tensor: (# equations, # active params, # active params)
    
        Notes
        -----
        If `self.active_energy` is True, the last wfn parameter is omitted.
        Only `self.nproj` projection equations are non-zero.
        """
        # Determine how many wavefunction parameters to use
        if self.active_energy:
            ncolumns = self.nactive - 1
        else:
            ncolumns = self.nactive
    
        # Allocate tensor: (nequation, ncolumns, ncolumns)
        f = np.zeros((self.nequation, ncolumns, ncolumns), order="F")
    
        # Only the projected equations get contributions
        f_proj = f[: self.nproj]
    
        # Loop over all (k, l) pairs
        for k in range(ncolumns):
            for l in range(ncolumns):
                # Slice the second derivative overlap vector: shape (nproj,)
                d2_ovlp_vector = self.dd_ovlp_s[:, k, l]
    
                # Apply perturbation operator: sum_n <m|V|n> * d2_ovlp
                # This fills the result for all projected equations
                self.f_pot_ci_op(d2_ovlp_vector, out=f_proj[:, k, l])
    
        # Store result
        self.d3_g_lambda_wfnparams2 = f


    def der2_g_e_wfnparams(self):
        r"""Derivative of the FANPT equations with respect to the energy and the wavefunction
        parameters.

        d^2G/dEdp_k = -<n|dpsi(l)/dp_k>

        Generates
        ---------
        d2_g_e_wfnparams : np.ndarray
            Derivative of the FANPT equations with respect to the energy and the wavefunction
            parameters.
            numpy array with shape (self.nequations, len(self.wfn_params_active)).
        """
        if self.active_energy:
            ncolumns = self.nactive - 1
            f = np.zeros((self.nequation, ncolumns), order="F")
            f[: self.nproj] = -self.d_ovlp_s[: self.nproj]
            self.d2_g_e_wfnparams = f
        else:
            self.d2_g_e_wfnparams = None

    def der3_g_e_wfnparams2(self):
        r"""Compute the third derivative of the FANPT equations with respect to two
        wavefunction parameters and the energy.
    
        Computes:
        d^3G / dp_k dp_l dE = -d^2 f / dp_k dp_l
    
        Only computed if `self.active_energy` is True.
    
        Generates
        ---------
        d3_g_wfnparams2_e : np.ndarray
            Third derivative tensor with shape:
            (# equations, # active params, # active params)
        """
        if self.active_energy:
            ncolumns = self.nactive - 1
            f = np.zeros((self.nequation, ncolumns, ncolumns), order="F")
            f[: self.nproj] = -self.dd_ovlp_s[: self.nproj]
            self.d3_g_e_wfnparams2 = f
        else:
            self.d3_g_e_wfnparams2 = None


    def gen_coeff_matrix(self):
        r"""Generate the coefficient matrix of the linear FANPT system of equations.

        dG/dp_k = <n|ham(l)|dpsi(l)/dp_k> - E * <n|dpsi/dp_k>

        If the energy is active, the last column of the matrix has the form:

        dG/dE = -<n|psi(l)>

        Generates
        ---------
        c_matrix : np.ndarray
            Coefficient matrix of the FANPT system of equations.
            numpy array with shape (self.nequations, len(self.nactive)).
        """
        self.c_matrix = self.fanci_objective.compute_jacobian(self.params)

    def der2_g_wfnparams2(self):
        r"""
        Compute the second derivative of the FANCI projection equations with respect to 
        two wavefunction parameters.
    
        Specifically, this evaluates the tensor:
            d²Gₘ / dp_k dp_l = ⟨m|H|∂²ψ/∂p_k ∂p_l⟩ - E ⋅ ∂²fₘ / ∂p_k ∂p_l
    
        The result is a rank-3 tensor where each entry corresponds to the second 
        derivative of the m-th projection equation with respect to the k-th and l-th 
        active wavefunction parameters, accounting for energy-dependent correction.
    
        Returns
        -------
        d2_g_wfnparams_wfnparams : np.ndarray
            A tensor of shape (nequation, ncolumns, ncolumns), where
            ncolumns = nactive - 1 if active_energy is True, else nactive.
            Contains the second derivatives of the projection equations.
        """
        if self.active_energy:
            ncolumns= self.nactive - 1
            f = np.zeros((self.nequation, ncolumns, ncolumns), order = "F")
            f_proj = f[:self.nproj]
            energy=self.params[-1]
            

        for i in range(ncolumns):
            for j in range(ncolumns):
                self.ham_ci_op(self.dd_ovlp_s[:,i,j], out= f_proj[:,i,j])
                dd_ovlp_proj= self.dd_ovlp_s[:, i, j][:self.nproj]
                dd_ovlp_proj *= energy
                f_proj[:,i,j] -= dd_ovlp_proj
        self.d2_g_wfnparams2 = f

