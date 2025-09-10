"""CI Singles and Doubles Wavefunction."""

from fanpy.tools.sd_list import sd_list
from fanpy.tools import slater
from fanpy.wfn.ci.base import CIWavefunction
import numpy as np

# Hard-coded alpha choices per pattern
ALPHA_BY_PATTERN = {
    "pos_diag": (0.5, 0.25),   # positive slope diagonals
    "neg_diag": (1, -0.5),  # negative slope diagonals
    "hch":      (1, -0.25),    # horizantal chess horse
    "vch":      (1.0, -1),    # vertical chess horse
}


class hCI(CIWavefunction):
    r"""Hierarchy Configuration Interaction Wavefunction.

    CI with restricted excitation number and seniority number.

    hierarchy parameter
        :math: `h = \alpha_1 e + \alpha_2 s`,
        where, 'e' is excitation degree & 's' is the seniority number.

    ***Reference: Hierarchy Configuration Interaction: Combining Seniority Number
                  and Excitation Degree by Fabris Kossoski, et.al.
                  https://doi.org/10.48550/arXiv.2203.06154

    Extra Attributes (hCI)
    ----------------------
    hci_version : {'old', 'new'}
        Version flag that controls how determinants are grouped into hierarchies.
    
    hci_pattern : {'pos_diag', 'neg_diag', 'hch', 'vch'}
        Partitioning scheme of the (excitation, seniority) plane used to select determinants.
    
    hierarchy : float
        Target hierarchy index. For ``'old'`` it is used directly; for ``'new'`` it
        seeds the construction of the admissible hierarchy set (``pos_hierarchies``).


    Attributes
    ----------
    nelec : int
        Number of electrons.
    spin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    _spin : float
        Total spin of each Slater determinant.
        :math:`\frac{1}{2}(N_\alpha - N_\beta)`.
        Default is no spin (all spins possible).
    _seniority : int
        Number of unpaired electrons in each Slater determinant.
    sds : tuple of int
        List of Slater determinants used to construct the CI wavefunction.
    dict_sd_index : dictionary of int to int
        Dictionary from Slater determinant to its index in sds.
    refwfn : {CIWavefunction, int, None}
        Reference wavefunction upon which the CC operator will act.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    spin : int
        Spin of the wavefunction
    seniority : int
        Seniority of the wavefunction
    dtype
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, params=None, sds=None, spin=None, seniority=None):
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign parameters of the wavefunction.
    import_params(self, guess)
        Transfers parameters from a guess wavefunction to the wavefunction.
    enable_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    assign_spin(self, spin=None)
        Assign the spin of the wavefunction.
    assign_seniority(self, seniority=None)
        Assign the seniority of the wavefunction.
    assign_sds(self, sds=None)
        Assign the list of Slater determinants from which the hCI wavefunction is constructed.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    Extra Methods
    -------------
    assign_hci_version(hci_version)
        Set the hCI wavefunction version, which controls how determinants are grouped
        into hierarchies for the current pattern.
    
    assign_pattern(hci_pattern)
        Set the hCI pattern used to organize determinants across hierarchies.
    
    assign_alphas()
        Derive and store the α-coefficients for the hierarchy relation associated
        with the current pattern.
    
    assign_hierarchy(hierarchy)
        Set the target hierarchy index used when selecting determinants.
    
    assign_pos_hierarchies()
        Compute and store the admissible hierarchy indices implied by the current
        version, pattern, and settings.

    """

    def __init__(
        self, nelec, nspin, hci_version, hci_pattern, sds=None, memory=None, hierarchy=None, refwfn=None
    ):
        self.assign_hci_version(hci_version=hci_version)
        self.assign_hci_pattern(hci_pattern=hci_pattern)
        self.assign_alphas()  # now derives from hci_pattern via ALPHA_BY_PATTERN
        self.assign_hierarchy(hierarchy=hierarchy)
        self.assign_pos_hierarchies()
        super().__init__(nelec, nspin, memory=memory)
        self.assign_sds(sds=sds)
        self.assign_refwfn(refwfn=refwfn)

    def assign_hci_version(self, hci_version):
        """
        Set the hCI wavefunction *version*, which controls how determinants are
        grouped into hierarchies for a given ``hci_pattern``.
    
        Versions
        --------
        "old"
            Append determinants belonging **only** to the hierarchy explicitly
            specified by the current pattern (i.e., the exact hierarchy index).
        "new"
            Append determinants selected by the pattern’s modulo-based hierarchy
            rule (see :meth:`assign_pos_hierarchies` for the precise selection
            logic).
    
        Parameters
        ----------
        hci_version : {'old', 'new', None}
            Version to use. If ``None``, the current setting is left unchanged.
    
        Raises
        ------
        TypeError
            If ``hci_version`` is not one of ``'old'``, ``'new'``, or ``None``.
        """


        if hci_version is None:
            hci_version = "new"
    
        if hci_version not in ("old", "new"):
            raise TypeError("hci_version must be 'old', 'new', or None (None defaults to 'new').")

        self.hci_version = hci_version

    def assign_hci_pattern(self, hci_pattern):
        """
        Set the hCI **pattern** used to organize determinants across hierarchies.
    
        Parameters
        ----------
        hci_pattern : {'pos_diag', 'neg_diag', 'vch', 'hch'}
            Selection topology in the (excitation order ``e``, seniority ``s``) plane:
            - 'pos_diag' : positive-diagonal relation between ``e`` and ``s``.
            - 'neg_diag' : negative-diagonal relation between ``e`` and ``s``.
            - 'vch'      : vertical chess horse relation between ``e`` and ``s``.
            - 'hch'      : horizontal chess horse relation between ``e`` and ``s``.
    
        Raises
        ------
        TypeError
            If ``hci_pattern`` is not one of the supported values.
    
        Notes
        -----
        This sets ``self.hci_pattern`` and is consumed by methods like
        :meth:`assign_alphas` and :meth:`assign_pos_hierarchies`.
        """


        if hci_pattern is None:
            raise TypeError("hci_pattern cannot be None.")
    
        key = str(hci_pattern).strip().lower()
    
        if key not in ALPHA_BY_PATTERN:
            valid = ", ".join(sorted(ALPHA_BY_PATTERN.keys()))
            raise TypeError(f"Unknown hci_pattern {key!r}. Valid options: {valid}")
    
        self.hci_pattern = key

    def assign_hierarchy(self, hierarchy=4.5):
        """Assign the hierachy number for hCI wavefunction.


        Parameters
        ----------
        hierarchy : {float, None}
            Hierarchy number to restrict the space of slater determinats with
            inclusion of limited excitation orders & restriction on seniority numbers.

            `None` means sds is set to `None` (default value: all orders of
             excitation and no seniority restrictions.).

        Raises
        ------
        TypeError
            If the hierarchy is not an integer, float or None.
        ValueError
            If the hierarchy is neither a positive number nor zero.

        """

        if not isinstance(hierarchy, (int, float, type(None))):
            raise TypeError("`hierarchy` must be provided as an integer, float or `None`.")

        else:
            self.hierarchy = hierarchy

    def assign_pos_hierarchies(self):
        """
        Compute and store the set of **admissible hierarchy indices** for determinant
        selection, based on the current ``hci_version``, ``hci_pattern``, and
        ``hierarchy`` value.
    
        Behavior
        --------
        If ``hci_version == 'new'``:
            - ``'pos_diag'``:   ``[0, 0.5, 1.0, ..., hierarchy]``  (step = 0.5)
            - ``'hch'``:        ``[0, 0.5, 1.0, ..., hierarchy]``  (step = 0.5)
            - ``'neg_diag'``:   ``[0, 1, 2, ..., hierarchy]``      (step = 1)
            - ``'vch'``:        ``[-hierarchy, ..., -1, 0, 1, ..., hierarchy]`` (step = 1)
    
            These ranges are constructed with ``numpy.arange`` and are inclusive of
            the upper bound (via ``+0.5`` or ``+1`` padding) to ensure the terminal
            value is present.
    
        If ``hci_version != 'new'`` (i.e., ``'old'``):
            - Use only the explicitly requested hierarchy index:
              ``pos_hierarchies = [self.hierarchy]``.
    
        Notes
        -----
        This method assumes ``self.hci_version``, ``self.hci_pattern``, and
        ``self.hierarchy`` have already been assigned.
        """

        if self.hci_version == "new":

            if self.hci_pattern == "pos_diag":
                self.pos_hierarchies = np.arange(0, self.hierarchy + 0.5, 0.5)

            elif self.hci_pattern == "neg_diag":
                self.pos_hierarchies = np.arange(0, self.hierarchy + 1, 1)

            elif self.hci_pattern == "hch":
                self.pos_hierarchies = np.arange(0, self.hierarchy + 0.5, 0.5)

            elif self.hci_pattern == "vch":
                self.pos_hierarchies = np.arange(-1 * self.hierarchy, self.hierarchy + 1, 1)

        else:
            self.pos_hierarchies = [self.hierarchy]

    def assign_alphas(self):
        """
        Derive alpha1/alpha2 from the chosen hci_pattern.
        """

        if not hasattr(self, "hci_pattern"):
            raise AttributeError("hci_pattern must be assigned before assign_alphas().")
    
        if self.hci_pattern not in ALPHA_BY_PATTERN:
            raise ValueError(f"Unknown hci_pattern {self.hci_pattern!r}")
    
        a1, a2 = ALPHA_BY_PATTERN[self.hci_pattern]
    
        # (Optional) but for safety incase the hard coded values for alpha has a type or value error:
        if not isinstance(a1, (int, float)) or not isinstance(a2, (int, float)):
            raise TypeError("alpha values must be numbers.")
        if a1 < -1 or a1 > 1 or a2 < -1 or a2 > 1:
            raise ValueError("`alpha` must be within [-1, 1].")
        self.alpha1 = a1
        self.alpha2 = a2

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction upon which the CC operator will act.

        Parameters
        ----------
        refwfn : {CIWavefunction, int, None}
            Wavefunction that will be modified by the CC operator.
            Default is the ground-state Slater determinant.

        Raises:
        ------
        TypeError
            If refwfn is not a CIWavefunction or int instance.
        AttributeError
            If refwfn does not have a sd_vec attribute.
        ValueError
            If refwfn does not have the right number of electrons.
            If refwfn does not have the right number of spin orbitals.

        """
        
        if refwfn is None:
            self.refwfn = slater.ground(nocc=self.nelec, norbs=self.nspin)
        elif isinstance(refwfn, int):
            if slater.total_occ(refwfn) != self.nelec:
                raise ValueError("refwfn must have {} electrons".format(self.nelec))
            # TODO: check that refwfn has the right number of spin-orbs
            self.refwfn = refwfn
        else:
            if not isinstance(refwfn, CIWavefunction):
                raise TypeError("refwfn must be a CIWavefunction or a int object")
            if not hasattr(refwfn, "sds"):  # NOTE: Redundant test.
                raise AttributeError("refwfn must have the sds attribute")
            if refwfn.nelec != self.nelec:
                raise ValueError("refwfn must have {} electrons".format(self.nelec))
            if refwfn.nspin != self.nspin:
                raise ValueError("refwfn must have {} spin orbitals".format(self.nspin))
            self.refwfn = refwfn

    def assign_sds(self, sds=None):
        """Generate the list of pairs of allowed excitation orders, 'e', and seniorities, 's'.
        Obtain the list of allowed Slater determinants corresponding to each (e,s) pair.
        Assign the obtained list of Slater determinants in the hCI wavefunction.

        Ignores user input and uses the Slater determinants for the FCI wavefunction (within the
        given spin).

        Parameters
        ----------
        sds : iterable of int
            List of Slater determinants (in the form of integers that describe the occupation as a
            bitstring)

        Raises
        ------
        ValueError
            If the sds is not `None` (default value).

        Notes
        -----
        Needs to have `nelec`, `nspin`, `spin`, `seniority`.

        """
        if __debug__ and sds is not None:
            raise ValueError(
                "Only the default list of Slater determinants is allowed. i.e. sds "
                "is `None`. If you would like to customize your CI wavefunction, use "
                "CIWavefunction instead."
            )

        # ******************* GET LISTS FOR ALLOWED (e, s) PAIRS ******************
        # Here, the way of obtaining (e, s) pairs is only for the reference systems
        # having maximum number of paired electrons, so the seniority of the reference
        # should be either 0 or 1.

        # np.array of allowed excitation orders for a given 'nelec'
        exc_orders = range(0, self.nelec + 1, 1)

        # Initialize the empty lists for allowed pairs [e,s]
        allowed_e = []
        allowed_s = []
        for h in self.pos_hierarchies:
            for e in exc_orders:
                # closed-shell system
                if self.nelec % 2 == 0:
                    if e % 2 == 0:  # if e is even
                        S_list = range(0, min(2 * e, self.nelec) + 2, 2)
                    elif e % 2 == 1:  # if e is odd
                        S_list = range(2, min(2 * e, self.nelec) + 2, 2)

                # open-shell system
                elif self.nelec % 2 == 1:
                    S_list = range(1, min(2 * e + 1, self.nelec) + 2, 2)

                # Calculate seniority number using heirarachy number formula
                s = (h - self.alpha1 * e) / self.alpha2

                if s >= 0 and s.is_integer() and s in S_list:
                    allowed_e.append(int(e))
                    allowed_s.append(int(s))

        e_s_pairs = list(zip(allowed_e, allowed_s))


        # ****************** GET SDS LIST FOR ALLOWED (e, s) PAIRS ******************
        ground_state = slater.ground(self.nelec, self.nspin)
        # Start an empty list with ground state
        allowed_sds = [ground_state]

        # Obtain list of Slater determinants for allowed (e, s) pairs
        # corresponding to given h, alpha1, alpha2
        for e, s in e_s_pairs:
            sd_ = sd_list(
                self.nelec, self.nspin, num_limit=None, exc_orders=[e], spin=None, seniority=s, hierarchy=True
            )

            allowed_sds.extend((sd_))
        if len(allowed_sds) == 1:
            raise Warning(
                "No compatible (e,s) pairs for given h, alpha1 & alpha2. Only ground state Slater determinant is allowed, proceeding with HF calculation."
            )
        allowed_sds = list(set(allowed_sds))
        print(f"[pattern={self.hci_pattern}] alpha1={self.alpha1:.3g}, alpha2={self.alpha2:.3g} | total determinants={len(allowed_sds)}")
        super().assign_sds(allowed_sds)


