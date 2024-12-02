"""CI Singles and Doubles Wavefunction."""
from fanpy.tools.sd_list import sd_list
from fanpy.tools import slater
from fanpy.wfn.ci.base import CIWavefunction



class hCI(CIWavefunction):
    r"""Hierarchy Configuration Interaction Wavefunction.

    CI with restricted excitation number and seniority number.

    hierarchy parameter
        :math: `h = \alpha_1 e + \alpha_2 s`,
        where, 'e' is excitation degree & 's' is the seniority number.

    ***Reference: Hierarchy Configuration Interaction: Combining Seniority Number
                  and Excitation Degree by Fabris Kossoski, et.al.
                  https://doi.org/10.48550/arXiv.2203.06154

    Extra Attributes for hCI
    ----------
    h  : float
        hierarchy number
    alpha1 :  float
        parameter to control excitation order
    alpha2 :  float
        parameter to control seniority

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
    assign_hierarchy(self, hierarchy=None)
        Assign the hierarchy number of the wavefunction.
    assign_alphas(self, alpha1, alpha2)
        Assign the weight coefficient 'alpha1' to limit the excitation
        orders and 'alpha2' to restrict the seniority.

    """
    def __init__(
        self, nelec, nspin,alpha1, alpha2,params=None, sds=None, memory=None, hierarchy=None, refwfn=None):
        self.assign_hierarchy(hierarchy=hierarchy)
        self.assign_alphas(alpha1=alpha1, alpha2=alpha2)
        super().__init__(nelec, nspin, memory=memory, params=params)

        self.assign_refwfn(refwfn=refwfn)
        self.assign_sds(sds=sds)


    def assign_hierarchy(self, hierarchy=2.5):
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
        #elif hierarchy is None:
        #    self.sds = None
        else:
            self.hierarchy = hierarchy



    def assign_alphas(self, alpha1=0.5, alpha2=0.25):
        """ Assign the alpha1 and alpha2 parameters for defined hierachy number to restrict
            excitation orders and seniority, respectively.

        Parameters
        ----------
        alpha1 : {int, float}
        alpha2 : {int, float}

        Raises
        ------
        TypeError
            If the alpha is not an integer or None.
        ValueError
            If the alpha is a negative number or greater than 1.

        """

        alphas = [alpha1, alpha2]
        if __debug__:
            for alpha in alphas:
                if not isinstance(alpha, (int, float)):
                    raise TypeError("alpha must be an integer or float.")
                if alpha <-1 or alpha > 1:
                    raise ValueError("`alpha` must be greater than or equal to -1 and less than or equal to 1..")

        self.alpha1 = alpha1
        self.alpha2 = alpha2


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
                "CIWavefunction instead.")

        #******************* GET LISTS FOR ALLOWED (e, s) PAIRS ******************
        # Here, the way of obtaining (e, s) pairs is only for the reference systems
        # having maximum number of paired electrons, so the seniority of the reference
        # should be either 0 or 1.

        # np.array of allowed excitation orders for a given 'nelec'
        exc_orders = range(0, self.nelec + 1, 1)

        # Initialize the empty lists for allowed pairs [e,s]
        allowed_e = []
        allowed_s = []

        for e in exc_orders:
            # closed-shell system
            if self.nelec % 2 == 0:
                if e % 2 == 0: #if e is even
                    S_list = range(0, min(2 * e, self.nelec)+2, 2)
                elif e % 2 == 1: #if e is odd
                    S_list = range(2, min(2 * e, self.nelec)+2, 2)

            # open-shell system
            elif self.nelec % 2 == 1:
                S_list = range(1, min(2 * e + 1, self.nelec)+2,2)

            # Calculate seniority number using heirarachy number formula
            s = (self.hierarchy - self.alpha1 * e)/self.alpha2

            #print(e, s)
            if s >= 0 and s.is_integer() and s in S_list:
                allowed_e.append(int(e))
                allowed_s.append(int(s))

        e_s_pairs = list(zip(allowed_e, allowed_s))


        #****************** GET SDS LIST FOR ALLOWED (e, s) PAIRS ******************
        ground_state = slater.ground(self.nelec, self.nspin)
        # Start an empty list with ground state
        allowed_sds = [ground_state]

        # Obtain list of Slater determinants for allowed (e, s) pairs
        # corresponding to given h, alpha1, alpha2
        for e, s in e_s_pairs:
            sd_ = sd_list(self.nelec,
                self.nspin,
                num_limit=None,
                exc_orders=[e],
                spin = None,
                seniority=s)

            # Remove ground state Slater determinant from the list before appending
            del sd_[0]

            # Check if list contains same Slater determinant multiple times
            if len(sd_) == len(set(sd_)):
                # Add the allowed Slater determinants to the list
                allowed_sds.extend(sd_)
            else:
                tmp_ = []
                for j in sd_:
                    if j not in tmp_:
                        tmp_.append(j)
                allowed_sds.extend(tmp_)

        if len(allowed_sds) == 1:
            raise Warning("No compatible (e,s) pairs for given h, alpha1 & alpha2. Only ground state Slater determinant is allowed, proceeding with HF calculation.")

        allowed_sds = [195, 198, 197, 202, 201, 210, 209, 226, 225, 387, 323, 643, 579, 1155,                 1091, 2179, 2115, 204, 212, 216, 228, 232, 240, 390, 326, 646, 582, 1158, 1094, 2182,             2118, 389, 325, 645, 581, 1157, 1093, 2181, 2117, 394, 330, 650, 586, 1162, 1098,                 2186, 2122, 393, 329, 649, 585, 1161, 1097, 2185, 2121, 402, 338, 658, 594, 1170,                 1106, 2194, 2130, 401, 337, 657, 593, 1169, 1105, 2193, 2129, 418, 354, 674, 610,                 1186, 1122, 2210, 2146, 417, 353, 673, 609, 1185, 1121, 2209, 2145, 771, 1283, 1539,               2307, 2563, 3075, 396, 332, 652, 588, 1164, 1100, 2188, 2124, 404, 340, 660, 596,                 1172, 1108, 2196, 2132, 408, 344, 664, 600, 1176, 1112, 2200, 2136, 420, 356, 676,                 612, 1188, 1124, 2212, 2148, 424, 360, 680, 616, 1192, 1128, 2216, 2152, 432, 368,                 688, 624, 1200, 1136, 2224, 2160, 774, 1286, 1542, 2310, 2566, 3078, 773, 1285, 1541,             2309, 2565, 3077, 778, 1290, 1546, 2314, 2570, 3082, 777, 1289, 1545, 2313, 2569,                 3081, 786, 1298, 1554, 2322, 2578, 3090, 785, 1297, 1553, 2321, 2577, 3089, 802, 1314,             1570, 2338, 2594, 3106, 801, 1313, 1569, 2337, 2593, 3105, 780, 1292, 1548, 2316,                 2572, 3084, 788, 1300, 1556, 2324, 2580, 3092, 792, 1304, 1560, 2328, 2584, 3096, 804,             1316, 1572, 2340, 2596, 3108, 808, 1320, 1576, 2344, 2600, 3112, 816, 1328, 1584,                 2352, 2608, 3120]
        print(allowed_sds)
        super().assign_sds(allowed_sds)
        print(f"Number of total slater determinants = ", {len(allowed_sds)})


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
                raise ValueError('refwfn must have {} electrons'.format(self.nelec))
            # TODO: check that refwfn has the right number of spin-orbs
            self.refwfn = refwfn
        else:
            if not isinstance(refwfn, CIWavefunction):
                raise TypeError('refwfn must be a CIWavefunction or a int object')
            if not hasattr(refwfn, 'sds'):  # NOTE: Redundant test.
                raise AttributeError('refwfn must have the sds attribute')
            if refwfn.nelec != self.nelec:
                raise ValueError('refwfn must have {} electrons'.format(self.nelec))
            if refwfn.nspin != self.nspin:
                raise ValueError('refwfn must have {} spin orbitals'.format(self.nspin))
            self.refwfn = refwfn




