import numpy as np
cimport numpy as np

cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_energy_one_proj_deriv(wfn, ham, list pspace):
    cdef double[:, ::1] one_int = ham.one_int
    # FIXME: move to Hamiltonian class
    cdef double[:, ::1] two_int_ijij = np.ascontiguousarray(ham._cached_two_int_ijij)
    cdef double[:, ::1] two_int_ijji = np.ascontiguousarray(ham._cached_two_int_ijji)
    cdef double[:, :, :, ::1] two_int = ham.two_int

    cdef double overlap
    cdef double integral
    cdef double norm = 0
    cdef double energy = 0

    cdef int pspace_size = len(pspace)
    cdef int wfn_nparams = wfn.nparams
    cdef int ham_nparams = ham.nparams

    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=1] d_overlap
    cdef np.ndarray[np.double_t, ndim=1] d_integral_wfn
    cdef np.ndarray[np.double_t, ndim=1] d_integral_ham
    cdef np.ndarray[np.double_t, ndim=1] d_norm = np.zeros(wfn_nparams)
    cdef np.ndarray[np.double_t, ndim=1] d_energy = np.zeros(wfn_nparams + ham_nparams)
    for i in range(pspace_size):
        overlap = wfn.get_overlap(pspace[i])
        integral = np.sum(ham.integrate_sd_wfn(pspace[i], wfn))

        energy += overlap * integral
        norm += overlap ** 2

        d_overlap = wfn.get_overlap(pspace[i], True)
        d_integral_wfn = np.sum(ham.integrate_sd_wfn(pspace[i], wfn, wfn_deriv=True), axis=0)
        d_norm += 2 * overlap * d_overlap
        d_energy[:wfn_nparams] += d_overlap * integral
        d_energy[:wfn_nparams] += overlap * d_integral_wfn

        d_integral_ham = np.sum(
            ham.integrate_sd_wfn_deriv(pspace[i], wfn, np.arange(ham.nparams)), axis=0
        )
        d_energy[wfn_nparams:] += overlap * d_integral_ham
    return energy, d_energy, norm, d_norm

    # energy /= norm
    # d_energy /= norm
    # d_energy[:wfn_nparams] -= d_norm * energy/ norm

    # return energy, d_energy
