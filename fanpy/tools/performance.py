r"""Functions for management of computational perfomance.

Functions
---------
binomial(n, k)
    Returns `n choose k`
permanent_combinatoric(matrix)
    Computes the permanent of a matrix using brute force combinatorics
permanent_ryser(matrix)
    Computes the permanent of a matrix using Ryser algorithm
adjugate(matrix)
    Returns adjugate of a matrix
permanent_borchardt(matrix)
    Computes the permanent of rank-2 Cauchy matrix

"""

import os
import psutil

# Memory Management Tools
def current_memory():
    """
    Obtain memory in use by the current Fanpy computational task.

    Return
    ----------
    current : Float
        Memory in use by Fanpy in Megabytes.
    """

    pid = os.getpid()
    process = psutil.Process(pid)
    current = process.memory_info().rss / 1024**2

    return current
