r"""
FanCI alias random number generator method module.

"""

import numpy as np


__all__ = [
    "Alias",
]


class Alias:
    r"""
    Sample a discrete probability distribution to generate random indices.

    Alias method from Walker, A. J. (1974). New fast method for generating
    discrete random numbers with arbitrary frequency distributions.
    Electronics Letters, 8(10), 127-128.

    """

    def __init__(self, pvec: np.ndarray) -> None:
        r"""
        Initialize the comparison and index tables for the Alias method.

        Parameters
        ----------
        pvec: np.ndarray
            Probability vector.

        """
        # Declare size of probability vector
        self.n = pvec.size
        # Make comparison and index tables
        self.cvec = np.asarray(pvec, dtype=float)
        self.cvec *= self.n / np.sum(pvec)
        self.ivec = np.empty(self.n, dtype=int)
        # Sort the data into greater of lesser than 1 / k
        lesser, greater = [], []
        for i, c in enumerate(self.cvec):
            (greater if c > 1 else lesser).append(i)
        # Populate index table
        while len(lesser) > 0 and len(greater) > 0:
            l = lesser.pop()
            g = greater.pop()
            self.ivec[l] = g
            self.cvec[g] = self.cvec[g] + self.cvec[l] - 1
            (greater if self.cvec[g] > 1 else lesser).append(g)

    def __call__(self, n: int) -> np.ndarray:
        r"""
        Generate random indices from `pvec` via the Alias method.

        Parameters
        ----------
        n: int
            Number of random indices to generate.

        Returns
        -------
        out: np.ndarray
            Array of random indices.

        """
        out = set()
        while len(out) < n:
            i = int(np.random.rand() * self.n)
            out.add(self.ivec[i] if np.random.rand() > self.cvec[i] else i)
        return np.array(sorted(out), dtype=int)
