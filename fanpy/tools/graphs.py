r"""Functions for constructing perfect matchings of a given graph.

Functions
---------
generate_complete_pmatch(indices, sign=1)
    Generate all of the perfect matches of a complete (sub)graph.
generate_biclique_pmatch(indices_one, indices_two, ordered_set=None, is_decreasing=False)
    Generate all of the perfect matches of a complete bipartite (sub)graph.
generate_complete_partitions_dumb(indices, dimensions)
    Generate all partitions with the given dimensions of a complete (sub)graph.
int_partition_recursive(coins, num_coin_types, total)
    Generates the combination of coins that results in the given total.

"""
from fanpy.tools.slater import sign_perm

import numpy as np


def generate_complete_pmatch(indices, sign=1):
    """Generate all of the perfect matches of a complete (sub)graph.

    Generated perfect matches correspond to a pairing scheme in geminal wavefunctions and the
    signature is needed to find the sign of the Slater determinant after "unpacking" the pairing
    scheme.

    Parameters
    ----------
    indices : list of int
        List of indices of the vertices used to create the complete graph.

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match.
    sign : {1, -1}
        Signature of the transpositions required to shuffle the `pairing_scheme` back into the
        original order in `indices`.

    Notes
    -----
    The `sign` gives the signature with respect to the original `indices`. If the `indices` are not
    ordered, then signatures won't really mean anything.

    """
    # pylint: disable=C0103
    indices = tuple(indices)
    n = len(indices)
    if n % 2 == 1 or n < 2:
        yield tuple(), sign
    elif n == 2:
        yield ((indices[0], indices[1]),), sign
    else:
        # smaller subset (all pairs without the last two indices)
        subsets_pairs = generate_complete_pmatch(indices[:-2], sign=sign)
        for scheme, inner_sign in subsets_pairs:
            # add in the last two indices
            yield scheme + (indices[-2:],), inner_sign
            # starting from the last
            for i in reversed(range(n // 2 - 1)):
                # replace ith pair in the scheme with last pair
                yield (
                    scheme[:i]
                    + ((scheme[i][0], indices[-2]), (scheme[i][1], indices[-1]))
                    + scheme[i + 1 :]
                ), -inner_sign
                yield (
                    scheme[:i]
                    + ((scheme[i][0], indices[-1]), (scheme[i][1], indices[-2]))
                    + scheme[i + 1 :]
                ), inner_sign


def generate_biclique_pmatch(indices_one, indices_two, ordered_set=None, is_decreasing=False):
    """Generate all of the perfect matches of a complete bipartite (sub)graph.

    Parameters
    ----------
    indices_one : list of int
        List of indices of the vertices used to create the first half of the complete bipartite
        graph.
    indices_two : list of int
        List of indices of the vertices used to create the second half of the complete bipartite
        graph.
    ordered_set : {tuple, list}
        Set of indices ordered in strictly increasing order.
        If not provided, then the given indices are ordered.
    is_decreasing : bool
        If True, indices are ordered so that they are decreasing. (Sometimes creators are ordered
        from greatest to smallest).
        Default is False.

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match.
    sign : {1, -1}
        Signature of the transpositions required to shuffle the `pairing_scheme` back into the
        original order in `indices`.

    Notes
    -----
    The generator must be iterated to raise error.

    """
    # pylint: disable=C0103
    # assume indices_one and indices_two are sorted
    sign = 1
    orig_sign = sign_perm(
        [i for pair in zip(indices_one, indices_two) for i in pair],
        ordered_set=ordered_set,
        is_decreasing=is_decreasing,
    )

    if len(indices_one) == 0 or len(indices_two) == 0:
        yield tuple(), sign
    elif len(indices_one) != len(indices_two):
        yield tuple(), sign
    elif len(set(indices_one).symmetric_difference(set(indices_two))) < len(
        indices_one + indices_two
    ):
        yield tuple(), sign
    else:
        # for new_indices in it.permutations(indices_two):
        #     yield tuple(zip(indices_one, new_indices))
        # NOTE: This is the code when life was simpler. Now, permutations code need to be
        #       implemented to account for the signature changes
        # Following code was adapted from the example permutations code in itertools.permutations
        # documentations
        pool = tuple(indices_two)
        # pool is the set of object from which you will be selecting
        n = len(pool)
        indices = list(range(n))
        # indices select the specific ordering
        cycles = list(reversed(range(1, n + 1)))
        # cycles keeps track of the number of swaps and the positions of elements that are swapped
        yield (  # pragma: no branch
            tuple(zip(indices_one, (pool[i] for i in indices))),
            sign * orig_sign,
        )
        # NOTE: to obtain the signature, the jumbld pair structure must be unzipped, then sorted
        #       from largest to smallest. orig_sign accounts for this transposition/permutation
        while n:  # pragma: no branch
            for i in reversed(range(n)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    # move ith index to the end
                    indices[i:] = indices[i + 1 :] + indices[i : i + 1]
                    # in order to move ith element to the end, it must jump over n-i-1 elements
                    # (because i starts from 0)
                    sign *= (-1) ** (n - i - 1)
                    # reset cycles (back to its original number)
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    # swap
                    indices[i], indices[-j] = indices[-j], indices[i]
                    # change sign because swapping any two elements with x elements in between will
                    # require x+(x+1)=2x+1 swaps
                    sign *= -1
                    yield (  # pragma: no branch
                        tuple(zip(indices_one, (pool[i] for i in indices))),
                        sign * orig_sign,
                    )
                    break
            else:
                return


# TODO: add sign for reordering the partitions back into original ordering?
def generate_unordered_partition(collection, bin_size_num):
    """Generate unordered partitions of the given collection into subsets of the given sizes.

    We can think of the partition as distributing the given elements into bins. Each bin has a fixed
    size and is full at the end of the distribution. In an unordered partition, the bins of the same
    size are not distinguishable. Additionally, there is no ordering within each bin.

    In order to avoid repeating the same partitions, we follow some conventions such that the
    repeated partitions resulting from reordering are not possible.

        1. Elements within a bin are ordered. For example, bin `[1, 2, 3]` has ordered elements
           where as `[1, 3, 2]` does not.
        2. First elements of bins of equal sizes are ordered. For example, the bins `[1, 4]` and
           `[2, 3]` are ordered if `[[1, 4], [2, 3]]` but are unordered if `[[2, 4], [1, 4]]`.

    Parameters
    ----------
    collection : list
        List of elements that will be partitioned.
        This should be an ordered list to ensure a correct number of partitions and correct ordering of the bins. 
    bin_size_num : list of 2-tuple of int
        List of tuples that describe the size and the number of the bins.
        First element of the tuple is the size of the bin.
        Second element of the tuple is the number of bins of the given size.

    Examples
    --------
    >>> unordered_partition([1, 2, 3], [(2, 1), (1, 1)])
    [[[1, 2], [3]], [[1, 3], [2]], [[2, 3], [1]]]

    """
    if len(collection) == 0:
        yield [[] for _, bin_size in bin_size_num for i in range(bin_size)]
        return

    last = collection[-1]
    # loop over all of the partitions with N-1 items (last item omitted)
    for prev_partition in generate_unordered_partition(collection[:-1], bin_size_num):
        ind_bin = -1  # index of the bin with respect to the all of the bins
        ind_size = 0  # index of the size of the bin
        ind_bin_size = -1  # index of the bin with respect to the bin of this size
        while ind_size < len(bin_size_num):
            ind_bin += 1
            ind_bin_size += 1
            # if we run out of bins of the given size
            if ind_bin_size == bin_size_num[ind_size][1]:
                # continue to the next size
                ind_size += 1
                # reset ind_bin_size
                ind_bin_size = 0
                # break out of while loop if there are no more bins
                if ind_size == len(bin_size_num):
                    break

            # select the bin
            subset = prev_partition[ind_bin]

            # ensure that the first elements from each subset is ordered
            if len(subset) == 0:
                # element can go into the empty subset/bin
                yield prev_partition[:ind_bin] + [subset + [last]] + prev_partition[ind_bin + 1 :]
                # if there are more than empty bins of the same size
                if bin_size_num[ind_size][1] > 1:  # pragma: no branch
                    # NOTE: If the subset/bin is empty, all subsequent subsets/bins of the same size
                    #       must also be empty (because the bins are always filled from left to
                    #       right)
                    #       We can skip filling the bins of the same size because the bins of the
                    #       same sizes are not distinguishable, i.e. unordered partition
                    # skip to the next bin that has a different size
                    ind_bin += bin_size_num[ind_size][1] - ind_bin_size - 1
                    ind_size += 1
                    ind_bin_size = -1
                continue

            # ensure that elements in each bin are ordered
            if not subset[-1] < last:  # pragma: no cover
                continue

            # ensure that the number of elements in each bin does not exceed limit
            if not len(subset) < bin_size_num[ind_size][0]:
                continue

            # add the last element to the selected bin
            yield prev_partition[:ind_bin] + [subset + [last]] + prev_partition[ind_bin + 1 :]


# FIXME: make this dynamic or store/cache some of the results on file
def int_partition_recursive(coins, num_coin_types, total):
    """Generate the combination of coins that results in the given total.

    Known as the coin problem, we can find different ways of dividing up a given number (e.g. number
    of electrons) as a sum of smaller numbers from a set (e.g. quasiparticles).

    Parameters
    ----------
    coins : int
        Values of the different coins that will be used to produce the total.
        Should not have repetitions.
        Should be ordered from smallest to largest.
    num_coin_types : int
        Number of different coins that can be used to produce the total.
        If the number of different coins is less than the total number of coins, only the first
        `num_coin_types` will be used.
    total : int
        Total sum of the coin values.

    Yields
    ------
    partition : list of int
        List of coins that sum up to the total.

    References
    ----------
    https://www.geeksforgeeks.org/dynamic-programming-set-7-coin-change/

    """
    # if total is zero
    if total == 0:
        yield []

    # if total is less than zero
    if total <= 0:
        return

    # if no coins can be used
    if num_coin_types <= 0:
        return

    # include last coin
    for partition in int_partition_recursive(
        coins, num_coin_types, total - coins[num_coin_types - 1]
    ):
        yield [coins[num_coin_types - 1]] + partition
    # exclude last coin
    yield from int_partition_recursive(coins, num_coin_types - 1, total)


def generate_general_pmatch(indices, connectivity_matrix):
    """Generate perfect matching of the given indices for the given graph.

    Parameters
    ----------
    indices : np.array(N)
        List of indices of the vertices used to create the complete graph.
    connectivity_matrix : np.ndarray(N, N)
        Boolean connectivity matrix indicating which vertices form an edge.
        Each row/column corresponds to the vertex in the indices.

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match.
    sign : {1, -1}
        Signature of the transpositions required to shuffle the `pairing_scheme` back into the
        original order in `indices`.

    """
    if isinstance(indices, (list, tuple)):
        indices = np.array(indices)
    if len(indices) == 2:
        yield [(indices[0], indices[1])], 1
    elif indices.size > 2:
        ind_one = indices[0]
        for j in np.where(connectivity_matrix[0, 1:])[0]:
            sign = (-1) ** j
            j += 1
            ind_two = indices[j]
            # filter out indices that are not used
            mask_bool = ~np.isin(indices, [ind_one, ind_two])
            mask_ind = np.where(mask_bool)[0]
            mask_ind = mask_ind[mask_ind > 0]

            for scheme, inner_sign in generate_general_pmatch(
                indices[mask_ind], connectivity_matrix[mask_ind[:, None], mask_ind[None, :]]
            ):
                yield [(ind_one, ind_two)] + scheme, sign * inner_sign
