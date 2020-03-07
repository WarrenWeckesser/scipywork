import numpy as np


def _nway_groups(*factors, values, levels=None):
    """
    Parameters
    ----------
    factors : one or more 1-d sequences of values
        The factors (i.e. the independent variables) to be analyzed.
        Generally these should be integers or categorical values.
    values : 1-d sequence
        The values associated with the corresponding factors.
        A real-valued array.

    Examples
    --------
    >>> x = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    >>> y = [5, 5, 5, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 5, 5]
    >>> z = [1.4, 2.0, 1.8, 1.7, 1.6, 1.8, 2.1, 2.0, 2.1,
    ...      1.9, 2.4, 2.3, 2.3, 2.9, 2.8]
    >>> levels, groups = _nway_groups(x, y, values=z)

    `levels` is the unique values in each factor.  As is easily
    verified, we find three distinct values in `x` and two in `y`.

    >>> levels
    (array([0, 1, 2]), array([5, 7]))

    `groups` is an object array with shape (3, 2).  Each value in
    the array is a 1-d sequence of values from `z`.
 
    >>> for i in range(len(groups)):
    ...      for j in range(groups.shape[1]):
    ...          print(i, j, (levels[0][i], levels[1][j]), groups[i, j])
    ...
    0 0 (0, 5) [1.4 2.  1.8]
    0 1 (0, 7) [1.7 1.6 1.8]
    1 0 (1, 5) [2.1 1.9]
    1 1 (1, 7) [2.1 2. ]
    2 0 (2, 5) [2.9 2.8]
    2 1 (2, 7) [2.4 2.3 2.3]
    """
    factors = [np.asarray(a) for a in factors]
    values = np.asarray(values)
    if len(factors) == 0:
        raise TypeError("At least one input factor is required.")
    if not all(len(a) == len(factors[0]) for a in factors[1:]):
        raise ValueError("All input factors must sequences with the same length.")
    if len(values) != len(factors[0]):
        raise ValueError('values must have the same length as each factor.')

    if levels is None:
        # Call np.unique with return_inverse=True on each factor.
        actual_levels, inverses = zip(*[np.unique(f, return_inverse=True)
                                        for f in factors])
        shape = [len(u) for u in actual_levels]
        groups = np.empty(shape, dtype=object)
        inverses = np.array(inverses)
        u, idx = np.unique(inverses, axis=1, return_inverse=True)
        u = u.T
        for i in range(len(u)):
            groups[tuple(u[i])] = values[idx == i]
    else:
        raise NotImplementedError('specifying levels is not implemented yet.')
        # `levels` is not None...
        if len(levels) != len(factors):
            raise ValueError('len(levels) must equal the number of input '
                             'sequences')

        factors = [np.asarray(factor) for factor in factors]
        mask = np.zeros((len(factors), len(factors[0])), dtype=np.bool_)
        inv = np.zeros((len(factors), len(factors[0])), dtype=np.intp)
        actual_levels = []
        for k, (levels_list, arg) in enumerate(zip(levels, factors)):
            if levels_list is None:
                levels_list, inv[k, :] = np.unique(arg, return_inverse=True)
                mask[k, :] = True
            else:
                q = arg == np.asarray(levels_list).reshape(-1, 1)
                mask[k, :] = np.any(q, axis=0)
                qnz = q.T.nonzero()
                inv[k, qnz[0]] = qnz[1]
            actual_levels.append(levels_list)

        mask_all = mask.all(axis=0)
        shape = [len(u) for u in actual_levels]
        count = np.zeros(shape, dtype=int)
        indices = tuple(inv[:, mask_all])
        np.add.at(count, indices, 1)

    return actual_levels, groups
