from collections import namedtuple
from types import SimpleNamespace
import numpy as np
from scipy import special


__all__ = [
    'AnovaOnewayResult',
    'anova_oneway', 'anova_oneway_from_xy',
    'anova_oneway_ci_diffs', 'anova_oneway_ci_means',
    'AnovaTwowayResult',
    'anova_twoway1', 'anova_twoway_balanced', 'anova_twoway_unbalanced',
    'anova_twoway_from_x_y_values'
]


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
        if len(levels) != len(args):
            raise ValueError('len(levels) must equal the number of input '
                             'sequences')

        args = [np.asarray(arg) for arg in args]
        mask = np.zeros((len(args), len(args[0])), dtype=np.bool_)
        inv = np.zeros((len(args), len(args[0])), dtype=np.intp)
        actual_levels = []
        for k, (levels_list, arg) in enumerate(zip(levels, args)):
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


class AnovaOnewayResult(SimpleNamespace):
    
    def __str__(self):
        vartot = self.SSb + self.SSw
        dftot = self.DFb + self.DFw
        mstot = vartot / dftot
        s = "\n".join([
                "ANOVA one-way",
                "Source                   SS  DF          MS        F       p",
                f"Between groups {self.SSb:12.5f} {self.DFb:3}  {self.MSb:10.3f} {self.F:8.3f} {self.p:<10.5g}",
                f"Within groups  {self.SSw:12.5f} {self.DFw:3}  {self.MSw:10.3f}",
                f"Total          {vartot:12.5f} {dftot:3}  {mstot:10.3f}"])
        return s


def anova_oneway(*args, **kwds):
    num_groups = len(args)
    groups = [np.asarray(arg, dtype=np.float64) for arg in args]
    means = [group.mean() for group in groups]
    n = 0
    grand_total = 0
    for group in groups:
        n += len(group)
        grand_total += group.sum()
    grand_mean = grand_total / n

    v = sum(((group - grand_mean)**2).sum() for group in groups)
    vb = sum(len(group)*(group.mean() - grand_mean)**2 for group in groups)
    vw = v - vb

    dfb = num_groups - 1
    dfw = n - num_groups
    msb = vb / dfb
    msw = vw / dfw
    F = msb / msw
    dof_num = num_groups - 1
    dof_den = n - num_groups
    p = special.fdtrc(dof_num, dof_den, F)
    result = AnovaOnewayResult(
                mean=grand_mean,
                group_means=means,
                group_sizes=[len(g) for g in groups],
                SSb=vb, SSw=vw,
                DFb=dfb, DFw=dfw,
                MSb=msb, MSw=msw,
                F=F, p=p)
    return result


def anova_oneway_from_xy(x, y):
    levels, idx = np.unique(x, return_inverse=True)
    groups = [y[idx == i] for i in range(len(levels))]
    result = anova_oneway(*groups)
    result.levels = levels
    return result


def anova_oneway_ci_diffs(result: AnovaOnewayResult, alpha):
    """
    Confidence intervals for the differences of the means.
    """
    means = result.group_means
    ngroups = len(means)
    for i in range(ngroups - 1):
        for j in range(i + 1, ngroups):
            t = special.stdtrit(result.DFw, 1 - alpha/2)
            c = np.sqrt(result.MSw * (1/result.group_sizes[i] + 1/result.group_sizes[j]))
            print(i, j, means[i] - means[j], t*c)


def anova_oneway_ci_means(result: AnovaOnewayResult, alpha):
    """
    Confidence intervals for the means of the groups.
    """
    means = result.group_means
    ngroups = len(means)
    for i in range(ngroups):
        t = special.stdtrit(result.DFw, 1 - alpha/2)
        c = np.sqrt(result.MSw * (1/result.group_sizes[i]))
        delta = t*c
        print(i, means[i], means[i] - delta, means[i] + delta)


class AnovaTwowayResult(SimpleNamespace):
    
    def __str__(self):
        if hasattr(self, 'SS01'):
            vartot = self.SSA + self.SSB + self.SS01 + self.SSerror
            dftot = self.DFA + self.DFB + self.DF01 + self.DFerror
            s = "\n".join([
                    "ANOVA two-way",
                    "Source                   SS  DF          MS        F       p",
                    f"Factor A       {self.SSA:12.5f} {self.DFA:3}  {self.MSA:10.3f} {self.FA:8.3f} {self.pA:<10.5g}",
                    f"Factor B       {self.SSB:12.5f} {self.DFB:3}  {self.MSB:10.3f} {self.FB:8.3f} {self.pB:<10.5g}",
                    f"Interaction    {self.SS01:12.5f} {self.DF01:3}  {self.MS01:10.3f} {self.F01:8.3f} {self.p01:<10.5g}",
                    f"Error          {self.SSerror:12.5f} {self.DFerror:3}  {self.MSerror:10.3f}",
                    f"Total          {vartot:12.5f} {dftot:3}"])
        else:
            # No interaction term.
            # XXX Clean up duplicated code.
            vartot = self.SSA + self.SSB + self.SSerror
            dftot = self.DFA + self.DFB + self.DFerror
            s = "\n".join([
                    "ANOVA two-way",
                    "Source                   SS  DF          MS        F       p",
                    f"Factor A       {self.SSA:12.5f} {self.DFA:3}  {self.MSA:10.3f} {self.FA:8.3f} {self.pA:<10.5g}",
                    f"Factor B       {self.SSB:12.5f} {self.DFB:3}  {self.MSB:10.3f} {self.FB:8.3f} {self.pB:<10.5g}",
                    f"Error          {self.SSerror:12.5f} {self.DFerror:3}  {self.MSerror:10.3f}",
                    f"Total          {vartot:12.5f} {dftot:3}"])
        return s


def anova_twoway_balanced(data):
    """
    Two-way ANOVA with balanced replication.

    The number of replications for each pair of factors must be the same.

    Parameters
    ----------
    data : array-like, either (m, n) or (m, n, r)
        `r` is the number of replications.  If `data` has shape (m, n),
        it implies `r` is 1.

    """
    # In the following, the two factors are "labeled" A and B.

    data = np.asarray(data)
    shp = data.shape
    if data.ndim == 2:
        raise ValueError("ndim = 2 not implemented yet, use anova_twoway1 instead.")

    grand_mean = data.mean()

    #mean0 = data.mean(axis=(1,2), keepdims=True)
    mean2 = data.mean(axis=2, keepdims=True)
    mean01 = data.mean(axis=(0,1), keepdims=True)

    #mean02 = data.mean(axis=(0,2), keepdims=True)
    meanB = data.mean(axis=(0,2), keepdims=True)
    #mean12 = data.mean(axis=(1,2), keepdims=True)
    meanA = data.mean(axis=(1,2), keepdims=True)

    ss_total = ((data - grand_mean)**2).sum()
    dof_total = shp[0]*shp[1]*shp[2] - 1

    ss_repl  = shp[0]*shp[1]*((mean01 - grand_mean)**2).sum()
    dof_repl = shp[2] - 1
    ms_repl  = ss_repl / dof_repl

    ssB    = shp[0]*shp[2]*((meanB - grand_mean)**2).sum()
    dofB   = shp[1] - 1
    msB    = ssB / dofB

    ssA    = shp[1]*shp[2]*((meanA - grand_mean)**2).sum()
    dofA   = shp[0] - 1
    msA    = ssA / dofA

    ss_inter  = shp[2]*((mean2 - meanA - meanB + grand_mean)**2).sum()
    dof_inter = (shp[0] - 1)*(shp[1] - 1)
    ms_inter  = ss_inter / dof_inter

    # These are from R. Johnson "Miller & Freund's Prob. & Stats for Engineers"
    #ss_error  = ((data - mean2 - mean01 + grand_mean)**2).sum()
    #dof_error = (shp[0]*shp[1] - 1)*(shp[2] - 1)
    # These are from Zar (fifth ed.)
    ss_error  = ((data - mean2)**2).sum()
    dof_error = (shp[0]*shp[1])*(shp[2] - 1)
    ms_error  = ss_error / dof_error

    F_repl  = ms_repl / ms_error
    FB      = msB / ms_error
    FA      = msA / ms_error
    F_inter = ms_inter / ms_error

    p_repl  = special.fdtrc(dof_repl, dof_error, F_repl)

    pA    = special.fdtrc(dofA, dof_error, FA)

    pB    = special.fdtrc(dofB, dof_error, FB)

    p_inter = special.fdtrc(dof_inter, dof_error, F_inter) 

    #print("                       SS  DF          MS        F       p")
    #print(f"Replicates   {ss_repl:12.5f} {dof_repl:3}  {ms_repl:10.3f} {F_repl:8.3f} {p_repl:<10.5g}")
    #print(f"Factor 0     {ss_12:12.5f} {dof_12:3}  {ms_12:10.3f} {F_12:8.3f} {p_12:<10.5g}")
    #print(f"Factor 1     {ss_02:12.5f} {dof_02:3}  {ms_02:10.3f} {F_02:8.3f} {p_02:<10.5g}")
    #print(f"Interaction  {ss_inter:12.5f} {dof_inter:3}  {ms_inter:10.3f} {F_inter:8.3f} {p_inter:<10.5g}")
    #print(f"Error        {ss_error:12.5f} {dof_error:3}  {ms_error:10.3f}")
    #print(f"Total        {ss_total:12.5f} {dof_total:3}")

    result = AnovaTwowayResult(SSB=ssB, SSA=ssA, SS01=ss_inter, SSerror=ss_error,
                               DFB=dofB, DFA=dofA, DF01=dof_inter, DFerror=dof_error,
                               MSB=msB, MSA=msA, MS01=ms_inter, MSerror=ms_error,
                               FB=FB, FA=FA, F01=F_inter,
                               pB=pB, pA=pA, p01=p_inter)
    return result


def anova_twoway1(data):
    """
    Two-way anova without replication.

    Parameters
    ----------
    data : array-like with shape (m, n)

    """
    data = np.asarray(data)
    shp = data.shape
    if data.ndim != 2:
        raise ValueError("This function is for two-way ANOVA with no replication.")
    r, c = shp

    # Work in progress...

    grand_mean = data.mean()
    mean0 = data.mean(axis=0, keepdims=True)
    mean1 = data.mean(axis=1, keepdims=True)

    ss_total = ((data - grand_mean)**2).sum()
    dof_total = shp[0]*shp[1] - 1

    ss0 = r*((mean0 - grand_mean)**2).sum()
    ss1 = c*((mean1 - grand_mean)**2).sum()

    df0 = c - 1
    df1 = r - 1
    ms0 = ss0 / df0
    ms1 = ss1 / df1

    sse = ss_total - ss0 - ss1
    dfe = (c - 1)*(r - 1)
    mse = sse / dfe

    F1 = ms1 / mse
    F0 = ms0 / mse

    p1 = special.fdtrc(df1, dfe, F1)
    p0 = special.fdtrc(df0, dfe, F0)

    #print("          SS   DF           MS            F          p")
    #print(f"{ss1:12.5f}  {df1:3} {ms1:12.5f} {F1:12.5f} {p1:10.6f}")
    #print(f"{ss0:12.5f}  {df0:3} {ms0:12.5f} {F0:12.5f} {p0:10.6f}")
    #print(f"{sse:12.5f}  {dfe:3} {mse:12.5f}")
    #print(f"{ss_total:12.5f}")

    result = AnovaTwowayResult(SSB=ss0, SSA=ss1, SSerror=sse,
                               DFB=df0, DFA=df1, DFerror=dfe,
                               MSB=ms0, MSA=ms1, MSerror=mse,
                               FB=F0, FA=F1,
                               pB=p0, pA=p1)
    return result

def anova_twoway_unbalanced(data):
    """
    Two-way anova with unbalanced replication.

    Parameters
    ----------
    data : 2-d array-like with cells that are 1-d array-like

    """
    nrows = len(data)
    ncols = np.array([len(row) for row in data])
    if not np.all(ncols == ncols[0]):
        raise ValueError("all rows must have the same length.")
    ncols = ncols[0]

    celln = np.zeros((nrows, ncols), dtype=int)

    cellsums = np.zeros((nrows, ncols))
    cellmeans = np.zeros((nrows, ncols))
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            celln[i, j] = len(cell)
            cellsums[i, j] = sum(cell)
            cellmeans[i, j] = np.mean(cell)

    rowsums = cellsums.sum(axis=1, keepdims=True)
    rowmeans = rowsums / celln.sum(axis=1, keepdims=True)

    colsums = cellsums.sum(axis=0, keepdims=True)
    colmeans = colsums / celln.sum(axis=0, keepdims=True)

    grand_mean = cellsums.sum() / celln.sum()

    v = sum(sum(sum((x - grand_mean)**2
                    for x in cell)
                for cell in row)
            for row in data)
    dfv = celln.sum() - 1

    vr = sum(sum(sum((rmean - grand_mean)**2
                     for x in cell)
                 for cell in row)
             for rmean, row in zip(rowmeans[:,0], data))
    dfvr = nrows - 1

    vc = sum(sum(sum((cmean - grand_mean)**2
                     for x in cell)
                 for cmean, cell in zip(colmeans[0], row))
             for row in data)
    dfvc = ncols - 1

    vi = 0.0
    for i in range(nrows):
        for j in range(ncols):
            vi += celln[i,j]*(cellmeans[i,j] - rowmeans[i,0] - colmeans[0,j] + grand_mean)**2
    dfvi = (nrows - 1)*(ncols - 1)

    ve = sum(sum(sum((x - cellmean)**2
                     for x in cell)
                 for cellmean, cell in zip(cmean, row))
             for cmean, row in zip(cellmeans, data))
    dfve = dfv - dfvc - dfvr - dfvi

    ms = v/dfv
    msr = vr/dfvr
    msc = vc/dfvc
    msi = vi/dfvi
    mse = ve/dfve
    ft = ms/mse
    fr = msr/mse
    fc = msc/mse
    fi = msi/mse
    pr = special.fdtrc(dfvr,dfve,fr)
    pc = special.fdtrc(dfvc,dfve,fc)
    pi = special.fdtrc(dfvi,dfve,fi)

    #print("                 SS   DF           MS            F            p")
    #print(f"rows:  {vr:12.5f}  {dfvr:3d} {msr:12.5f} {fr:12.5f} {pr:12.5f}")
    #print(f"cols:  {vc:12.5f}  {dfvc:3d} {msc:12.5f} {fc:12.5f} {pc:12.5f}")
    #print(f"inter: {vi:12.5f}  {dfvi:3d} {msi:12.5f} {fi:12.5f} {pi:12.5f}")
    #print(f"error: {ve:12.5f}  {dfve:3d} {mse:12.5f}")
    #print(f"total: {v:12.5f}  {dfv:3d}")

    result = AnovaTwowayResult(SSB=vc, SSA=vr, SS01=vi, SSerror=ve,
                               DFB=dfvc, DFA=dfvr, DF01=dfvi, DFerror=dfve,
                               MSB=msc, MSA=msr, MS01=msi, MSerror=mse,
                               FB=fc, FA=fr, F01=fi,
                               pB=pc, pA=pr, p01=pi)
    return result


def anova_twoway_from_x_y_values(x, y, values):
    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')
    if len(values) != len(x):
        raise ValueError('values must have the same length as x.')
    levels, groups = _nway_groups(x, y, values=values)
    return anova_twoway_unbalanced(groups)    


def anova_twoway(data):
    """
    Two-way ANOVA.

    Perform the two-way (or two factor) analyis of variance calculation.

    Parameters
    ----------
    data : array_like with shape (m, n) or (m, n, r), OR nested sequences
           that looks like an array with shape (m, n, *), where * represents
           a dimension along which the lengths may be different.

    """
    # TO DO: handle all the various input formats of data.
    pass


def anova_threeway_balanced(data):
    """
    Three-way ANOVA for balanced inputs.

    Parameters
    ----------
    data : array_like, shape (m, n, p, r)
        r is the number of replicates.
    """
    data = np.asarray(data)
    shp = data.shape
    n = data.size
    a, b, c, r = shp

    grand_mean = data.mean()
    mean3 = data.mean(axis=3, keepdims=True)
    mean01 = data.mean(axis=(0,1), keepdims=True)
    mean02 = data.mean(axis=(0,2), keepdims=True)
    mean03 = data.mean(axis=(0,3), keepdims=True)
    mean12 = data.mean(axis=(1,2), keepdims=True)
    mean13 = data.mean(axis=(1,3), keepdims=True)
    mean23 = data.mean(axis=(2,3), keepdims=True)
    mean013 = data.mean(axis=(0, 1, 3), keepdims=True)
    mean023 = data.mean(axis=(0, 2, 3), keepdims=True)
    mean123 = data.mean(axis=(1, 2, 3), keepdims=True)
    mean012 = data.mean(axis=(0, 1, 2), keepdims=True)

    ss_total = ((data - grand_mean)**2).sum()
    dof_total = n - 1

    ss_error = ((data - mean3)**2).sum()
    dof_error = n - a*b*c
    # XXX check this DOF calculation...
    ms_error = ss_error / dof_error

    ssa = r*b*c*((mean123 - grand_mean)**2).sum()
    dof_a = a - 1
    msa = ssa / dof_a
    F_a = msa / ms_error
    p_a = special.fdtrc(dof_a, dof_error, F_a)

    ssb = r*a*c*((mean023 - grand_mean)**2).sum()
    dof_b = b - 1
    msb = ssb / dof_b
    F_b = msb / ms_error
    p_b = special.fdtrc(dof_b, dof_error, F_b)

    ssc = r*a*b*((mean013 - grand_mean)**2).sum()
    dof_c = c - 1
    msc = ssc / dof_c
    F_c = msc / ms_error
    p_c = special.fdtrc(dof_c, dof_error, F_c)

    ssab = r*c*((mean23 - mean123 - mean023 + grand_mean)**2).sum()
    dof_ab = (a - 1)*(b - 1)
    msab = ssab / dof_ab
    F_ab = msab / ms_error
    p_ab = special.fdtrc(dof_ab, dof_error, F_ab)

    ssac = r*b*((mean13 - mean123 - mean013 + grand_mean)**2).sum()
    dof_ac = (a - 1)*(c - 1)
    msac = ssac / dof_ac
    F_ac = msac / ms_error
    p_ac = special.fdtrc(dof_ac, dof_error, F_ac)

    ssbc = r*a*((mean03 - mean023 - mean013 + grand_mean)**2).sum()
    dof_bc = (a - 1)*(b - 1)
    msbc = ssbc / dof_bc
    F_bc = msbc / ms_error
    p_bc = special.fdtrc(dof_bc, dof_error, F_bc)

    ssabc = r*((mean3 - mean23 - mean13 - mean03 + mean123 + mean023 + mean013 - grand_mean)**2).sum()
    dof_abc = (a - 1)*(b - 1)*(c - 1)
    msabc = ssabc / dof_abc
    F_abc = msabc / ms_error
    p_abc = special.fdtrc(dof_abc, dof_error, F_abc)


    print("                SS  DF         MS            F          p")
    print(f"a:      {ssa:10.5f} {dof_a:3d} {msa:10.5f} {F_a:12.6f} {p_a:10.6f}")
    print(f"b:      {ssb:10.5f} {dof_b:3d} {msb:10.5f} {F_b:12.6f} {p_b:10.6f}")
    print(f"c:      {ssc:10.5f} {dof_c:3d} {msc:10.5f} {F_c:12.6f} {p_c:10.6f}")
    print(f"a*b:    {ssab:10.5f} {dof_ab:3d} {msab:10.5f} {F_ab:12.6f} {p_ab:10.6f}")
    print(f"a*c:    {ssac:10.5f} {dof_ac:3d} {msac:10.5f} {F_ac:12.6f} {p_ac:10.6f}")
    print(f"b*c:    {ssbc:10.5f} {dof_bc:3d} {msbc:10.5f} {F_bc:12.6f} {p_bc:10.6f}")
    print(f"a*b*c:  {ssabc:10.5f} {dof_abc:3d} {msabc:10.5f} {F_abc:12.6f} {p_abc:10.6f}")
    print(f"error:  {ss_error:10.5f}  {dof_error:2} {ms_error:10.5f}")
    print(f"total:  {ss_total:10.5f}  {dof_total:2}")

    """
    ss_repl  = shp[0]*shp[1]*((mean01 - grand_mean)**2).sum()
    dof_repl = shp[2] - 1
    ms_repl  = ss_repl / dof_repl

    ss_02    = shp[0]*shp[2]*((mean02 - grand_mean)**2).sum()
    dof_02   = shp[1] - 1
    ms_02    = ss_02 / dof_02

    ss_12    = shp[1]*shp[2]*((mean12 - grand_mean)**2).sum()
    dof_12   = shp[0] - 1
    ms_12    = ss_12 / dof_12

    ss_inter  = shp[2]*((mean2 - mean12 - mean02 + grand_mean)**2).sum()
    dof_inter = (shp[0] - 1)*(shp[1] - 1)
    ms_inter  = ss_inter / dof_inter

    # These are from R. Johnson "Miller & Freund's Prob. & Stats for Engineers"
    #ss_error  = ((data - mean2 - mean01 + grand_mean)**2).sum()
    #dof_error = (shp[0]*shp[1] - 1)*(shp[2] - 1)
    # These are from Zar (fifth ed.)
    ss_error  = ((data - mean2)**2).sum()
    dof_error = (shp[0]*shp[1])*(shp[2] - 1)
    ms_error  = ss_error / dof_error

    F_repl  = ms_repl / ms_error
    F_02    = ms_02 / ms_error
    F_12    = ms_12 / ms_error
    F_inter = ms_inter / ms_error

    p_repl  = special.fdtrc(dof_repl, dof_error, F_repl)
    p_12    = special.fdtrc(dof_12, dof_error, F_12)
    p_02    = special.fdtrc(dof_02, dof_error, F_02)
    p_inter = special.fdtrc(dof_inter, dof_error, F_inter) 

    print("                       SS  DF          MS        F       p")
    #print(f"Replicates   {ss_repl:12.5f} {dof_repl:3}  {ms_repl:10.3f} {F_repl:8.3f} {p_repl:<10.5g}")
    print(f"Factor 0     {ss_12:12.5f} {dof_12:3}  {ms_12:10.3f} {F_12:8.3f} {p_12:<10.5g}")
    print(f"Factor 1     {ss_02:12.5f} {dof_02:3}  {ms_02:10.3f} {F_02:8.3f} {p_02:<10.5g}")
    print(f"Interaction  {ss_inter:12.5f} {dof_inter:3}  {ms_inter:10.3f} {F_inter:8.3f} {p_inter:<10.5g}")
    print(f"Error        {ss_error:12.5f} {dof_error:3}  {ms_error:10.3f}")
    print(f"Total        {ss_total:12.5f} {dof_total:3}")
    """

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


def _sum_of_squares(a, axis=0):
    """
    Square each element of the input array, and return the sum(s) of that.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    sum_of_squares : ndarray
        The sum along the given axis for (a**2).

    See also
    --------
    _square_of_sums : The square(s) of the sum(s) (the opposite of
    `_sum_of_squares`).
    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)


def _square_of_sums(a, axis=0):
    """
    Sum elements of the input array, and return the square(s) of that sum.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    square_of_sums : float or ndarray
        The square of the sum over `axis`.

    See also
    --------
    _sum_of_squares : The sum of squares (the opposite of `square_of_sums`).
    """
    a, axis = _chk_asarray(a, axis)
    s = np.sum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s


# f_oneway copied from scipy.

F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))


def f_oneway(*args):
    """
    Performs a 1-way ANOVA.

    The one-way ANOVA tests the null hypothesis that two or more groups have
    the same population mean.  The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group.

    Returns
    -------
    statistic : float
        The computed F-value of the test.
    pvalue : float
        The associated p-value from the F-distribution.

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.

    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. The population standard deviations of the groups are all equal.  This
       property is known as homoscedasticity.

    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`) although
    with some loss of power.

    The algorithm is from Heiman[2], pp.394-7.


    References
    ----------
    .. [1] R. Lowry, "Concepts and Applications of Inferential Statistics",
           Chapter 14, 2014, http://vassarstats.net/textbook/

    .. [2] G.W. Heiman, "Understanding research methods and statistics: An
           integrated introduction for psychology", Houghton, Mifflin and
           Company, 2001.

    .. [3] G.H. McDonald, "Handbook of Biological Statistics", One-way ANOVA.
           http://www.biostathandbook.com/onewayanova.html

    Examples
    --------
    >>> import scipy.stats as stats

    [3]_ Here are some data on a shell measurement (the length of the anterior
    adductor muscle scar, standardized by dividing by length) in the mussel
    Mytilus trossulus from five locations: Tillamook, Oregon; Newport, Oregon;
    Petersburg, Alaska; Magadan, Russia; and Tvarminne, Finland, taken from a
    much larger data set used in McDonald et al. (1991).

    >>> tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,
    ...              0.0659, 0.0923, 0.0836]
    >>> newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835,
    ...            0.0725]
    >>> petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
    >>> magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764,
    ...            0.0689]
    >>> tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
    >>> stats.f_oneway(tillamook, newport, petersburg, magadan, tvarminne)
    (7.1210194716424473, 0.00028122423145345439)

    """
    args = [np.asarray(arg, dtype=float) for arg in args]
    # ANOVA on N groups, each in its own array
    num_groups = len(args)
    alldata = np.concatenate(args)
    bign = len(alldata)
    print("bign =", bign)
    print("alldata.mean() =", alldata.mean())

    # Determine the mean of the data, and subtract that from all inputs to a
    # variance (via sum_of_sq / sq_of_sum) calculation.  Variance is invariance
    # to a shift in location, and centering all data around zero vastly
    # improves numerical stability.
    offset = alldata.mean()
    alldata -= offset

    sstot = _sum_of_squares(alldata) - (_square_of_sums(alldata) / bign)
    ssbn = 0
    for a in args:
        ssbn += _square_of_sums(a - offset) / len(a)

    # Naming: variables ending in bn/b are for "between treatments", wn/w are
    # for "within treatments"
    ssbn -= _square_of_sums(alldata) / bign
    print("ssbn =", ssbn)
    sswn = sstot - ssbn
    print("sswn =", sswn)
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    msb = ssbn / dfbn
    msw = sswn / dfwn
    f = msb / msw

    prob = special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf

    return F_onewayResult(f, prob)
