import warnings
from types import SimpleNamespace
import numpy as np
from scipy import special
from ._util import _nway_groups


__all__ = [
    'AnovaTwowayResult',
    'anova_twoway1', 'anova_twoway_balanced',
    'anova_twoway_from_x_y_values'
]


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
        raise ValueError("ndim = 2 not implemented yet, use anova_twoway1 "
                         "instead.")

    grand_mean = data.mean()

    mean2 = data.mean(axis=2, keepdims=True)

    meanB = data.mean(axis=(0, 2), keepdims=True)
    meanA = data.mean(axis=(1, 2), keepdims=True)

    ssB = shp[0]*shp[2]*((meanB - grand_mean)**2).sum()
    dofB = shp[1] - 1
    msB = ssB / dofB

    ssA = shp[1]*shp[2]*((meanA - grand_mean)**2).sum()
    dofA = shp[0] - 1
    msA = ssA / dofA

    ss_inter = shp[2]*((mean2 - meanA - meanB + grand_mean)**2).sum()
    dof_inter = (shp[0] - 1)*(shp[1] - 1)
    ms_inter = ss_inter / dof_inter

    # These are from R. Johnson "Miller & Freund's Prob. & Stats for Engineers"
    # ss_error  = ((data - mean2 - mean01 + grand_mean)**2).sum()
    # dof_error = (shp[0]*shp[1] - 1)*(shp[2] - 1)
    #
    # These are from Zar (fifth ed.)
    ss_error = ((data - mean2)**2).sum()
    dof_error = (shp[0]*shp[1])*(shp[2] - 1)
    ms_error = ss_error / dof_error

    FB = msB / ms_error
    FA = msA / ms_error
    F_inter = ms_inter / ms_error

    pA = special.fdtrc(dofA, dof_error, FA)
    pB = special.fdtrc(dofB, dof_error, FB)
    p_inter = special.fdtrc(dof_inter, dof_error, F_inter)

    result = AnovaTwowayResult(SSB=ssB, SSA=ssA,
                               SS01=ss_inter, SSerror=ss_error,
                               DFB=dofB, DFA=dofA,
                               DF01=dof_inter, DFerror=dof_error,
                               MSB=msB, MSA=msA,
                               MS01=ms_inter, MSerror=ms_error,
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
        raise ValueError("This function is for two-way ANOVA with no "
                         "replication.")
    r, c = shp

    # Work in progress...

    grand_mean = data.mean()
    mean0 = data.mean(axis=0, keepdims=True)
    mean1 = data.mean(axis=1, keepdims=True)

    ss_total = ((data - grand_mean)**2).sum()

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

    result = AnovaTwowayResult(SSB=ss0, SSA=ss1, SSerror=sse,
                               DFB=df0, DFA=df1, DFerror=dfe,
                               MSB=ms0, MSA=ms1, MSerror=mse,
                               FB=F0, FA=F1,
                               pB=p0, pA=p1)
    return result


def anova_twoway_unbalanced(data):
    """
    Two-way anova with unbalanced replication.

    WORK IN PROGRESS -- DO NOT USE!

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

    # v = sum(sum(sum((x - grand_mean)**2
    #                 for x in cell)
    #             for cell in row)
    #         for row in data)
    dfv = celln.sum() - 1

    vr = sum(sum(sum((rmean - grand_mean)**2
                     for x in cell)
                 for cell in row)
             for rmean, row in zip(rowmeans[:, 0], data))
    dfvr = nrows - 1

    vc = sum(sum(sum((cmean - grand_mean)**2
                     for x in cell)
                 for cmean, cell in zip(colmeans[0], row))
             for row in data)
    dfvc = ncols - 1

    vi = 0.0
    for i in range(nrows):
        for j in range(ncols):
            vi += celln[i, j]*(cellmeans[i, j] - rowmeans[i, 0]
                               - colmeans[0, j] + grand_mean)**2
    dfvi = (nrows - 1)*(ncols - 1)

    ve = sum(sum(sum((x - cellmean)**2
                     for x in cell)
                 for cellmean, cell in zip(cmean, row))
             for cmean, row in zip(cellmeans, data))
    dfve = dfv - dfvc - dfvr - dfvi

    msr = vr/dfvr
    msc = vc/dfvc
    msi = vi/dfvi
    mse = ve/dfve
    fr = msr/mse
    fc = msc/mse
    fi = msi/mse
    pr = special.fdtrc(dfvr, dfve, fr)
    pc = special.fdtrc(dfvc, dfve, fc)
    pi = special.fdtrc(dfvi, dfve, fi)

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
    groups1d = groups.ravel()
    if any(g is None for g in groups1d):
        raise ValueError('There is an input combination that has no data.')
    lengths = [len(t) for t in groups1d]
    if not all(n == lengths[0] for n in lengths):
        warnings.warn("dataset is unbalanced; check result carefully",
                      RuntimeWarning)
        result = anova_twoway_unbalanced(groups)
    else:
        result = anova_twoway_balanced(groups)
    return result


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
