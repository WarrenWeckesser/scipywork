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
        raise ValueError("ndim = 2 not implemented yet, use anova_twoway1 instead.")

    grand_mean = data.mean()

    #mean0 = data.mean(axis=(1,2), keepdims=True)
    mean2 = data.mean(axis=2, keepdims=True)
    #mean01 = data.mean(axis=(0,1), keepdims=True)

    #mean02 = data.mean(axis=(0,2), keepdims=True)
    meanB = data.mean(axis=(0,2), keepdims=True)
    #mean12 = data.mean(axis=(1,2), keepdims=True)
    meanA = data.mean(axis=(1,2), keepdims=True)

    #ss_total = ((data - grand_mean)**2).sum()
    #dof_total = shp[0]*shp[1]*shp[2] - 1

    #ss_repl  = shp[0]*shp[1]*((mean01 - grand_mean)**2).sum()
    #dof_repl = shp[2] - 1
    #ms_repl  = ss_repl / dof_repl

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

    #F_repl  = ms_repl / ms_error
    FB      = msB / ms_error
    FA      = msA / ms_error
    F_inter = ms_inter / ms_error

    #p_repl  = special.fdtrc(dof_repl, dof_error, F_repl)
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
    #dof_total = shp[0]*shp[1] - 1

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

    #v = sum(sum(sum((x - grand_mean)**2
    #                for x in cell)
    #            for cell in row)
    #        for row in data)
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

    #ms = v/dfv
    msr = vr/dfvr
    msc = vc/dfvc
    msi = vi/dfvi
    mse = ve/dfve
    #ft = ms/mse
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
