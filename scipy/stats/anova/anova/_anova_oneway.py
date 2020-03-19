from types import SimpleNamespace
import numpy as np
from scipy import special


__all__ = [
    'AnovaOnewayResult',
    'anova_oneway', 'anova_oneway_from_xy',
    'anova_oneway_ci_diffs', 'anova_oneway_ci_means',
]


def _fmt(value):
    return np.array2string(np.array(value))


class AnovaOnewayResult(SimpleNamespace):

    def __str__(self):
        vartot = self.SSb + self.SSw
        dftot = self.DFb + self.DFw
        mstot = vartot / dftot
        # Format all the values as strings, so we can align them nicely
        # in the output.
        SSb = _fmt(self.SSb)
        SSw = _fmt(self.SSw)
        vartot = _fmt(vartot)
        SSwidth = max(len(SSb), len(SSw), len(vartot))
        MSb = _fmt(self.MSb)
        MSw = _fmt(self.MSw)
        mstot = _fmt(mstot)
        MSwidth = max(len(MSb), len(MSw), len(mstot))
        DFb = str(self.DFb)
        DFw = str(self.DFw)
        dftot = str(dftot)
        DFwidth = max(2, len(DFb), len(DFw), len(dftot))
        F = _fmt(self.F)
        Fwidth = len(F)
        p = format(self.p, '.5g')
        pwidth = len(p)

        s = "\n".join([
                "ANOVA one-way",
                (f"Source         {'SS':>{SSwidth}}  {'DF':>{DFwidth}}  "
                 f"{'MS':>{MSwidth}}  {'F':>{Fwidth}}  {'p':>{pwidth}}"),
                (f"Between groups {SSb:{SSwidth}}  {DFb:>{DFwidth}}  "
                 f"{MSb:{MSwidth}}  {F}  {p}"),
                (f"Within groups  {SSw:{SSwidth}}  {DFw:>{DFwidth}}  "
                 f"{MSw:{MSwidth}}"),
                (f"Total          {vartot:{SSwidth}}  {dftot:>{DFwidth}}  "
                 f"{mstot:{MSwidth}}")])
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

    # Check for the edge case where the values in each group are constant.
    # When this happens, vw is 0.  If we don't handle this explicitly, vw
    # might contain numerical noise, and then F will be nonsense.
    if all([np.all(group[0] == group) for group in groups]):
        vw = 0.0
    else:
        vw = v - vb

    dfb = num_groups - 1
    dfw = n - num_groups
    msb = vb / dfb
    msw = vw / dfw
    if msw > 0:
        F = msb / msw
    else:
        F = np.inf
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
            c = np.sqrt(result.MSw * (1/result.group_sizes[i]
                                      + 1/result.group_sizes[j]))
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
