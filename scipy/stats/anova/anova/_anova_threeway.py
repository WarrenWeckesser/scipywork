import numpy as np
from scipy import special


def anova_threeway_balanced(data):
    """
    Three-way ANOVA for balanced inputs.

    WORK IN PROGRESS -- currently just prints results.

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
    #mean01 = data.mean(axis=(0,1), keepdims=True)
    #mean02 = data.mean(axis=(0,2), keepdims=True)
    mean03 = data.mean(axis=(0,3), keepdims=True)
    #mean12 = data.mean(axis=(1,2), keepdims=True)
    mean13 = data.mean(axis=(1,3), keepdims=True)
    mean23 = data.mean(axis=(2,3), keepdims=True)
    mean013 = data.mean(axis=(0, 1, 3), keepdims=True)
    mean023 = data.mean(axis=(0, 2, 3), keepdims=True)
    mean123 = data.mean(axis=(1, 2, 3), keepdims=True)
    #mean012 = data.mean(axis=(0, 1, 2), keepdims=True)

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
