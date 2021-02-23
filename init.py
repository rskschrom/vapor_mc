import numpy as np
import thermo as th

# get interpolated profiles from model sounding
def interp_sounding(fsnd, zedges):
    # read file
    data = np.genfromtxt(fsnd, skip_header=1)
    psnd = data[:,0]*100.
    tsnd = data[:,1]+273.15
    tdsnd = data[:,2]+273.15
    zsnd = data[:,5]

    # interpolate sounding data to center height bins
    zcen = 0.5*(zedges[:-1]+zedges[1:])
    pint = np.interp(zcen, zsnd, psnd)
    tint = np.interp(zcen, zsnd, tsnd)
    tdint = np.interp(zcen, zsnd, tdsnd)

    si = th.svp_liq(tdint)/th.svp_ice(tint)-1.

    return pint, tint, si
