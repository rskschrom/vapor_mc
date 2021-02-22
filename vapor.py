import numpy as np
import matplotlib.pyplot as plt
import thermo as th
import micro as mi
import forward as fo

# volume change
def dvoldt(vol, phi, tmpk, si, pres):
    a = (3.*vol/(4.*np.pi*phi))**(1./3.)
    c = a*phi
    rad = (3.*vol/(4.*np.pi))**(1./3.)

    # particle effective density
    rhod = mi.rhodep(a, 0.1, 0.4, 0.6, 0.7, 920.)
    rhod[a<=c] = 920.

    vol_ch = 4.*np.pi*mi.capac(a,c)*si*th.g_diff(tmpk, pres, si, rad)/rhod
    return vol_ch, vol_ch*rhod

# aspect ratio change
def dphidt(vol, phi):
    gam = 0.28
    dphidt = phi/vol*(gam-1.)/(gam+2.)
    return dphidt

# a- and c-volume changes
def dvolacdt(vola, volc, tmpk, si, pres, gam):
    #vola = vola_max[~vola_max.mask]
    #volc = volc_max[~volc_max.mask]
    vola_ch, mass_ch = dvoldt(vola, volc/vola, tmpk, si, pres)
    volc_ch = volc/vola*vola_ch*(2.*gam+1.)/(gam+2.)
    return vola_ch, volc_ch, mass_ch

# define model structure
def model(sigx, sigy, corr, gam, bval, tmpk, si, pres, nt, npar):
    # create initial distribution of particles
    res = 10.**np.random.multivariate_normal([-13.,-13.], [[sigx**2., corr*sigx*sigy],
                                                           [corr*sigx*sigy, sigy**2.]], size=(npar))
    vola = res[:,0]
    volc = res[:,1]

    # radar parameters
    wavl = 32.1
    diel = 3.17+1j*0.009

    # integrate model in time
    dt = 1.
    for i in range(nt):
        dvadt, dvcdt, dmdt = dvolacdt(vola, volc, tmpk, si, pres, gam)
        vola = vola+dt*dvadt
        volc = volc+dt*dvcdt

    # simulate radar variables
    zh, zdr, kdp, rhohv = fo.radar(vola, volc, wavl, diel, bval)
    return zh, zdr, kdp, rhohv

