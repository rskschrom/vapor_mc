import numpy as np
import matplotlib.pyplot as plt
import thermo as th
import single_particle as sp

# capacitance
def capac(a, c):
    ecc = np.empty(a.shape)
    capac = np.empty(a.shape)
    ecc[a>c] = np.sqrt(1.-(c[a>c]/a[a>c])**2.)
    ecc[a<c] = np.sqrt(1.-(a[a<c]/c[a<c])**2.)
    capac[a>c] = a[a>c]*ecc[a>c]/np.arcsin(ecc[a>c])
    capac[a<c] = c[a<c]*ecc[a<c]/np.log(c[a<c]/a[a<c]*(1.+ecc[a<c]))
    capac[a==c] = a[a==c]
    return capac

# volume change
def dvoldt(vol, phi, tmpk, si, pres):
    rhoi = 920.
    a = (3.*vol/(4.*np.pi*phi))**(1./3.)
    c = a*phi
    rad = (3.*vol/(4.*np.pi))**(1./3.)
    vol_ch = 4.*np.pi*capac(a,c)*si*th.g_diff(tmpk, pres, si, rad)/rhoi
    return vol_ch

# aspect ratio change
def dphidt(vol, phi):
    gam = 0.28
    dphidt = phi/vol*(gam-1.)/(gam+2.)
    return dphidt

# a- and c-volume changes
def dvolacdt(vola, volc, tmpk, si, pres, gam):
    vola_ch = dvoldt(vola, volc/vola, tmpk, si, pres)
    volc_ch = volc/vola*vola_ch*(2.*gam+1.)/(gam+2.)
    return vola_ch, volc_ch

# radar forward model
def radar(vola, volc, wavl, diel, bval):
    # calculate particle properties
    phi = volc/vola
    a = 1.e3*(3.*vola/(4.*np.pi*phi))**(1./3.)
    c = a*phi

    # calculate polarizabilities using rayleigh theory
    npar = len(a)
    alp_a = np.empty([npar], dtype=complex)
    alp_b = np.empty([npar], dtype=complex)
    alp_c = np.empty([npar], dtype=complex)
    res = sp.oblate_polz(diel, a[a>c], c[a>c])
    alp_a[a>c] = res[0]
    alp_b[a>c] = res[1]
    alp_c[a>c] = res[2] 
    res = sp.prolate_polz(diel, c[a<c], a[a<c])
    alp_a[a<c] = res[0]
    alp_b[a<c] = res[1]
    alp_c[a<c] = res[2]

    # randomly sample orientations
    alpha = 2.*np.pi*np.random.rand(npar)
    beta = np.arccos(1.-2.*np.random.beta(1., bval, size=npar))
    gamma = 2.*np.pi*np.random.rand(npar)
    arot = sp.euler(alpha, beta, gamma)
    atens = np.zeros([3,3,npar], dtype=complex)
    atens[0,0,:] = alp_a
    atens[1,1,:] = alp_b
    atens[2,2,:] = alp_c
    alpha_rot = np.einsum('jik,jmk->imk', arot, np.einsum('ijk,jmk->imk', atens, arot))

    # calculate radar variables
    k = 2.*np.pi/wavl
    shh = k**2./(4.*np.pi)*alpha_rot[1,1,:]
    svv = k**2./(4.*np.pi)*alpha_rot[2,2,:]
    sig_hh = 4.*np.pi*np.abs(shh)**2.
    sig_vv = 4.*np.pi*np.abs(svv)**2.
    kdp = 180.e-3/np.pi*wavl*np.real(shh-svv)

    zhh = wavl**4./(np.pi**5*0.93)*np.sum(sig_hh)
    zvv = wavl**4./(np.pi**5*0.93)*np.sum(sig_vv)
    zdr = zhh/zvv
    kdp = np.sum(kdp)
    rhohv = np.abs(np.sum(np.conj(alpha_rot[1,1,:])*alpha_rot[2,2,:]))
    rhohv = rhohv/np.sqrt(np.sum(np.abs(alpha_rot[1,1,:])**2.)*np.sum(np.abs(alpha_rot[2,2,:])**2.))
    return 10.*np.log10(zhh), 10.*np.log10(zdr), kdp, rhohv

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
        dvadt, dvcdt = dvolacdt(vola, volc, tmpk, si, pres, gam)
        vola = vola+dt*dvadt
        volc = volc+dt*dvcdt

    # simulate radar variables
    zh, zdr, kdp, rhohv = radar(vola, volc, wavl, diel, bval)
    return zh, zdr, kdp, rhohv

