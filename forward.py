import single_particle as sp
import numpy as np
import micro as mi
import sys

# branched planar crystal polarizabilities
def polz_bpc(diel, pd_coeffs, ph_coeffs, aval, cval, rho):
    # polarizability arrays
    alp_a = np.empty([len(aval)], dtype=complex)
    alp_b = np.empty([len(aval)], dtype=complex)
    alp_c = np.empty([len(aval)], dtype=complex)

    # equivalent spheroids for branched planar crystals
    lasp = -np.log10(cval/aval)
    pcd = pd_coeffs[0]+pd_coeffs[1]*rho+\
           pd_coeffs[2]*lasp+pd_coeffs[3]*rho**2.+\
           pd_coeffs[4]*lasp**2.
    pch = ph_coeffs[0]+ph_coeffs[1]*rho+\
           ph_coeffs[2]*lasp+ph_coeffs[3]*rho**2.+\
           ph_coeffs[4]*lasp**2.
    aeqv = aval*(1.+pcd/100.)
    ceqv = cval*(1.+pch/100.)
    sph_ind = (cval/aval<0.1)
    alp_a[sph_ind], alp_b[sph_ind], alp_c[sph_ind] = sp.oblate_polz(diel, aeqv[sph_ind], ceqv[sph_ind])

    # soft spheroids for everything else
    eps_snow = sp.maxwell_mixing(rho, diel)

    # oblate spheroids
    sph_ind = (cval/aval>=0.1)&(cval/aval<1.)
    alp_a[sph_ind], alp_b[sph_ind], alp_c[sph_ind] = sp.oblate_polz(eps_snow[sph_ind], aval[sph_ind], cval[sph_ind])

    # near-spherical particles
    sph_ind = (np.abs(cval/aval-1.)<1.e-16)
    alp_a[sph_ind], alp_b[sph_ind], alp_c[sph_ind] = sp.oblate_polz(eps_snow[sph_ind],
                                                                    aval[sph_ind],
                                                                    aval[sph_ind]*1.000001)

    # prolate particles
    sph_ind = (cval/aval>1.)
    alp_a[sph_ind], alp_b[sph_ind], alp_c[sph_ind] = sp.prolate_polz(eps_snow[sph_ind],
                                                                     cval[sph_ind], aval[sph_ind])

    if np.isnan(np.abs(alp_a)).any():
        print(aval[np.isnan(np.abs(alp_a))], cval[np.isnan(np.abs(alp_a))])
        sys.exit()

    return alp_a, alp_b, alp_c


# single-scattering radar properties
def radar_scat(vola, volc, wavl, diel, bval, pdc, phc):
    # calculate particle properties
    phi = volc/vola
    a = 1.e3*(3.*vola/(4.*np.pi*phi))**(1./3.)
    c = a*phi
    rhoe = mi.rhoeff(a, 0.1, 0.4, 0.6, 0.7, 920.)

    # calculate polarizabilities using rayleigh theory
    npar = len(a)
    alp_a, alp_b, alp_c = polz_bpc(diel, pdc, phc, a, c, rhoe)
    '''
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
    '''

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

    # calculate radar scattering properties
    k = 2.*np.pi/wavl
    shh = k**2./(4.*np.pi)*alpha_rot[1,1,:]
    svv = k**2./(4.*np.pi)*alpha_rot[2,2,:]
    cov_hhvv = np.conj(shh)*svv
    sig_hh = 4.*np.pi*np.abs(shh)**2.
    sig_vv = 4.*np.pi*np.abs(svv)**2.
    kdp = 180.e-3/np.pi*wavl*np.real(shh-svv)

    return sig_hh, sig_vv, kdp, cov_hhvv

# simulate radar profile using beam weighting function
def radar_column(vola, volc, zpar, vt, wavl, diel, bval, zedges, nscale, pdc, phc):
    # individual particle scattering properties
    sig_hh, sig_vv, kdp_par, cov_hhvv = radar_scat(vola, volc, wavl, diel, bval, pdc, phc)
    shh2 = sig_hh/(4.*np.pi)
    svv2 = sig_vv/(4.*np.pi)

    # calculate radar vertical bins
    zcen = 0.5*(zedges[1:]+zedges[:-1])
    nbin = len(zcen)

    # integrate radar variables at each bin
    zhh = np.empty([nbin])
    zvv = np.empty([nbin])
    kdp = np.empty([nbin])
    rhohv = np.empty([nbin])
    mdv = np.empty([nbin])

    dz = zedges[1]-zedges[0]

    for i in range(nbin):
        #wgt = np.exp(-4.*np.log(2.)*(zcen[i]-zpar)**2./(beam_width_z)**2.)
        #wgt = nscale*wgt/(beam_width_z**2.*np.pi/(16.*np.log(2.)))
        wgt = np.piecewise(zpar, [zpar<zedges[i],(zpar>=zedges[i])&(zpar<=zedges[i+1]),zpar>zedges[i+1]],
                              [0.,1.,0.])*nscale/dz
        #wgt = nscale*wgt/(beam_width_z**2.*np.pi/(16.*np.log(2.)))
        zhh[i] = wavl**4./(np.pi**5*0.93)*np.sum(wgt*sig_hh)
        zvv[i] = wavl**4./(np.pi**5*0.93)*np.sum(wgt*sig_vv)
        kdp[i] = np.sum(wgt*kdp_par)
        rhohv[i] = np.abs(np.sum(wgt*cov_hhvv))
        rhohv[i] = rhohv[i]/np.sqrt(np.sum(wgt*shh2)*np.sum(wgt*svv2))
        mdv[i] = -np.sum(wgt*vt*sig_hh)/np.sum(wgt*sig_hh)

    zh = 10.*np.log10(zhh)
    zdr = 10.*np.log10(zhh/zvv)

    return zh, zdr, kdp, rhohv, mdv
    
# simulate radar profile with aggregates
def radar_agg(pr_props, agg_props, wavl, diel, bval, zedges, nscale, pdc, phc):
    # individual particle scattering properties for pristine
    vola = pr_props[0]
    volc = pr_props[1]
    zpar = pr_props[2]
    vt = pr_props[3]
    sig_hh, sig_vv, kdp_par, cov_hhvv = radar_scat(vola, volc, wavl, diel, bval, pdc, phc)
    shh2 = sig_hh/(4.*np.pi)
    svv2 = sig_vv/(4.*np.pi)
    
    # individual particle scattering properties for aggregates
    mass_agg = agg_props[0]
    zagg = agg_props[1]
    vt_agg = agg_props[2]
    sig_hh_agg, sig_vv_agg, kdp_agg, cov_hhvv_agg = radar_scat(mass_agg/100., mass_agg/100., wavl, diel, bval, pdc, phc)
    shh2_agg = sig_hh_agg/(4.*np.pi)
    svv2_agg = sig_vv_agg/(4.*np.pi)

    # calculate radar vertical bins
    zcen = 0.5*(zedges[1:]+zedges[:-1])
    nbin = len(zcen)

    # integrate radar variables at each bin
    zhh = np.empty([nbin])
    zvv = np.empty([nbin])
    kdp = np.empty([nbin])
    rhohv = np.empty([nbin])
    mdv = np.empty([nbin])

    dz = zedges[1]-zedges[0]

    for i in range(nbin):
        #wgt = np.exp(-4.*np.log(2.)*(zcen[i]-zpar)**2./(beam_width_z)**2.)
        #wgt = nscale*wgt/(beam_width_z**2.*np.pi/(16.*np.log(2.)))
        wgt = np.piecewise(zpar, [zpar<zedges[i],(zpar>=zedges[i])&(zpar<=zedges[i+1]),
                           zpar>zedges[i+1]],
                           [0.,1.,0.])*nscale/dz
        wgt_agg = np.piecewise(zagg, [zagg<zedges[i],(zagg>=zedges[i])&(zagg<=zedges[i+1]),
                               zagg>zedges[i+1]],
                               [0.,1.,0.])*nscale/dz
        #wgt = nscale*wgt/(beam_width_z**2.*np.pi/(16.*np.log(2.)))
        zhh[i] = wavl**4./(np.pi**5*0.93)*(np.sum(wgt*sig_hh)+np.sum(wgt_agg*sig_hh_agg))
        zvv[i] = wavl**4./(np.pi**5*0.93)*(np.sum(wgt*sig_vv)+np.sum(wgt_agg*sig_vv_agg))
        kdp[i] = np.sum(wgt*kdp_par)+np.sum(wgt_agg*kdp_agg)
        rhohv[i] = np.abs(np.sum(wgt*cov_hhvv)+np.sum(wgt_agg*cov_hhvv_agg))
        rhohv[i] = rhohv[i]/np.sqrt((np.sum(wgt*shh2)+np.sum(wgt_agg*shh2_agg))*
                                    (np.sum(wgt*svv2)+np.sum(wgt_agg*svv2_agg)))
        mdv[i] = -(np.sum(wgt*vt*sig_hh)+np.sum(wgt_agg*vt_agg*sig_hh_agg))/\
                  (np.sum(wgt*sig_hh)+np.sum(wgt_agg*sig_hh_agg))

    zh = 10.*np.log10(zhh)
    zdr = 10.*np.log10(zhh/zvv)

    return zh, zdr, kdp, rhohv, mdv

# radar forward model for box model
def radar(vola, volc, wavl, diel, bval):
    sig_hh, sig_vv, kdp, cov_hhvv = radar_scat(vola, volc, wavl, diel, bval)
    shh2 = sig_hh/(4.*np.pi)
    svv2 = sig_vv/(4.*np.pi)

    zhh = wavl**4./(np.pi**5*0.93)*np.sum(sig_hh)
    zvv = wavl**4./(np.pi**5*0.93)*np.sum(sig_vv)
    zdr = zhh/zvv
    kdp = np.sum(kdp)
    rhohv = np.abs(np.sum(cov_hhvv))
    rhohv = rhohv/np.sqrt(np.sum(shh2)*np.sum(svv2))
    return 10.*np.log10(zhh), 10.*np.log10(zdr), kdp, rhohv

