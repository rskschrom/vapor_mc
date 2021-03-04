import single_particle as sp
import numpy as np
import micro as mi
import orient_dist as ods
import sys

# regression functions
def lpos(x, a):
    return a*x
    
def logr_plot(x, coeffs):
    return coeffs[0]*np.log(x)+coeffs[1]
    
def plaw_plot(x, coeffs):
    return coeffs[0]*x**coeffs[1]

# radar variable functions
def scat_fit(mass, rho, shpow0_coef, svpow0_coef, ph2_coef, pv2_coef, kdp2_coef):
    # zeroth-order coefficients
    shpow0 = plaw_plot(1.e6*mass, shpow0_coef)
    svpow0 = plaw_plot(1.e6*mass, svpow0_coef)

    # second-order coefficients
    pch2 = logr_plot(rho, ph2_coef)
    pcv2 = logr_plot(rho, pv2_coef)
    kdp2 = lpos(1.e6*mass, kdp2_coef)
    shpow2 = shpow0*(pch2/100.-1.)
    svpow2 = svpow0*(pcv2/100.-1.)

    # create legendre coefficient arrays
    npar = mass.shape[0]
    legc_shpow = np.zeros([npar,7])
    legc_svpow = np.zeros([npar,7])
    legc_kdp = np.zeros([npar,7])

    legc_shpow[:,0] = shpow0
    legc_shpow[:,2] = shpow2
    legc_svpow[:,0] = svpow0
    legc_svpow[:,2] = svpow2
    legc_kdp[:,2] = kdp2

    return legc_shpow, legc_svpow, legc_kdp
    
# precalculate aggregate scattering coefficients
def agg_coeffs(fr_name, b):
    lmax = 6
    msum = np.empty([lmax+1])
    for l in range(lmax+1):
        msum[l] = ods.beta_legendre(1., b, l)

    # scattering fits for aggregates
    exp_name = 'euler_new_nopiv'
    shpow0_coef = np.genfromtxt(f'coeffs/sigh0_coef_{exp_name}_{fr_name}.txt')
    svpow0_coef = np.genfromtxt(f'coeffs/sigv0_coef_{exp_name}_{fr_name}.txt')
    ph2_coef = np.genfromtxt(f'coeffs/ph2_coef_{exp_name}_{fr_name}.txt')
    pv2_coef = np.genfromtxt(f'coeffs/pv2_coef_{exp_name}_{fr_name}.txt')
    kdp2_coef = np.genfromtxt(f'coeffs/k2_coef_{exp_name}_{fr_name}.txt')
    coeffs = [shpow0_coef, svpow0_coef, ph2_coef, pv2_coef, kdp2_coef]
    return coeffs, msum
    
# orientation averaging in theta
def oravg(coeffs, b, msum):    
    # get coefficients
    cleg_shh2 = coeffs[0]
    cleg_svv2 = coeffs[1]
    cleg_kdp = coeffs[2]

    lmax = cleg_shh2.shape[1]-1
    lvals = np.arange(lmax+1)
    msum.shape = (1,lmax+1)
    lvals.shape = (1,lmax+1)
    #print(msum.shape, cleg_shh2.shape, lvals.shape)
    
    shh2_oavg = np.sum(msum*cleg_shh2*np.sqrt(2.*lvals+1), axis=1)
    svv2_oavg = np.sum(msum*cleg_svv2*np.sqrt(2.*lvals+1), axis=1)
    kdp_oavg = np.sum(msum*cleg_kdp*np.sqrt(2.*lvals+1), axis=1)

    return shh2_oavg, svv2_oavg, kdp_oavg


# compute aggregate scattering properties
def radar_agg(eps, wavl, mass_agg, rho_agg, vt_agg, coeffs, msum, bval):
    shpow0_coef = coeffs[0]
    svpow0_coef = coeffs[1]
    ph2_coef = coeffs[2]
    pv2_coef = coeffs[3]
    kdp2_coef = coeffs[4]
    legc_shpow, legc_svpow, legc_kdp = scat_fit(mass_agg, rho_agg, shpow0_coef, svpow0_coef,
                                                ph2_coef, pv2_coef, kdp2_coef)

    # integrate orientation distribution with legendre series
    coeffs = [legc_shpow, legc_svpow, legc_kdp]
    shh2, svv2, kdp = oravg(coeffs, bval, msum)
    sigh_reg = 4.*np.pi*shh2
    sigv_reg = 4.*np.pi*svv2
    kdp_reg = 180.e-3/np.pi*wavl*kdp
    return sigh_reg, sigv_reg, kdp_reg

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
    alp_a[sph_ind], alp_b[sph_ind], alp_c[sph_ind] = sp.sphere_polz(eps_snow[sph_ind], aval[sph_ind])

    # prolate particles
    sph_ind = (cval/aval>1.)
    alp_a[sph_ind], alp_b[sph_ind], alp_c[sph_ind] = sp.prolate_polz(eps_snow[sph_ind],
                                                                     cval[sph_ind], aval[sph_ind])

    if np.isnan(np.abs(alp_a)).any():
        print(aval[np.isnan(np.abs(alp_a))],
              cval[np.isnan(np.abs(alp_a))],
              eps_snow[np.isnan(np.abs(alp_a))])
        sys.exit()

    return alp_a, alp_b, alp_c


# single-scattering radar properties for branched planar crystals
def radar_bpc(a, c, rhoe, wavl, diel, bval, pdc, phc):
    # calculate polarizabilities using rayleigh theory
    alp_a, alp_b, alp_c = polz_bpc(diel, pdc, phc, a, c, rhoe)
    npar = len(a)

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
    
# simulate radar profile with aggregates
def radar_column(pr_props, agg_props, wavl, diel, bval, zedges, nscale, pdc, phc, acoeffs, msum, abval):
    # individual particle scattering properties for pristine
    vola = pr_props[0]
    volc = pr_props[1]
    zpar = pr_props[2]
    vt = pr_props[3]
    
    # calculate density for pristine
    phi = volc/vola
    a = 1.e3*(3.*vola/(4.*np.pi*phi))**(1./3.)
    c = a*phi
    rhoe = mi.rhoeff(a, 0.1, 0.4, 0.6, 0.7, 920.)
    
    sig_hh, sig_vv, kdp_par, cov_hhvv = radar_bpc(a, c, rhoe, wavl, diel, bval, pdc, phc)
    shh2 = sig_hh/(4.*np.pi)
    svv2 = sig_vv/(4.*np.pi)

    # individual particle scattering properties for aggregates
    mass_agg = agg_props[0]
    zagg = agg_props[1]
    vt_agg = agg_props[2]
    rho_agg = np.full([len(mass_agg)], agg_props[3])
    phi_agg = np.full([len(mass_agg)], 1.)
    a_agg = 1.e3*(3.*mass_agg/(4.*np.pi*rho_agg))**(1./3.)
    c_agg = a_agg*phi_agg
    #sig_hh_agg, sig_vv_agg, kdp_agg, cov_hhvv_agg = radar_bpc(a_agg, c_agg, rho_agg,
    #                                                           wavl, diel, bval, pdc, phc)
    sig_hh_agg, sig_vv_agg, kdp_agg = radar_agg(diel, wavl, mass_agg, rho_agg,
                                                vt_agg, acoeffs, msum, bval)
    shh2_agg = sig_hh_agg/(4.*np.pi)
    svv2_agg = sig_vv_agg/(4.*np.pi)
    cov_hhvv_agg = np.sqrt(shh2_agg*svv2_agg)

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
    bw = 2.*dz

    for i in range(nbin):
        # perfect binning weigting functions
        #wgt = np.piecewise(zpar, [zpar<zedges[i],(zpar>=zedges[i])&(zpar<=zedges[i+1]),
        #                   zpar>zedges[i+1]],
        #                   [0.,1.,0.])*nscale/dz
        #wgt_agg = np.piecewise(zagg, [zagg<zedges[i],(zagg>=zedges[i])&(zagg<=zedges[i+1]),
        #                       zagg>zedges[i+1]],
        #                       [0.,1.,0.])*nscale/dz
        zcen = 0.5*(zedges[i]+zedges[i+1])
        wgt = np.exp(-8.*np.log(2.)*(zpar-zcen)**2./bw**2.)/(bw*np.sqrt(np.pi/(8.*np.log(2.))))*nscale
        wgt_agg = np.exp(-8.*np.log(2.)*(zagg-zcen)**2./bw**2.)/(bw*np.sqrt(np.pi/(8.*np.log(2.))))*nscale
        
        # radar observables
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

