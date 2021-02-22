import numpy as np
from scipy import integrate
from scipy.special import ellipkinc, ellipeinc

def maxwell_mixing(rho_snow, eps_ice):
    # estimate refractive index (Maxwell-Garnett)
    rho_ice = 920.
    eps_fac = (eps_ice-1.)/(eps_ice+2.)
    eps_snow = (1.+2.*rho_snow/rho_ice*eps_fac)/(1.-rho_snow/rho_ice*eps_fac)
    return eps_snow

# calculate ellipsoid shape factors with elliptical integrals
def ellipsoid_shape_func(a, b, c):
    phi = np.arccos(c/a)
    m = (a**2.-b**2.)/(a**2.-c**2.)
    la = a*b*c/((a**2.-b**2.)*np.sqrt(a**2.-c**2.))*(ellipkinc(phi,m)-ellipeinc(phi,m))
    lc = b/(b**2.-c**2.)*(b-a*c/np.sqrt(a**2.-c**2.)*ellipeinc(phi,m))
    lb = 1.-la-lc
    return la, lb, lc

# integrate ellipsoid shape factors numerically
def ellipsoid_shape_facs(a, b, c):
    f = lambda x,av,bv,cv,dv : 0.5*av*bv*cv/((x+dv**2.)*
                               np.sqrt((x+av**2.)*(x+bv**2.)*(x+cv**2.)))

    la, err = integrate.quad(f, 0., np.inf, args=(a,b,c,a))
    lb, err = integrate.quad(f, 0., np.inf, args=(a,b,c,b))
    lc, err = integrate.quad(f, 0., np.inf, args=(a,b,c,c))

    return la, lb, lc

# ellipsoid polarizabilities
def ellipsoid_polz(diel, a, b, c):
    la, lb, lc = ellipsoid_shape_func(a, b, c)
    alph_a = 4./3.*np.pi*a*b*c*(diel-1.)/(1.+la*(diel-1.))
    alph_b = 4./3.*np.pi*a*b*c*(diel-1.)/(1.+lb*(diel-1.))
    alph_c = 4./3.*np.pi*a*b*c*(diel-1.)/(1.+lc*(diel-1.))

    return alph_a, alph_b, alph_c

# oblate spheroid (a=b>c) polarizabilities
def oblate_polz(diel, a, c):
    f = np.sqrt((a/c)**2.-1.)
    lc = (1.+f**2.)/f**2.*(1.-np.arctan(f)/f)
    la = (1.-lc)/2.
    alph_a = 4./3.*np.pi*a**2.*c*(diel-1.)/(1.+la*(diel-1.))
    alph_c = 4./3.*np.pi*a**2.*c*(diel-1.)/(1.+lc*(diel-1.))

    return alph_a, alph_a, alph_c

# prolate spheroid (a>b=c) polarizabilities
def prolate_polz(diel, a, b):
    ec = np.sqrt(1.-(b/a)**2.)
    la = (1.-ec**2.)/ec**2.*(1./(2.*ec)*np.log((1.+ec)/(1.-ec))-1.)
    lb = (1.-la)/2.
    alph_a = 4./3.*np.pi*b**2.*a*(diel-1.)/(1.+la*(diel-1.))
    alph_b = 4./3.*np.pi*b**2.*a*(diel-1.)/(1.+lb*(diel-1.))

    return alph_a, alph_b, alph_b

# anisotropic oblate spheroid (a=b>c) polarizabilities
def ani_oblate_polz(diel_a, diel_c, a, c):
    f = np.sqrt((a/c)**2.-1.)
    lc = (1.+f**2.)/f**2.*(1.-np.arctan(f)/f)
    la = (1.-lc)/2.
    alph_a = 4./3.*np.pi*a**2.*c*(diel_a-1.)/(1.+la*(diel_a-1.))
    alph_c = 4./3.*np.pi*a**2.*c*(diel_c-1.)/(1.+lc*(diel_c-1.))

    return alph_a, alph_a, alph_c

# spheroid inclusions with beta orientation dist
def eps_spheroid_beta(eps_inc, vfrac, angmom, a, c):
    ma = angmom[0]
    mc = angmom[1]
    if a>c:
        f = np.sqrt((a/c)**2.-1.)
        lc = (1.+f**2.)/f**2.*(1.-np.arctan(f)/f)
    else:
        ec = np.sqrt(1.-(a/c)**2.)
        lc = (1.-ec**2.)/ec**2.*(1./(2.*ec)*np.log((1.+ec)/(1.-ec))-1.)

    la = (1.-lc)/2.
    lam_a = 1./(1.+la*(eps_inc-1.))
    lam_c = 1./(1.+lc*(eps_inc-1.))

    # calculate effective dielectric tensor
    eps_a = (1.-vfrac+vfrac*eps_inc*(lam_a+ma*(lam_a-lam_c)))/(1.-vfrac+vfrac*(lam_a+ma*(lam_a-lam_c)))
    eps_c = (1.-vfrac+vfrac*eps_inc*(lam_c+mc*(lam_a-lam_c)))/(1.-vfrac+vfrac*(lam_c+mc*(lam_a-lam_c)))
    return eps_a, eps_c

# get euler angle rotation matrices
def euler(alpha, beta, gamma):
    nang = len(alpha)
    c1 = np.cos(alpha)
    s1 = np.sin(alpha)
    c2 = np.cos(beta)
    s2 = np.sin(beta)
    c3 = np.cos(gamma)
    s3 = np.sin(gamma)

    euler_mat = np.empty([3,3,nang])
    euler_mat[0,0,:] = c1*c2*c3-s1*s3
    euler_mat[0,1,:] = -c3*s1-c1*c2*s3
    euler_mat[0,2,:] = c1*s2
    euler_mat[1,0,:] = c1*s3+c2*c3*s1
    euler_mat[1,1,:] = c1*c3-c2*s1*s3
    euler_mat[1,2,:] = s1*s2
    euler_mat[2,0,:] = -c3*s2
    euler_mat[2,1,:] = s2*s3
    euler_mat[2,2,:] = c2
    return euler_mat

# rotation of polarizability tensor via euler angles
def pol_euler(alpha, beta, gamma, alpha_a, alpha_b, alpha_c):
    # create rotation matrices
    a_alp = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                      [np.sin(alpha), np.cos(alpha), 0.],
                      [0., 0., 1.]])
    a_bet = np.array([[np.cos(beta), 0., np.sin(beta)],
                      [0., 1., 0.],
                      [-np.sin(beta), 0., np.cos(beta)]])
    a_gam = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                      [np.sin(gamma), np.cos(gamma), 0.],
                      [0., 0., 1.]])

    # transform polarizability tensor
    alpha_tensor = np.array([[alpha_a,0.,0.],
                             [0.,alpha_b,0.],
                             [0.,0.,alpha_c]])
    arot = a_alp @ (a_bet @ a_gam)
    alpha_rot = arot.T @ (alpha_tensor @ arot)
    return alpha_rot

# scattering matrix from incident and scattering direction
# th_i - incident elevation angle from x-y plane
# ph_i - incident azimuthal angle from x axis
# th_s - scattering elevation angle from x-y plane
# ph_s - scattering azimuthal angle from x axis
def scat_matrix(th_i, phi_i, th_s, phi_s, alpha_tensor, k):
    # calculate incident field directions
    sti = np.sin(th_i)
    cti = np.cos(th_i)
    spi = np.sin(phi_i)
    cpi = np.cos(phi_i)
    apol_i = np.array([[-spi, cpi, 0.],
                       [-cpi*sti, -spi*sti, cti]])

    # calculate scattered field directions
    sts = np.sin(th_s)
    cts = np.cos(th_s)
    sps = np.sin(phi_s)
    cps = np.cos(phi_s)
    apol_s = np.array([[-sps, cps, 0.],
                       [-cps*sts, -sps*sts, cts]])
    alpha_pol = np.matmul(apol_s, np.matmul(alpha_tensor, apol_i.T))

    smat = k**2./(4.*np.pi)*alpha_pol
    return smat
