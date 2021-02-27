import numpy as np

# inherent growth ratio (rough fit to Chen and Lamb [1994])
def igr(tmpk):
    tmpc = tmpk-273.15
    igr = np.piecewise(tmpc, [tmpc<-30., (tmpc>=-30.)&(tmpc<-23.),
                              (tmpc>=-23.)&(tmpc<-15.),
                              (tmpc>=-15.)&(tmpc<-7.),
                              (tmpc>=-7.)&(tmpc<-4.),
                              (tmpc>=-4.)&(tmpc<=0.), tmpc>0.],
                             [1.1, lambda tmpc: 1.1*(-23.-tmpc)/7.+2.*(tmpc+30.)/7.,
                              lambda tmpc: 2.*(-15.-tmpc)/8.+0.28*(tmpc+23.)/8.,
                              lambda tmpc: 0.28*(-7.-tmpc)/8.+2.1*(tmpc+15.)/8.,
                              lambda tmpc: 2.1*(-4.-tmpc)/3.+0.75*(tmpc+7.)/3.,
                              lambda tmpc: 0.75*(0.-tmpc)/4.+1.*(tmpc+4.)/4., 1.])
    #igr = np.piecewise(tmpc, [tmpc<-30.,(tmpc>=-30)&(tmpc<-20.),(tmpc>=-20)&(tmpc<=0.),tmpc>0.],
    #                         [1.1,0.8,0.5,1.])
    return igr

# fall speed
def fall_speed(vola, volc):
    rhoi = 920.
    area = np.empty(vola.shape)
    a = (3.*vola/(4.*np.pi*volc/vola))**(1./3.)
    c = a*volc/vola

    area[vola>=volc] = np.pi*a[vola>=volc]**2.
    area[vola<volc] = np.pi*a[vola<volc]*c[vola<volc]
    av = 0.6
    bv = 0.3
    #vt = av*(mass/(area*rhoe/920.))**bv
    vt = av*(920.*vola/area)**bv
    return vt

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

# deposition density function
def rhodep(a, ac, fb, ft, fi, rhoi):
    # set density parameters
    amax = 3.
    nsb = 5

    # calculate fmb and ai
    wt = ft/2.*amax
    wsb = 1./((nsb-1)/fb+1.)*(amax-ac)
    wmb = min(max(wsb/2., ac/2.), min(wsb/2., ac/2.))
    fmb = wmb/(ac/2.)
    ai = fi*amax+(1.-fi)*ac

    # calculate deposition density (scaled linear fraction x rhoi)
    rhod = rhoi*(fb/(amax-ai)*(ft*amax-ai+ai*amax*(1.-ft)/a)+(1.-fb)*fmb*ac/a)
    rhod[a<=ac] = rhoi
    rhod[(a>ac)&(a<=ai)] = rhoi*(fb+ac/a[(a>ac)&(a<=ai)]*fmb*(1.-fb))

    # hard limit on minimum rhodep
    rhod = np.maximum(rhod, 150.)
    return rhod

# get areas of each region of branched planar
def area_bound(a):
    area = np.sqrt(3.)/4.*a**2.
    return area

def area_gap(a, fmb, fb, ac):
    area = np.sqrt(3.)/2.*fmb*(1.-fb)*ac*(a-ac)+\
           np.sqrt(3.)/4.*fb*(a**2.-ac**2.)
    return area

def area_star(a, fmb, fb, ft, ac, ag, amax):
    b = fb/(amax-ag)*(ft*amax-ag)
    c = (fb*ag*amax*(1.-ft)/(amax-ag))+(1.-fb)*fmb*ac
    area = np.sqrt(3.)/4.*b*(a**2.-ag**2.)+np.sqrt(3.)/2.*c*(a-ag)
    return area

# effective density of branched planar
def rhoeff(a, ac, fb, ft, fi, rhoi):
    # calculate fmb and ai
    nsb = 5
    amax = 3.
    wt = ft/2.*amax
    wsb = 1./((nsb-1)/fb+1.)*(amax-ac)
    wmb = min(max(wsb/2., ac/2.), min(wsb/2., ac/2.))
    fmb = wmb/(ac/2.)
    ai = fi*amax+(1.-fi)*ac
    
    # calculate effective densities
    rhoeff = (area_bound(ac)+area_gap(ai, fmb, fb, ac)+
              area_star(a, fmb, fb, ft, ac, ai, amax))/area_bound(a)
    rhoeff[a<=ac] = rhoi
    rhoeff[(a>ac)&(a<=ai)] = 920.*(area_bound(ac)+area_gap(a[(a>ac)&(a<=ai)], fmb, fb, ac))/area_bound(a[(a>ac)&(a<=ai)])

    # hard limit on minimum rhoeff
    rhoeff = np.maximum(rhoeff, 150.)

    return rhoeff
