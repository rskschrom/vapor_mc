import numpy as np

# Murphy + Koop (2005) RMET (Pa; T > 100 K)
def svp_ice(tmpk):
    svp_ice = np.exp(9.550426-5723.265/tmpk+3.53068*np.log(tmpk)-0.00728332*tmpk)
    return svp_ice

        
   

# Murphy + Koop (2005) RMET (Pa; 332 K > T > 123 K)
def svp_liq(tmpk):
    lsvp_liq = 54.842763-6763.22/tmpk-4.210*np.log(tmpk)+0.000367*tmpk+\
               np.tanh(0.0415*(tmpk-218.8))*\
               (53.878-1331.22/tmpk-9.44523*np.log(tmpk)+0.014025*tmpk)
    svp_liq = np.exp(lsvp_liq)
    return svp_liq

# Murphy + Koop (2005) RMET (J mol-1; 273 K > T > 236 K)
def lheat_vap(tmpk):
    lheat_vap = 56579.-42.212*tmpk+np.exp(0.1149*(281.6-tmpk))
    return lheat_vap

# Murphy + Koop (2005) RMET (J mol-1; T > 30 K)
def lheat_sub(tmpk):
    lheat_sub = 46782.5+35.8925*tmpk-0.07414*tmpk**2.+541.5*np.exp(-(tmpk/123.75)**2.)
    return lheat_sub

# modified diffusivity P+K97 (eqn. 13-14)
def dv_mod(tmpk, pres, rad):
    delv = 1.3*8.e-8
    alpc = 0.036

    # regular diffusivity (m2 s-1; PK97 eqn. 13-3)
    dv = 0.211e-4*(tmpk/273.15)**1.94*(101325./pres)

    # modified diffusivity
    dv_mod = dv/(rad/(rad+delv)+dv/(rad*alpc)*np.sqrt(2.*np.pi*18.e-3/(8.314*tmpk)))
    
    return dv_mod

# modified conductivity P+K97 (eqn. 13-20)
def ka_mod(tmpk, pres, rad):
    delt = 2.16e-7
    alpt = 0.7
    cpa = 1004.
    rho = pres/(287.*tmpk)

    # regular conductivity (J m-1 s-1 k-1; PK97 eqn. 13-18a)
    ka = (5.69+0.017*(tmpk-273.15))*1.e-3

    # modified diffusivity
    ka_mod = 4.184*ka/(rad/(rad+delt)+4.184*ka/(rad*alpt*rho*cpa)*np.sqrt(2.*np.pi*29.e-3/(8.314*tmpk)))
    
    return ka_mod

# combined diffusivity
def g_diff(tmpk, pres, si, rad):
    ka = ka_mod(tmpk, pres, rad)
    dv = dv_mod(tmpk, pres, rad)
    vapi = (si+1.)*svp_ice(tmpk)
    rv = 8.314/(18.e-3)
    ls = lheat_sub(tmpk)/(18.e-3)

    g_diff = 1./(rv*tmpk/(vapi*dv)+ls/(tmpk*ka)*(ls/(tmpk*rv)-1.))        

    return g_diff
