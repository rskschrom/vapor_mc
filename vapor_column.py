import numpy as np
from math import isnan
import vapor as va
import single_particle as sp
import forward as fo
import micro as mi
import thermo as th
from init import interp_sounding
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.pyplot as plt

# function for creating color map
def createCmap(mapname):
    fil = open(f'color_tables/{mapname}.rgb')
    cdata = np.genfromtxt(fil, skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap

# set model run parameters
nt = 5400
nbin = 31
npar_max = 250000
npar_init = 100000
nagg_max = 50000

nscale = 1000.
dt = 1.
dz = 100.
ze = np.arange(nbin+1)*dz

# set radar parameters
wavl = 32.1
k = 2.*np.pi/wavl
diel = 3.17+1j*0.009
bval = 51.
rad_freq = 20
nprof = int(nt/rad_freq)

# set initial particle properties
z0 = 2500.
zvar = 100.
vola = np.zeros([npar_max])
volc = np.zeros([npar_max])
z = np.zeros([npar_max])
zagg = np.zeros([nagg_max])
mass_agg = np.zeros([nagg_max])

rho_agg = 100.
eff_agg = 0.3

tmpk = np.full([npar_max], 258.)
si = np.full([npar_max], 0.15)
pres = np.full([npar_max], 85000.)

# particle mask
parm = np.zeros([npar_max], dtype=np.bool_)
parm[:npar_init] = 1
aggm = np.zeros([nagg_max], dtype=np.bool_)

# randomly sample initial particle properties
res = 10.**np.random.multivariate_normal([-13.5,-13.5],
                                          [[0.2,0.0],
                                           [0.0,0.2]], size=(npar_init))
vola[:npar_init] = res[:,0]
volc[:npar_init] = res[:,1]
z[:npar_init] = np.random.normal(z0, zvar, size=npar_init)

# forward model coefficients
data = np.genfromtxt('coeffs/spheroid_coeff.txt', skip_header=1)
pd_coeffs = data[:,0]
ph_coeffs = data[:,1]

# output variables
lea = 0.
tau = np.empty([nprof,nbin])
tau_bin = np.empty([nbin])
ncol = np.empty([nprof,nbin])
ncol_bin = np.empty([nbin])
zagg2 = np.zeros([nprof,nagg_max])
tagg2 = np.zeros([nprof,nagg_max])

zh = np.empty([nprof,nbin])
zdr = np.empty([nprof,nbin])
kdp = np.empty([nprof,nbin])
rhohv = np.empty([nprof,nbin])
mdv = np.empty([nprof,nbin])
tmpk2d = np.empty([nprof,nbin])
si2d = np.empty([nprof,nbin])

# set initial thermodynamic state from sounding
#fsnd = 'obs/sounding_20180223_4.txt'
fsnd = 'obs/sounding_20180304_2.txt'
#zoffs = 1000.
zoffs = 3800.
pres_bin, tmpk_bin, si_bin = interp_sounding(fsnd, ze+zoffs)

# set conditions at each particle
bin_inds = np.digitize(z[:npar_init], ze)-1

for j in range(nbin):
    pres[:npar_init][bin_inds==j] = pres_bin[j]
    tmpk[:npar_init][bin_inds==j] = tmpk_bin[j]
    si[:npar_init][bin_inds==j] = si_bin[j]

# integrate model
for i in range(nt-1):
    print(f'time step: {i:d} num. pristine: {np.sum(parm):d}'+
          f' num. aggregate: {np.sum(aggm):d} lprecip (in.): {lea:.2e}'+
          f' tmp12: {tmpk_bin[12]:.3f} si12: {si_bin[12]:.4f}')
    dvadt, dvcdt, dmdt = va.dvolacdt(vola, volc, tmpk, si, pres)
    vola_new = vola+dt*dvadt
    volc_new = volc+dt*dvcdt
    vt = mi.fall_speed(vola, volc)
    z = z-dt*vt
    vola = vola_new
    volc = volc_new

    # remove very small particles for negative si
    parm[(si<0.)&(vola_new<1.e-16)] = 0
    parm[(si<0.)&(volc_new<1.e-16)] = 0

    # accumulate particles at surface and mask/remove
    a = (3.*vola[z<0.]**2./(4.*np.pi*volc[z<0.]))**(1./3.)
    rhoe = mi.rhodep(a, 0.1, 0.4, 0.6, 0.7, 920.)
    lea = lea+np.sum(vola[z<0.]*rhoe)/25.4
    parm[z<0.] = 0
    
    # get bin indices for particles height
    bin_inds = np.digitize(z[parm], ze)-1
    bina_inds = np.digitize(zagg[aggm], ze)-1

    # get unmasked slices of pristine arrays
    vola_valid = vola[parm]
    volc_valid = volc[parm]
    vt_valid = vt[parm]
    dmdt_valid = dmdt[parm]
    tmpk_valid = tmpk[parm]
    si_valid = si[parm]
    z_valid = z[parm]
    pres_valid = pres[parm]
    parm_valid = parm[parm]
    pinds_valid = np.arange(npar_max)[parm]
    
    # get unmasked slices of aggregate arrays
    za_valid = zagg[aggm]
    massa_valid = mass_agg[aggm]
    aggm_valid = aggm[aggm]
    vta_valid = mi.fall_speed(massa_valid/rho_agg, massa_valid/rho_agg)
    ainds_valid = np.arange(nagg_max)[aggm]
    
    # loop over height bins to adjust thermo and do aggregation
    for j in range(nbin):
        # get pristine thermodynamic array slices within height bin
        dmdt_hb = dmdt_valid[bin_inds==j]
        tmpk_hb = tmpk_valid[bin_inds==j]
        pres_hb = pres_valid[bin_inds==j]
        si_hb = si_valid[bin_inds==j]
        
        # particle array mask indices
        pinds_hb = pinds_valid[bin_inds==j]
        ainds_hb = ainds_valid[bina_inds==j]
    
        # adjust temp and humidity based on vapor growth
        rho_vap = th.svp_ice(tmpk_bin[j])*(si_bin[j]+1.)/(462.*tmpk_bin[j])
        mass_change = np.sum(dmdt_hb*nscale*dt/dz)
        
        new_rho_vap = rho_vap-mass_change
        tmpk_bin[j] = tmpk_bin[j]+th.lheat_sub(tmpk_bin[j])/1.8e-2*mass_change/(pres_bin[j]/(287.*tmpk_bin[j]))/1004.
        si_bin[j] = 462.*tmpk_bin[j]*new_rho_vap/th.svp_ice(tmpk_bin[j])-1.

        # set all particles within bin to new bin thermo state
        tmpk_hb = tmpk_bin[j]
        si_hb = si_bin[j]
        pres_hb = pres_bin[j]
        tmpk[pinds_hb] = tmpk_hb
        si[pinds_hb] = si_hb
        pres[pinds_hb] = pres_hb
        
        # aggregation
        #---------------------------------------------------
        parm_hb = parm_valid[bin_inds==j]
        aggm_hb = aggm_valid[bina_inds==j]
        if (np.sum(parm_hb)+np.sum(aggm_hb))>10:
            # binned pristine arrays
            vola_hb = vola_valid[bin_inds==j]
            volc_hb = volc_valid[bin_inds==j]
            vt_hb = vt_valid[bin_inds==j]
            z_hb = z_valid[bin_inds==j]
        
            # get pristine physical properties
            a = 1.e3*(3.*vola_hb/(4.*np.pi*volc_hb/vola_hb))**(1./3.)
            c = a*volc_hb/vola_hb
            rhoe = mi.rhoeff(a, 0.1, 0.4, 0.6, 0.7, 920.)
            mass_par = (rhoe*vola_hb)
        
            # binned aggregate arrays
            massa_hb = massa_valid[bina_inds==j]
            vta_hb = vta_valid[bina_inds==j]
            za_hb = za_valid[bina_inds==j]
        
            # number of collisions from statistical mechanics
            nj = np.sum(parm_hb)+np.sum(aggm_hb)
            vt_mn = np.mean(np.concatenate((vt_hb, vta_hb)))
            rad_mn = np.mean(((3.*np.concatenate((vola_hb, massa_hb/rho_agg)))/(4.*np.pi))**(1./3.))
            tau_bin[j] = 1./(4.*np.pi*nj*nscale/dz*vt_mn*rad_mn**2.)
            ncol_bin[j] = nj*nscale*(1.-np.exp(-1./tau_bin[j]))
            nagg_exp = eff_agg*ncol_bin[j]/nscale
            #print(f'fraction of collisions: {nagg_exp:.1f}')
        
            if not isnan(nagg_exp):
                # random sampling for fractional expected # of collisions
                nagg_sam = int(int(nagg_exp)+np.heaviside(nagg_exp-np.random.rand(), 1.))
                if nagg_sam>0:
                    # get pristine physical properties
                    a = 1.e3*(3.*vola_hb/(4.*np.pi*volc_hb/vola_hb))**(1./3.)
                    c = a*volc_hb/vola_hb
                    rhoe = mi.rhoeff(a, 0.1, 0.4, 0.6, 0.7, 920.)
                    mass_hb = (rhoe*vola_hb)
                
                    # collate pristine and aggregate arrays
                    mass_comb = np.concatenate((mass_hb, massa_hb))
                    z_comb = np.concatenate((z_hb, za_hb))
                    comb_hb = np.concatenate((parm_hb, aggm_hb))
                                
                    # favor larger sizes as the "aggregator", random for "aggregated"
                    ncomb_hb = np.sum(comb_hb)
                    ind_sample = np.random.choice(np.arange(ncomb_hb), size=(5,nagg_sam))
                    mass_sample = mass_comb[ind_sample]
                    maxind_mass = np.argmax(mass_sample, axis=0)
                    ind_agtr = ind_sample[maxind_mass,np.arange(nagg_sam)]
                    comb_hb[ind_agtr] = 0
                
                    ind_agtd = np.random.choice(np.arange(ncomb_hb)[comb_hb], size=(nagg_sam))
                    comb_hb[ind_agtd] = 0

                    # calculate aggregate physical properties from constituents
                    arnd_ind = np.random.choice(np.arange(nagg_max)[aggm==0], size=nagg_sam)
                    mass_agg[arnd_ind] = mass_comb[ind_agtr]+mass_comb[ind_agtd]
                    aggm[arnd_ind] = 1
                    zagg[arnd_ind] = np.random.rand()*dz+ze[j]
                
                    # update original pristine and aggregate mask arrays
                    parm_hb = comb_hb[:len(parm_hb)]
                    aggm_hb = comb_hb[len(parm_hb):]
                    #parm_valid[bin_inds==j] = parm_hb
                    #aggm_valid[bina_inds==j] = aggm_hb
                    parm[pinds_hb] = parm_hb
                    aggm[ainds_hb] = aggm_hb
                    #print(mass_agg[arnd_ind], mass_par[pind_agtr], mass_par[pind_agtd])

    # update pristine array mask            
    #parm[parm] = parm_valid
    #aggm[aggm] = aggm_valid
    
    # aggregate sedimentation (simple for now)
    vt_agg = mi.fall_speed(mass_agg/rho_agg, mass_agg/rho_agg)
    zagg = zagg-dt*vt_agg
    lea = lea+np.sum(mass_agg[zagg<0.])/25.4
    aggm[zagg<0.] = 0

    # simulate radar variables
    if (i % rad_freq)==0:
        pi = int(i/rad_freq)
        par_props = [vola[parm], volc[parm], z[parm], vt[parm]]
        agg_props = [mass_agg[aggm], zagg[aggm], vt_agg[aggm], rho_agg]
        mass_valid = mass_agg[aggm]
        zh[pi,:], zdr[pi,:], kdp[pi,:], rhohv[pi,:], mdv[pi,:] = fo.radar_agg(par_props, agg_props,
                                                                              wavl, diel, bval,
                                                                              ze, nscale,
                                                                              pd_coeffs, ph_coeffs)
        tmpk2d[pi,:] = tmpk_bin
        si2d[pi,:] = si_bin
        vt_mn = np.mean(vt_valid[bin_inds==j])
        rad_mn = np.mean(((3.*vola_valid[bin_inds==j])/(4.*np.pi))**(1./3.))
        #tau[pi,:] = tau_bin
        #ncol[pi,:] = ncol_bin
        #zagg2[pi,:] = zagg
        #tagg2[pi,:] = 1.*i
        
    # nucleate new ice
    res = 10.**np.random.multivariate_normal([-13.,-13.],
                                              [[0.2,0.0],
                                               [0.0,0.2]], size=(1))
    rnd_ind = np.random.choice(np.arange(npar_max)[parm==0])
    vola[rnd_ind] = res[:,0]
    volc[rnd_ind] = res[:,1]
    z[rnd_ind] = np.random.normal(z0, zvar, size=1)
    parm[rnd_ind] = 1
    bins_new = np.digitize(z[rnd_ind], ze)-1
    tmpk[rnd_ind] = tmpk_bin[bins_new]
    si[rnd_ind] = si_bin[bins_new]

# plot
#----------------------------------
# change fonts
mpl.rc('font',**{'family':'sans-serif',
                 'sans-serif':['Helvetica'],
                 'size':18})
mpl.rc('text', usetex=True)

fig = plt.figure(figsize=(12,12))

# color map stuff
zh_map = createCmap('zh2_map')
zdr_map = createCmap('zdr_map')
rhv_map = createCmap('phv_map')
kdp_map = createCmap('kdp_map')
vel_map = createCmap('vel2_map')

# coordinates
t = np.arange(nprof+1)*rad_freq*dt
z = np.arange(nbin+1)*dz
t2d, z2d = np.meshgrid(t, z, indexing='ij')
tcnt2d, zcnt2d = np.meshgrid(0.5*(t[1:]+t[:-1]), 0.5*(z[1:]+z[:-1]), indexing='ij')

# mask data by reflectivity
zh_thresh =  -30.
zdr = np.ma.masked_where(zh<zh_thresh, zdr)
kdp = np.ma.masked_where(zh<zh_thresh, kdp)
rhohv = np.ma.masked_where(zh<zh_thresh, rhohv)
mdv = np.ma.masked_where(zh<zh_thresh, mdv)
zh = np.ma.masked_where(zh<zh_thresh, zh)

# tick formatters
fmt_null = mpl.ticker.NullFormatter()
fmt_int = mpl.ticker.StrMethodFormatter('{x:.0f}')

# zh
#---------------------------------------------------------------
ax = fig.add_subplot(5,1,1)
plt.pcolormesh(t2d, z2d, zh, cmap=zh_map, vmin=-30., vmax=30.)
cs2 = ax.contour(tcnt2d, zcnt2d, tmpk2d-273.15, levels=[-20.,-15.,-10.], linewidths=2., linestyles='-.', colors='k')
cs3 = ax.contour(tcnt2d, zcnt2d, si2d*100., levels=[5.,10.,15.], linewidths=2., linestyles='--', colors='b')

ax.clabel(cs2, inline=1, fontsize=12, fmt='%3.0f')
ax.clabel(cs3, inline=1, fontsize=12, fmt='%2.0f')

ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('$Z_{H}$', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_null)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar(format='%.0f')
cb.set_label('(dBZ)')

# zdr
#---------------------------------------------------------------
ax = fig.add_subplot(5,1,2)
plt.pcolormesh(t2d, z2d, zdr, cmap=zdr_map, vmin=-2.4, vmax=6.9)

tau = np.ma.masked_invalid(tau)
ncol = np.ma.masked_invalid(ncol)

#cs2 = ax.contour(tcnt2d, zcnt2d, ncol/nscale, levels=[0.1,0.3,0.5,0.7,0.9], linewidths=2., linestyles='--', colors='k')
#ax.clabel(cs2, inline=1, fontsize=12, fmt='%3.f')

ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('$Z_{DR}$', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_null)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar(format='%.1f')
cb.set_label('(dB)')

# kdp
#---------------------------------------------------------------
ax = fig.add_subplot(5,1,3)
plt.pcolormesh(t2d, z2d, kdp, cmap=kdp_map, vmin=-0.8, vmax=2.3)
ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('$K_{DP}$', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_null)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar(format='%.1f')
cb.set_label('(deg/km)')

#plt.scatter(tagg2[:,::10], zagg2[:,::10], c='k', s=10., alpha=0.3)

# rhohv
#---------------------------------------------------------------
ax = fig.add_subplot(5,1,4)
plt.pcolormesh(t2d, z2d, rhohv, cmap=rhv_map, vmin=0.71, vmax=1.05)
#ax.set_xlabel('time (seconds)', fontsize=16)
ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('$\\rho_{HV}$', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_null)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar(format='%.2f')

# mdv
#---------------------------------------------------------------
ax = fig.add_subplot(5,1,5)
plt.pcolormesh(t2d, z2d, mdv, cmap='Spectral_r', vmin=-1., vmax=0.)
ax.set_xlabel('time (seconds)', fontsize=16)
ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('MDV', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_int)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar(format='%.1f')
cb.set_label('(m/s)')

#plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.savefig(f'time_height_vapor_alt.png', bbox_inches='tight')
