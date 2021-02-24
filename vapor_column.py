import numpy as np
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
    #fil = open('/home/rschrom/micro/2d_bulk_kinematic/'+mapname+'.rgb')
    fil = open('/home/robert/research/nws/'+mapname+'.rgb')
    cdata = np.genfromtxt(fil,skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap

# set model run parameters
nt = 7200
nbin = 31
npar_max = 10000
npar_init = 2000
nscale = 5000.
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
#gam = 0.28
z0 = 2500.
zvar = 100.
vola = np.zeros([npar_max])
volc = np.zeros([npar_max])
z = np.zeros([npar_max])
tmpk = np.full([npar_max], 258.)
si = np.full([npar_max], 0.15)
pres = np.full([npar_max], 85000.)

# particle mask
parm = np.zeros([npar_max], dtype=np.bool_)
parm[:npar_init] = 1

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
zloc = np.empty([nprof,nbin])
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
print(pres_bin)
print(tmpk_bin)
print(si_bin)
#si_bin = th.svp_liq(tmpk_bin)/th.svp_ice(tmpk_bin)-1.

'''
# test sounding interpolation
data = np.genfromtxt(fsnd, skip_header=1)
psnd = data[:,0]
tsnd = data[:,1]
zsnd = data[:,5]
plt.plot(tmpk_bin, 0.5*(ze[:-1]+ze[1:])+zoffs, 'k-', lw=3., alpha=0.3)
plt.plot(tsnd, zsnd, 'r-', lw=3., alpha=0.3)
ax = plt.gca()
ax.set_ylim([0., 4000.])
plt.savefig('test_snd.png')
'''
# set conditions at each particle
bin_inds = np.digitize(z[:npar_init], ze)-1

for j in range(nbin):
    pres[:npar_init][bin_inds==j] = pres_bin[j]
    tmpk[:npar_init][bin_inds==j] = tmpk_bin[j]
    si[:npar_init][bin_inds==j] = si_bin[j]

# integrate model
for i in range(nt-1):
    print(f'time step: {i:d} num. particles: {np.sum(parm):d} lprecip (in.): {lea:.2e} tmp12: {tmpk_bin[12]:.3f} si12: {si_bin[12]:.4f}')
    dvadt, dvcdt, dmdt = va.dvolacdt(vola, volc, tmpk, si, pres)
    vola_new = vola+dt*dvadt
    volc_new = volc+dt*dvcdt
    vt = mi.fall_speed(vola, volc)
    z = z-dt*vt

    # remove very small particles for negative si
    parm[(si<0.)&(vola_new<1.e-16)] = 0
    parm[(si<0.)&(volc_new<1.e-16)] = 0

    # accumulate particles at surface and mask/remove
    a = (3.*vola[z<0.]**2./(4.*np.pi*volc[z<0.]))**(1./3.)
    rhoe = mi.rhodep(a, 0.1, 0.4, 0.6, 0.7, 920.)
    lea = lea+np.sum(vola[z<0.]*rhoe)/25.4
    parm[z<0.] = 0

    # advance to next step
    vola = vola_new
    volc = volc_new

    # adjust thermodynamic environment (binned appx.)
    dmdt_valid = dmdt[parm]
    tmpk_valid = tmpk[parm]
    si_valid = si[parm]
    z_valid = z[parm]
    pres_valid = pres[parm]
    bin_inds = np.digitize(z[parm], ze)-1
    
    mass_ch_bin = np.zeros([nbin])
    for j in range(nbin):
        # adjust temp and humidity based on vapor growth
        rho_vap = th.svp_ice(tmpk_bin[j])*(si_bin[j]+1.)/(462.*tmpk_bin[j])
        mass_change = np.sum(dmdt_valid[bin_inds==j]*nscale*dt/dz)
        mass_ch_bin[j] = mass_change
        #if len(z_valid[bin_inds==j])>0:
        #    print(np.min(z_valid[bin_inds==j]), np.max(z_valid[bin_inds==j]), ze[j], ze[j+1])
        new_rho_vap = rho_vap-mass_change
        tmpk_bin[j] = tmpk_bin[j]+th.lheat_sub(tmpk_bin[j])/1.8e-2*mass_change/(pres_bin[j]/(287.*tmpk_bin[j]))/1004.
        si_bin[j] = 462.*tmpk_bin[j]*new_rho_vap/th.svp_ice(tmpk_bin[j])-1.

        # set all particles within bin to new bin thermo state
        tmpk_valid[bin_inds==j] = tmpk_bin[j]
        si_valid[bin_inds==j] = si_bin[j]
        pres_valid[bin_inds==j] = pres_bin[j]
        tmpk[parm] = tmpk_valid
        si[parm] = si_valid
        pres[parm] = pres_valid

    # simulate radar variables
    if (i % rad_freq)==0:
        pi = int(i/rad_freq)
        zh[pi,:], zdr[pi,:], kdp[pi,:], rhohv[pi,:], mdv[pi,:] = fo.radar_column(vola[parm],
                                                                                 volc[parm],
                                                                                 z[parm],
                                                                                 vt[parm],
                                                                                 wavl, diel, bval,
                                                                                 ze, nscale,
                                                                                 pd_coeffs, ph_coeffs)
        print(zh[pi,:])
        #print(tmpk_bin)
        #print(si_bin)
        #print(mass_ch_bin)
        tmpk2d[pi,:] = tmpk_bin
        si2d[pi,:] = si_bin

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
    #print(tmpk[rnd_ind], si[rnd_ind])

# plot
#----------------------------------
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
zh_thresh =  -40.
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
plt.pcolormesh(t2d, z2d, zh, cmap=zh_map, vmin=-40., vmax=20.)
cs2 = ax.contour(tcnt2d, zcnt2d, tmpk2d-273.15, levels=[-20.,-15.,-10.], linewidths=2., linestyles='-.', colors='k')
cs3 = ax.contour(tcnt2d, zcnt2d, si2d*100., levels=[5.,10.,15.], linewidths=2., linestyles='--', colors='b')

ax.clabel(cs2, inline=1, fontsize=12, fmt='%3.0f')
ax.clabel(cs3, inline=1, fontsize=12, fmt='%2.0f')

ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('$Z_{H}$', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_null)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar()
cb.set_label('(dBZ)')

#plt.scatter(t2d[1:,1:], zloc, c='k', s=20., alpha=0.3)

# zdr
#---------------------------------------------------------------
ax = fig.add_subplot(5,1,2)
plt.pcolormesh(t2d, z2d, zdr, cmap=zdr_map, vmin=-2.4, vmax=6.9)
ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('$Z_{DR}$', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_null)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar()
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

cb = plt.colorbar()
cb.set_label('(deg/km)')

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

cb = plt.colorbar()

# mdv
#---------------------------------------------------------------
ax = fig.add_subplot(5,1,5)
plt.pcolormesh(t2d, z2d, mdv, cmap='Spectral_r', vmin=-0.5, vmax=0.)
ax.set_xlabel('time (seconds)', fontsize=16)
ax.set_ylabel('height (m)', fontsize=16)
ax.set_title('MDV', fontsize=18, x=0., y=1.02, ha='left')
ax.xaxis.set_major_formatter(fmt_int)
ax.yaxis.set_major_formatter(fmt_int)
ax.set_ylim([0.,3000.])

cb = plt.colorbar()
cb.set_label('(m/s)')

#plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.savefig(f'time_height_vapor.png', bbox_inches='tight')
