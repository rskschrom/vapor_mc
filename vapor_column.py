import numpy as np
import vapor as va
import single_particle as sp
import forward as fo
import micro as mi
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# function for creating color map
def createCmap(mapname):
    #fil = open('/home/rschrom/micro/2d_bulk_kinematic/'+mapname+'.rgb')
    fil = open('/home/robert/research/nws/'+mapname+'.rgb')
    cdata = np.genfromtxt(fil,skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap

# set initial particle properties
nt = 3600
dt = 1.
npar_max = 5000
npar_init = 2000
lea = 0.

gam = 0.28
z0 = 800.
vola = np.zeros([npar_max])
volc = np.zeros([npar_max])
z = np.zeros([npar_max])
parm = np.zeros([npar_max], dtype=np.bool_)
parm[:npar_init] = 1
res = 10.**np.random.multivariate_normal([-13.5,-13.5],
                                          [[0.2,0.0],
                                           [0.0,0.2]], size=(npar_init))
#vola[:,0] = res[:,0]
#volc[:,0] = res[:,1]
vola[:npar_init] = res[:,0]
volc[:npar_init] = res[:,1]
z[:npar_init] = np.random.normal(z0, 100., size=npar_init)

# set conditions at each particle
tmpk = np.full(npar_max, 258.15)
si = np.full(npar_max, 0.15)
pres = 85000.

# forward model coefficients
data = np.genfromtxt('coeffs/spheroid_coeff.txt', skip_header=1)
pd_coeffs = data[:,0]
ph_coeffs = data[:,1]

# radar parameters
wavl = 32.1
k = 2.*np.pi/wavl
diel = 3.17+1j*0.009
bval = 51.

nbin = 61
dz = 50.
beam_width_z = 100.
ze = np.arange(nbin+1)*dz
nscale = 500.
rad_freq = 20
nprof = int(nt/rad_freq)

zloc = np.empty([nprof,nbin])
zh = np.empty([nprof,nbin])
zdr = np.empty([nprof,nbin])
kdp = np.empty([nprof,nbin])
rhohv = np.empty([nprof,nbin])
mdv = np.empty([nprof,nbin])

#zh[0,:], zdr[0,:], kdp[0,:], rhohv[0,:] = fo.radar(vola[:,0], volc[:,0], wavl, diel, bval)

# integrate model
for i in range(nt-1):
    print(i, np.sum(parm), lea)
    dvadt, dvcdt, dmdt = va.dvolacdt(vola, volc, tmpk, si, pres, gam)
    vola_new = vola+dt*dvadt
    volc_new = volc+dt*dvcdt
    vt = mi.fall_speed(vola, volc)
    z = z-dt*vt

    # advance to next step
    vola = vola_new
    volc = volc_new

    # simulate radar variables
    if (i % rad_freq)==0:
        pi = int(i/rad_freq)
        zh[pi,:], zdr[pi,:], kdp[pi,:], rhohv[pi,:], mdv[pi,:] = fo.radar_column(vola[parm],
                                                                                 volc[parm],
                                                                                 z[parm],
                                                                                 vt[parm],
                                                                                 wavl, diel, bval,
                                                                                 ze, beam_width_z, nscale,
                                                                                 pd_coeffs, ph_coeffs)
        print(zh[pi,:])

    # nucleate new ice
    res = 10.**np.random.multivariate_normal([-13.,-13.],
                                              [[0.2,0.0],
                                               [0.0,0.2]], size=(1))
    rnd_ind = np.random.choice(np.arange(npar_max)[parm==0])
    vola[rnd_ind] = res[:,0]
    volc[rnd_ind] = res[:,1]
    z[rnd_ind] = np.random.normal(z0, 100., size=1)
    parm[rnd_ind] = 1

    # accumulate particles at surface and mask/remove
    a = (3.*vola[z<0.]**2./(4.*np.pi*volc[z<0.]))**(1./3.)
    rhoe = mi.rhodep(a, 0.1, 0.4, 0.6, 0.7, 920.)
    lea = lea+np.sum(vola[z<0.]*rhoe)/25.4
    parm[z<0.] = 0

# plot
#----------------------------------
fig = plt.figure(figsize=(16,12))

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
ax.xaxis.set_major_formatter(fmt_int)
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
