import numpy as np
import vapor as va
from joblib import Parallel, delayed

# set ensemble parameters
nt = 3600
npar = 500
nparam = 10

sigx = 0.3*np.random.rand(nparam)
sigy = 0.3*np.random.rand(nparam)
corr = np.random.rand(nparam)
gam = np.random.rand(nparam)+0.28
tmpk = 258.15
si = 0.15*np.random.rand(nparam)
pres = 85000.
bval = 10.**(2.*np.random.rand(nparam))

# do model runs
zh = np.empty([nparam])
zdr = np.empty([nparam])
kdp = np.empty([nparam])
rhohv = np.empty([nparam])

'''
outfile = open('ensemble.txt', 'w')
outfile.write('sx\tsy\tcxy\tgam\tb\tsi\tzh\tzdr\tkdp\trhohv\n')
for i in range(nparam):
    zh[i], zdr[i], kdp[i], rhohv[i] = va.model(sigx[i], sigy[i], corr[i], gam[i], bval[i], tmpk, si[i], pres, nt, npar)

    outfile.write(f'{sigx[i]:.3f}\t{sigy[i]:.3f}\t{corr[i]:.3f}\t')
    outfile.write(f'{gam[i]:.3f}\t{bval[i]:.1f}\t{si[i]:.3f}\t')
    outfile.write(f'{zh[i]:.1f}\t{zdr[i]:.3f}\t{kdp[i]:.2e}\t{rhohv[i]:.3f}\n')
outfile.close()
'''
res = Parallel(n_jobs=4)(delayed(va.model)(sigx[i], sigy[i], corr[i], gam[i], bval[i], tmpk, si[i], pres, nt, npar) for i in range(nparam))
print(res)

# write output to file
outfile = open('ensemble.txt', 'w')
outfile.write('sx\tsy\tcxy\tgam\tb\tsi\tzh\tzdr\tkdp\trhohv\n')
for i in range(nparam):
    zh = res[i][0]
    zdr = res[i][1]
    kdp = res[i][2]
    rhohv = res[i][3]
    outfile.write(f'{sigx[i]:.3f}\t{sigy[i]:.3f}\t{corr[i]:.3f}\t')
    outfile.write(f'{gam[i]:.3f}\t{bval[i]:.1f}\t{si[i]:.3f}\t')
    outfile.write(f'{zh:.1f}\t{zdr:.3f}\t{kdp:.2e}\t{rhohv:.3f}\n')
outfile.close()
