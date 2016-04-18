import numpy,corrcal2,time
from matplotlib import pyplot as plt
reload(corrcal2)
#You'll need to compile corrcal2_funs.c into a shared library with e.g.
#gcc-4.9 -fopenmp -std=c99 -O3 -shared -fPIC -o libcorrcal2_funs.so corrcal2_funs.c -lm -lgomp   
#the library will need to be in your LD_LIBRARY_PATH.  If it doesn't show up
#and you compiled in the current directory, either do
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:.
#or change the ctypes.cdll.LoadLibrary call in corrcal2.py to have your full path location


#read relevant data.
mat=corrcal2.read_sparse('signal_sparse2_test.dat')
f=open('ant1.dat');ant1=numpy.fromfile(f,'int64')-1;f.close()
f=open('ant2.dat');ant2=numpy.fromfile(f,'int64')-1;f.close()
f=open('gtmp.dat');gvec=numpy.fromfile(f,'float64');f.close()
f=open('vis.dat');data=numpy.fromfile(f,'float64');f.close()

#if you want to test timings, you can do so here.  Set t_min to some length of 
#time, and code will see how many gradient operations it can get through 
#during at least t_min seconds
niter=0;
t_min=1e-4
t1=time.time()
while (time.time()-t1)<t_min:
    grad=corrcal2.get_gradient(gvec,data,mat,ant1,ant2)
    niter=niter+1
t2=time.time()
nant=gvec.size/2
#time per gradient
print 'average time was ' + repr((time.time()-t1)/niter)
#time (in microseconds) per visibility
print 'scaled_time was ' + repr( (t2-t1)/niter/nant/(nant-1)*1e6)


#scipy nonlinear conjugate gradient seems to work pretty well.
#note that it can use overly large step sizes in trials causing
#matrices to go non-positive definite.  If you rescale the gains
#by some large factor, this seems to go away.  If you routinely
#hit non-positive definite conditions, try increasing fac (or writing your
#own minimizer...)
from scipy.optimize import fmin_cg
fac=1000.0;
t1=time.time()
asdf=fmin_cg(corrcal2.get_chisq,gvec*fac,corrcal2.get_gradient,(data,mat,ant1,ant2,fac))
t2=time.time()
print 'elapsed time to do nonlinear fit for ' + repr(nant) + ' antennas was ' + repr(t2-t1)
fit_gains=asdf/fac
