import numpy
import corrcal2
from matplotlib import pyplot as plt

nn=5
x=numpy.arange(nn)
xmat=numpy.repeat([x],nn,axis=0)
ymat=xmat.copy().transpose()

scat=0.01

xmat=xmat+scat*numpy.random.randn(xmat.shape[0],xmat.shape[1])
ymat=ymat+scat*numpy.random.randn(ymat.shape[0],ymat.shape[1])

xpos=numpy.reshape(xmat,xmat.size)
ypos=numpy.reshape(ymat,ymat.size)
antvec=numpy.arange(xpos.size)

xx=numpy.repeat([xpos],xmat.size,axis=0)
yy=numpy.repeat([ypos],ymat.size,axis=0).transpose()

antmat=numpy.repeat([antvec],antvec.size,axis=0)
ant1=antmat.copy()
ant2=antmat.copy().transpose()

umat=xx-xx.transpose()
vmat=yy-yy.transpose()
isok=numpy.where(ant2>ant1)

ant1_org=ant1[isok]
ant2_org=ant2[isok]
u_org=umat[isok]
v_org=vmat[isok]
vis_org=numpy.random.randn(ant1_org.size)+numpy.complex(0,1)*numpy.random.randn(ant1_org.size)

noise_org=numpy.ones(u_org.size)

vis,u,v,noise,ant1,ant2,edges=corrcal2.grid_data(vis_org,u_org,v_org,noise_org,ant1_org,ant2_org)
for i in range(len(edges)-1):
    mystd=numpy.std(u[edges[i]:edges[i+1]])+numpy.std(v[edges[i]:edges[i+1]])
    print edges[i],edges[i+1],mystd
v1=numpy.zeros(2*vis.size)
v1[0::2]=1
v2=numpy.zeros(2*vis.size)
v2[1::2]=1
vecs=numpy.vstack([v1,v2])
src=v1*10

big_noise=numpy.zeros(2*noise.size)
big_noise[0::2]=noise
big_noise[1::2]=noise

big_vis=numpy.zeros(2*vis.size)
big_vis[0::2]=numpy.real(vis)
big_vis[1::2]=numpy.imag(vis)

mycov=corrcal2.sparse_2level(big_noise,100*vecs,500*src,2*edges)
guess=numpy.zeros(2*len(ant1))
guess[0::2]=1.0
fac=1000.0
from scipy.optimize import fmin_cg
gvec=numpy.zeros(2*ant1.max()+2)
gvec[0::2]=1.0
gvec=gvec+0.1*numpy.random.randn(gvec.size)
gvec[0]=1
gvec[1]=0

asdf=fmin_cg(corrcal2.get_chisq,gvec*fac,corrcal2.get_gradient,(big_vis+500*src,mycov,ant1,ant2,fac))
