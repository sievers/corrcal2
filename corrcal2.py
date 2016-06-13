import numpy 
import ctypes
import time
try:
    import pyfof
    have_fof=True
except:
    have_fof=False

mylib=ctypes.cdll.LoadLibrary("libcorrcal2_funs.so")

sparse_mat_times_vec_c=mylib.sparse_mat_times_vec_wrapper
sparse_mat_times_vec_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]

make_small_block_c=mylib.make_small_block
make_small_block_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

make_all_small_blocks_c=mylib.make_all_small_blocks
make_all_small_blocks_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
#void make_all_small_blocks(double *diag, double *vecs, long *lims, int nblock, int n, int nsrc, double *out)

chol_c=mylib.chol
chol_c.argtypes=[ctypes.c_void_p,ctypes.c_int]

many_chol_c=mylib.many_chol
many_chol_c.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]

tri_inv_c=mylib.tri_inv
tri_inv_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]

many_tri_inv_c=mylib.many_tri_inv
many_tri_inv_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]

invert_all_small_blocks_c=mylib.invert_all_small_blocks
invert_all_small_blocks_c.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

mymatmul_c=mylib.mymatmul
mymatmul_c.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int] 

mult_vecs_by_blocs_c=mylib.mult_vecs_by_blocs
mult_vecs_by_blocs_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]

apply_gains_to_mat_c=mylib.apply_gains_to_mat
apply_gains_to_mat_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]

apply_gains_to_mat_dense_c=mylib.apply_gains_to_mat_dense
apply_gains_to_mat_dense_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]


sum_grads_c=mylib.sum_grads
sum_grads_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]

class sparse_2level:
    def __init__(self,diag,vecs,src,lims,isinv=0):
        self.diag=diag.copy()
        self.vecs=numpy.matrix(vecs.copy())
        self.src=numpy.matrix(src.copy())
        #self.lims=lims.copy()
        self.lims=numpy.zeros(len(lims),dtype='int64')
        self.lims[:]=lims
        self.isinv=isinv
        self.nblock=len(lims)-1
    def copy(self):
        return sparse_2level(self.diag,self.vecs,self.src,self.lims,self.isinv)
    def __mul__(self,vec):
        ans=numpy.zeros(vec.shape)
        #void sparse_mat_times_vec_wrapper(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, double *vec, double *ans)
        n=self.diag.size
        nvec=self.vecs.shape[0]
        nsrc=self.src.shape[0]
        sparse_mat_times_vec_c(self.diag.ctypes.data,self.vecs.ctypes.data,self.src.ctypes.data,n,nvec,nsrc,self.nblock,self.lims.ctypes.data,self.isinv,vec.ctypes.data,ans.ctypes.data)
        return ans
    def expand(self):
        m1=self.src.transpose()*self.src
        for i in range(0,self.nblock):
            i1=self.lims[i]
            i2=self.lims[i+1]
            tmp=self.vecs[:,i1:i2].transpose()*self.vecs[:,i1:i2]
            m1[i1:i2,i1:i2]+=tmp            
        mm=numpy.diag(self.diag)
        if (self.isinv):
            return mm-m1
        else:
            return mm+m1
    def inv(self):

        t1=time.time();
        myinv=self.copy()
        myinv.isinv=~self.isinv
        myinv.diag=1.0/self.diag
        nvec=self.vecs.shape[0]
        nn=self.diag.size
        tmp=numpy.zeros([self.nblock,nvec,nvec])
        
        make_all_small_blocks_c(self.diag.ctypes.data,self.vecs.ctypes.data,self.lims.ctypes.data,self.nblock,nn,nvec,tmp.ctypes.data)

        myeye=numpy.repeat([numpy.eye(nvec)],self.nblock,axis=0);
        if self.isinv:
            tmp2=myeye-tmp
        else:
            tmp2=myeye+tmp
        many_chol_c(tmp2.ctypes.data,nvec,self.nblock)
        tmp3=many_tri_inv(tmp2)
        tmp4=mult_vecs_by_blocks(self.vecs,tmp3,self.lims)
        
        for i in range(tmp4.shape[0]):
            tmp4[i,:]=tmp4[i,:]*myinv.diag
        #return tmp4
        #invert_all_small_blocks_c(tmp.ctypes.data,self.nblock,nvec,numpy.int(self.isinv),tmp2.ctypes.data)
        t2=time.time()
        #print 'took ' + repr(t2-t1) + ' seconds to do inverse.'

        #return tmp,tmp3,tmp4
        myinv.vecs[:]=tmp4
        
        nsrc=self.src.shape[0]
        tmp=0*self.src
        n=self.diag.size
        nvec=self.vecs.shape[0]
        nblock=self.lims.size-1
        dptr=myinv.diag.ctypes.data
        sptr=myinv.src.ctypes.data
        vptr=myinv.vecs.ctypes.data
        #we can do the block multiply simply by sending in 0 for nsrc
        for i in range(nsrc):
            sparse_mat_times_vec_c(dptr,vptr,sptr,n,nvec,0,nblock,self.lims.ctypes.data,myinv.isinv,self.src[i].ctypes.data,tmp[i].ctypes.data)
        
        small_mat=tmp*self.src.transpose()
        if self.isinv:
            small_mat=numpy.eye(nsrc)-small_mat
        else:
            small_mat=numpy.eye(nsrc)+small_mat
        
        small_mat=numpy.linalg.inv(numpy.linalg.cholesky(small_mat))
        myinv.src=small_mat*tmp

        return myinv
    def apply_gains_to_mat(self,g,ant1,ant2):
        
        apply_gains_to_mat_c(self.vecs.ctypes.data,g.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,self.vecs.shape[1]/2,self.vecs.shape[0])
        apply_gains_to_mat_c(self.src.ctypes.data,g.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,self.src.shape[1]/2,self.src.shape[0])


def get_chisq_dense(g,data,noise,sig,ant1,ant2,scale_fac=1.0,normfac=1.0):
    g=g/scale_fac
    cov=sig.copy()
    n=sig.shape[0]
    assert(sig.shape[1]==n)

    apply_gains_to_mat_c(cov.ctypes.data,g.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,n/2,n)

    cov=cov.transpose().copy()
    apply_gains_to_mat_c(cov.ctypes.data,g.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,n/2,n)

    cov=cov.transpose().copy()
    cov=cov+noise
    cov=0.5*(cov+cov.transpose())
    cov_inv=numpy.linalg.inv(cov)
    rhs=numpy.dot(cov_inv,data)
    chisq=numpy.sum(data*numpy.asarray(rhs))
    nn=g.size/2
    chisq=chisq+normfac*( (numpy.sum(g[1::2]))**2 + (numpy.sum(g[0::2])-nn)**2)
    print chisq, numpy.mean(g[0::2]),numpy.mean(g[1::2])
    return chisq
def get_gradient_dense(g,data,noise,sig,ant1,ant2,scale_fac=1.0,normfac=1.0):
    do_times=False
    g=g/scale_fac
    cov=sig.copy()
    n=sig.shape[0]
    apply_gains_to_mat_c(cov.ctypes.data,g.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,n/2,n)
    cov=cov.transpose().copy()
    apply_gains_to_mat_c(cov.ctypes.data,g.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,n/2,n)
    cov=cov+noise
    cov=0.5*(cov+cov.transpose())
    cov_inv=numpy.linalg.inv(cov)
    sd=numpy.dot(cov_inv,data)
    
    #make g*(c_inv)*d
    gsd=sd.copy()
    apply_gains_to_mat_c(gsd.ctypes.data,g.ctypes.data,ant2.ctypes.data,ant1.ctypes.data,n/2,1)

    cgsd=numpy.dot(gsd,sig)

    tmp=cgsd.copy()
    cgsd=numpy.zeros(tmp.size)
    cgsd[:]=tmp[:]

    tmp=sd.copy()
    sd=numpy.zeros(tmp.size)
    sd[:]=tmp[:]

    tmp=gsd.copy()
    gsd=numpy.zeros(tmp.size)
    gsd[:]=tmp[:]

    nant=numpy.max([numpy.max(ant1),numpy.max(ant2)])+1
    grad=numpy.zeros(2*nant)

    r1=g[2*ant1]
    r2=g[2*ant2]
    i1=g[2*ant1+1]
    i2=g[2*ant2+1]
    m1r_v2=0*cgsd
    m1i_v2=0*cgsd
    m2r_v2=0*cgsd
    m2i_v2=0*cgsd
    
    m1r_v2[0::2]=r2*sd[0::2]-i2*sd[1::2];
    m1r_v2[1::2]=i2*sd[0::2]+r2*sd[1::2];
    m1i_v2[0::2]=i2*sd[0::2]+r2*sd[1::2];
    m1i_v2[1::2]=-r2*sd[0::2]+i2*sd[1::2];
    m2r_v2[0::2]=r1*sd[0::2]+i1*sd[1::2];
    m2r_v2[1::2]=-i1*sd[0::2]+r1*sd[1::2];
    m2i_v2[0::2]=i1*sd[0::2]-r1*sd[1::2];
    m2i_v2[1::2]=r1*sd[0::2]+i1*sd[1::2];


    if do_times:
        t2=time.time();
        print t2-t1

    v1_m1r_v2=cgsd*m1r_v2;v1_m1r_v2=v1_m1r_v2[0::2]+v1_m1r_v2[1::2];
    v1_m1i_v2=cgsd*m1i_v2;v1_m1i_v2=v1_m1i_v2[0::2]+v1_m1i_v2[1::2];
    v1_m2r_v2=cgsd*m2r_v2;v1_m2r_v2=v1_m2r_v2[0::2]+v1_m2r_v2[1::2];
    v1_m2i_v2=cgsd*m2i_v2;v1_m2i_v2=v1_m2i_v2[0::2]+v1_m2i_v2[1::2];
    if do_times:
        t2=time.time();
        print t2-t1

    #print v1_m1r_v2[0:5]

    sum_grads_c(grad.ctypes.data,v1_m1r_v2.ctypes.data,v1_m1i_v2.ctypes.data,ant1.ctypes.data,v1_m2i_v2.size)
    sum_grads_c(grad.ctypes.data,v1_m2r_v2.ctypes.data,v1_m2i_v2.ctypes.data,ant2.ctypes.data,v1_m2i_v2.size)
    if do_times:
        t2=time.time();
        print t2-t1
    #chisq=numpy.sum(sd*data)
    #print chisq


    nn=g.size/2.0
    grad_real=2*(numpy.sum(g[0::2])-nn)/nn
    grad_im=2*numpy.sum(g[1::2])



    return -2*grad/scale_fac + normfac*(grad_real+grad_im)/scale_fac




def get_chisq(g,data,mat,ant1,ant2,scale_fac=1.0,normfac=1.0):
    g=g/scale_fac
    do_times=False
    if do_times:
        t1=time.time()
    mycov=mat.copy()
    mycov.apply_gains_to_mat(g,ant1,ant2)
    if do_times:
        t2=time.time();
        print t2-t1    
    mycov_inv=mycov.inv()
    if do_times:
        t2=time.time();
        print t2-t1
    sd=mycov_inv*data
    chisq=numpy.sum(sd*data)

    nn=g.size/2
    chisq=chisq+normfac*( (numpy.sum(g[1::2]))**2 + (numpy.sum(g[0::2])-nn)**2)
    

    print chisq
    return chisq

def get_gradient(g,data,mat,ant1,ant2,scale_fac=1.0,normfac=1.0):
    g=g/scale_fac
    do_times=False
    if do_times:
        t1=time.time()
    mycov=mat.copy()
    mycov.apply_gains_to_mat(g,ant1,ant2)
    if do_times:
        t2=time.time();
        print t2-t1
    mycov_inv=mycov.inv()
    if do_times:
        t2=time.time();
        print t2-t1
    sd=mycov_inv*data
    gsd=sd.copy();
    apply_gains_to_mat_c(gsd.ctypes.data,g.ctypes.data,ant2.ctypes.data,ant1.ctypes.data,gsd.size/2,1);
    tmp=mat.copy()
    tmp.diag[:]=0
    cgsd=tmp*gsd

    if do_times:
        t2=time.time();
        print t2-t1

    nant=numpy.max([numpy.max(ant1),numpy.max(ant2)])+1
    grad=numpy.zeros(2*nant)

    r1=g[2*ant1]
    r2=g[2*ant2]
    i1=g[2*ant1+1]
    i2=g[2*ant2+1]
    m1r_v2=0*cgsd
    m1i_v2=0*cgsd
    m2r_v2=0*cgsd
    m2i_v2=0*cgsd
    
    m1r_v2[0::2]=r2*sd[0::2]-i2*sd[1::2];
    m1r_v2[1::2]=i2*sd[0::2]+r2*sd[1::2];
    m1i_v2[0::2]=i2*sd[0::2]+r2*sd[1::2];
    m1i_v2[1::2]=-r2*sd[0::2]+i2*sd[1::2];
    m2r_v2[0::2]=r1*sd[0::2]+i1*sd[1::2];
    m2r_v2[1::2]=-i1*sd[0::2]+r1*sd[1::2];
    m2i_v2[0::2]=i1*sd[0::2]-r1*sd[1::2];
    m2i_v2[1::2]=r1*sd[0::2]+i1*sd[1::2];


    if do_times:
        t2=time.time();
        print t2-t1

    v1_m1r_v2=cgsd*m1r_v2;v1_m1r_v2=v1_m1r_v2[0::2]+v1_m1r_v2[1::2];
    v1_m1i_v2=cgsd*m1i_v2;v1_m1i_v2=v1_m1i_v2[0::2]+v1_m1i_v2[1::2];
    v1_m2r_v2=cgsd*m2r_v2;v1_m2r_v2=v1_m2r_v2[0::2]+v1_m2r_v2[1::2];
    v1_m2i_v2=cgsd*m2i_v2;v1_m2i_v2=v1_m2i_v2[0::2]+v1_m2i_v2[1::2];
    if do_times:
        t2=time.time();
        print t2-t1

    #print v1_m1r_v2[0:5]

    sum_grads_c(grad.ctypes.data,v1_m1r_v2.ctypes.data,v1_m1i_v2.ctypes.data,ant1.ctypes.data,v1_m2i_v2.size)
    sum_grads_c(grad.ctypes.data,v1_m2r_v2.ctypes.data,v1_m2i_v2.ctypes.data,ant2.ctypes.data,v1_m2i_v2.size)
    if do_times:
        t2=time.time();
        print t2-t1
    #chisq=numpy.sum(sd*data)
    #print chisq



    nn=g.size/2.0
    grad_real=2*(numpy.sum(g[0::2])-nn)/nn
    grad_im=2*numpy.sum(g[1::2])
    return -2*grad/scale_fac + normfac*(grad_real+grad_im)/scale_fac
    #return -2*grad/scale_fac

def chol(mat):
    n=mat.shape[0]
    chol_c(mat.ctypes.data,n)
    

def many_chol(mat):
    nmat=mat.shape[0]
    n=mat.shape[1]
    many_chol_c(mat.ctypes.data,n,nmat)

def tri_inv(mat):
    n=mat.shape[0]
    mat_inv=0*mat
    tri_inv_c(mat.ctypes.data,mat_inv.ctypes.data,n)
    return mat_inv

def many_tri_inv(mat):
    mat_inv=0*mat
    sz=mat.shape

    #if only one matrix comes in, do the correct thing
    if len(sz)==2:
        tri_inv_c(mat.ctypes.data,mat_inv.ctypes.data,sz[0])
        return mat_inv

    nmat=mat.shape[0]
    n=mat.shape[1]
    many_tri_inv_c(mat.ctypes.data,mat_inv.ctypes.data,n,nmat)
    return mat_inv


def read_sparse(fname):
    f=open(fname)
    n=numpy.fromfile(f,'int32',1);
    isinv=(numpy.fromfile(f,'int32',1)[0]!=0);
    nsrc=numpy.fromfile(f,'int32',1);
    nblock=numpy.fromfile(f,'int32',1);
    nvec=numpy.fromfile(f,'int32',1);
    lims=numpy.fromfile(f,'int32',(nblock+1))
    diag=numpy.fromfile(f,'float64',n)
    vecs=numpy.fromfile(f,'float64',nvec*n)
    src=numpy.fromfile(f,'float64',nsrc*n)
    crap=numpy.fromfile(f)
    f.close()
    if crap.size>0:
        print 'file ' + fname + ' had unexpected length.'
        return

    vecs=vecs.reshape([nvec,n])
    if nsrc>0:
        src=src.reshape([nsrc,n])

    mat=sparse_2level(diag,vecs,src,lims,isinv)
    return mat

def make_uv_grid(u,v,tol=0.01,do_fof=True):
    isconj=(v<0)|((v<tol)&(u<0))
    u=u.copy()
    v=v.copy()
    u[isconj]=-1*u[isconj]
    v[isconj]=-1*v[isconj]
    if (have_fof & do_fof):
        uv=numpy.stack([u,v]).transpose()
        groups=pyfof.friends_of_friends(uv,tol)
        myind=numpy.zeros(len(u))
        for j,mygroup in enumerate(groups):
            myind[mygroup]=j
        ii=numpy.argsort(myind)
        edges=numpy.where(numpy.diff(myind[ii])!=0)[0]+1
    else:
        #break up uv plane into tol-sized blocks
        u_int=numpy.round(u/tol)
        v_int=numpy.round(v/tol)
        uv=u_int+numpy.complex(0,1)*v_int
        ii=numpy.argsort(uv)
        uv_sort=uv[ii]
        edges=numpy.where(numpy.diff(uv_sort)!=0)[0]+1
    edges=numpy.append(0,edges)
    edges=numpy.append(edges,len(u))
    
    #map isconj into post-sorting indexing
    isconj=isconj[ii]
    return ii,edges,isconj


def grid_data(vis,u,v,noise,ant1,ant2,tol=0.1,do_fof=True):
    """Re-order the data into redundant groups.  Inputs are (vis,u,v,noise,ant1,ant2,tol=0.1)
    where tol is the UV-space distance for points to be considered redundant.  Data will be
    reflected to have positive u, or positive v for u within tol of zero.  If pyfof is
    available, use that for group finding."""

    ii,edges,isconj=make_uv_grid(u,v,tol,do_fof)
    tmp=ant1[isconj]
    ant1[isconj]=ant2[isconj]
    ant2[isconj]=tmp
    vis=vis[ii]
    vis[isconj]=numpy.conj(vis[isconj])

    ant1=ant1[ii]
    ant2=ant2[ii]
    noise=noise[ii]


    return vis,u,v,noise,ant1,ant2,edges,ii,isconj


def mymatmul(a,b):
    n=a.shape[0]
    k=a.shape[1]
    kk=b.shape[0]
    m=b.shape[1]
    c=numpy.zeros([n,m])
    
    mymatmul_c(a.ctypes.data,k,b.ctypes.data,m,n,m,k,c.ctypes.data,m)
    return c

def mult_vecs_by_blocks(vecs,blocks,edges):
    n=vecs.shape[1]
    nvec=vecs.shape[0]
    nblock=edges.size-1
    ans=numpy.zeros([nvec,n])
    if (edges.dtype.name!='int64'):
        edges.numpy.asarray(edges,dtype='int64')
    mult_vecs_by_blocs_c(vecs.ctypes.data,blocks.ctypes.data,n,nvec,nblock,edges.ctypes.data,ans.ctypes.data)
    return ans

def make_uv_from_antpos(xyz,rmax=0,tol=0.0):
    """Take a list of antenna positions and create a UV snapshot out of it."""
    xyz=xyz.copy()
    nant=xyz.shape[0]
    if xyz.shape[1]==2:
        xyz=numpy.c_[xyz,numpy.zeros(nant)]
    mymed=numpy.median(xyz,axis=0)
    xyz=xyz-numpy.repeat([mymed],nant,axis=0)
    
    if (rmax>0):
        r=numpy.sqrt(numpy.sum(xyz**2,axis=1))
        xyz=xyz[r<=rmax,:]
        nant=xyz.shape[0]

    #fit to a plane by modelling z=a*x+b*y+c
    mat=numpy.c_[xyz[:,0:2],numpy.ones(nant)]
    lhs=numpy.dot(mat.transpose(),mat)
    rhs=numpy.dot(mat.transpose(),xyz[:,2])
    fitp=numpy.dot(numpy.linalg.inv(lhs),rhs)
    
    
    #now rotate into that plane.  z should now be vertical axis
    vz=numpy.dot(fitp,[1.0,0,0])
    vvec=numpy.asarray([1,0,vz])
    vvec=vvec/numpy.linalg.norm(vvec)
    uz=numpy.dot(fitp,[0,1.0,0])
    uvec=numpy.asarray([0,1,uz])
    uvec=uvec/numpy.linalg.norm(uvec)
    wvec=numpy.cross(uvec,vvec)
    rotmat=numpy.vstack([uvec,vvec,wvec]).transpose()
    xyz=numpy.dot(xyz,rotmat)
    
    x=xyz[:,0].copy()
    xmat=numpy.repeat([x],nant,axis=0)
    y=xyz[:,1].copy()
    ymat=numpy.repeat([y],nant,axis=0)
    antvec=numpy.arange(nant)

    ant1=numpy.repeat([antvec],nant,axis=0)
    ant2=ant1.copy().transpose()
    
    u=xmat-xmat.transpose()
    v=ymat-ymat.transpose()
    u=numpy.tril(u)
    v=numpy.tril(v)

    ii=(numpy.abs(u)>0)&(numpy.abs(v)>0)
    u=u[ii]
    v=v[ii]
    ant1=ant1[ii]
    ant2=ant2[ii]

    ii=(u<0)|((u<tol)&(v<0))
    tmp=ant1[ii]
    ant1[ii]=ant2[ii]
    ant2[ii]=tmp
    u[ii]=u[ii]*-1
    v[ii]=v[ii]*-1

    return u,v,ant1,ant2,xyz
