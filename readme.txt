Once code is set up, the important thing needed to run on ones data is
to get the sparse matrix describing sky and source correlations set
up.  Look at read_sparse in corrcal2.py for an example of how the
matrix is structured.  The key thing is to make sure that the
visibilities are grouped into redundant blocks, and that the
real/imaginary parts are separate/imaginaries follow their respective
reals. 

The important fields are:
diag:  the noise variance of visibilities
lims:  the indices that set off the redundant blocks.
vecs:  the vectors describing the sky covariances within blocks.  It's
       currently assumed that the number of vectors is the same for
       each block.  If you don't want this, you can zero-pad.
src:   the per-visibility response to sources with known positions.
isinv: is the covariance matrix an inverse.  You will start with this 
       flag set to False.

When you have these in place, you can create a sparse matrix with 
mat=corrcal2.sparse_2level(diag,vecs,src,lims,isinv)
Note that if you want to run with classic redundant calibration, the
source vector will be zeros, and the sky vectors will be some large
number times
[1 0 1 0 1 0 1 0...
 0 1 0 1 0 1 0 1....]
which says there's random signal in the real visibilities which is
uncorrelated with the imaginary visibilities.  

To run, you'll also need to get data and per-visibility antenna
1/antenna 2 (assumed zero-offset indexing on the antennas) read in,
plus a (hopefully non-awful) guess at the initial gains.  Then you can
fit for gains with the scipy non-linear conjugate gradient solver
(from scipy.optimize import fmin_cg).  One final wrinkle is that scipy
often tries trial steps far too large for the gain errors, causing
matrix to go non-positive definite.  If you hit this, you can set a
scale factor to some large number until the minimizer behaves.  Look
in corrcal2_example.py (which runs a whole PAPER-sized problem) to see
how this works.
<<<<<<< HEAD


=======
>>>>>>> cae43e04589df767c3d872c7aa07e862335f2c6b
