struct sparse_2level {
  double *diag;
  double *vecs;
  double *src;
  int n;
  int nvec;
  int nsrc;
  int nblock;
  long *lims;
  int isinv;
};

struct sparse_2level *fill_struct_sparse(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, int isinv);



