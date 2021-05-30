"""
Pytorch implementation of efficient matrix-matrix multiplication for
Toeplitz, Block Toeplitz

Notes:
 - Ordering of points must be in C memory order (what fft and ifft expects)
 - beware of reshape/flattening tensors --- must maintain C memory order

acm
"""
import torch
from torch import nn
from ziggy.misc.cg import conj_grad
import numpy as np


def gram_solve(xgrids, kernel_fun, vec, K_matmul=None,
               maxiter=20, do_precond=True,
               tol=1e-10, callback=None, mult_RT=True):
    """ Solves a system of the following form
        K_uu^{-1/2} v = res where K_uu^{-1/2} = R^T Kuu^{-1}
        Hence, we first compute d = Kuu^{-1} v by solving Kuu d = v
        Then, we compute res = R^T d = R^T Kuu^{-1} v

    Where K_uu is M x M and vec.t() is M x L

    Note: vec is L x M because L will typically correspond to batch, and it's
    easier to deal w/ when it's along the first dimension.

    Args:
        - xinduce_grids : list of D grids that add up to M total points
        - kernel_fun : function
        - vec    : bsz x M matrix, where M = m1*m2*...*mD

    Res:
        - res : if return RT: bsz x M' matrix, where M' = (2m1-2)*(2m2-2)*...*(2mD-2)
                else: bsz X M matrix, where M = m1*m2*...*mD
    """
    assert len(vec.shape) == 2
    bsz, M = vec.shape
    if K_matmul is None:
        K_matmul = ToeplitzMatmul(xgrids, kernel_fun, batch_shape=vec.shape[:-1])
    else:
        K_matmul.set_batch_shape(vec.shape[:-1])

    Kmul    = lambda x: K_matmul(x.t(), multiply_type="gram").t()
    precond = None
    if do_precond:
        precond = lambda x: K_matmul(x.t(), multiply_type="circ_inv").t()

    # solve system w/ conjugate gradients
    d = conj_grad(Kmul, vec.t(), precond=precond,
                    maxiter=maxiter, tol=tol, callback=callback)  # (M, bsz)
    if mult_RT:
        res = K_matmul(d.t(), multiply_type="RTv")
        return res
    else:
        return d.t()


class ToeplitzMatmul(nn.Module):
    """ Implementation of a (hierarchical/block) toeplitz structured matrix
    defined by a list of grids and a kernel_fun

    Note ii) these toeplitz multiplies are implemented by embedding the
    block toeplitz matrix into a circulant matrix, which admits fast
    diagonalization via FFT.  This also means the circulant matrix admits
    a fast inverse representation (e.g. just D^{-1}).  However, this does
    not mean we can easily compute K_toe^{-1}*vec products easily --- the
    upper left block will not be K_toe^{-w

    Note: it turns out most of the time in the forward multiplication is in
    allocation of memory --- notably the allocation of complex
    numbers (i.e. when the complex value is just 0i).

    One way to cope is to pre-allocate memory for the vector that is being
    left multiplied by the (block) Toeplitz matrix in this module.
    To do so, the module requires the batch_shape of the vector (e.g. how many
    vectors and in what shape they will be input).

    """
    def __init__(self, xgrids, kernel, batch_shape=None):
        super(ToeplitzMatmul, self).__init__()
        # create toeplitz gram representation
        self.device = xgrids[0].device
        self.dims = tuple(len(xg) for xg in xgrids)
        self.ndim = len(self.dims)
        self.M    = np.prod(self.dims)
        self.xgrids = xgrids
        self.K = self.toeplitz_gram(xgrids, kernel)
        self.C = self.circulant_embed(self.K)

        # Diagonal of the Circulant Embedding + sqrt
        Cc = self.make_complex(self.C)

        D0 = torch.fft(Cc, signal_ndim=self.ndim)
        D0 = D0[...,0].clamp(min=1e-6)
        D1 = torch.zeros_like(D0, dtype=D0.dtype, device=D0.device)
        self.D = torch.stack([D0, D1], dim=-1)
        self.D_sqrt = torch.stack([torch.sqrt(self.D[...,0]), D1], dim=-1)
        Di0 = 1. / self.D[...,0]
        self.Di = torch.stack([Di0, D1], dim=-1)

        """
        self.D = torch.fft(Cc, signal_ndim=self.ndim)
        self.D[...,1] = 0.
        self.D[...,0] = self.D[...,0].clamp(min=1e-6)  # for some eigenvalues become slightly negative
        #print("gram solve", self.D[...,0].min())
        self.D_sqrt = torch.sqrt(self.D)
        self.Di = self.D.clone()
        self.Di[...,0] = 1/self.D[...,0]
        """
        self.Di_sqrt = torch.sqrt(self.Di)

        # create padding slices for results --- batch dim (None),
        # variable dims, real dim (0)
        self.res_idx = [slice(None)] + \
                       [slice(0, d, 1) for d in self.dims] + \
                       [0]

        # pre-allocate complex part to avoid using big memory allocations
        self.Cc_shape = Cc.shape
        if batch_shape is not None:
            self.batch_shape = batch_shape
            self.cvec_shape = tuple(batch_shape) + Cc.shape
            #self.cvec = torch.zeros(tuple(batch_shape) + Cc.shape, dtype=self.K.dtype).to(self.device)
            #self.tmp  = torch.zeros_like(self.cvec, dtype=self.K.dtype).to(self.device)

    def set_batch_shape(self, batch_shape):
        self.batch_shape = batch_shape
        self.cvec_shape = tuple(batch_shape) + self.Cc_shape

    def _shift_axes_(self, a, steps=-1):
        """ shift axes number of steps """
        order = np.arange(len(a.shape))
        order = np.roll(order, steps)
        return a.permute(list(order))

    def forward(self, vec, multiply_type="gram"):
        """ matrix-vector multiply
            vec : bsz x M vector

        Let C^{1/2] = [[A, B]
                       [B', D]]
        Let R = [A, B] --> RR' = K
        1) multiply_type = gram: compute Kv

        2) multiply_type = RTv: compute R^T v
        cvec = embed and make complex v
        Fv = fft(v) -> prod = D_sqrt * Fv ->  cres = ifft(prod)
        return cres[..., 0] / return real part

        3) multiply_type = Rv: compute R v
        cvec = make complex v
        Fv = fft(v) -> prod = D_sqrt * Fv -> cres = ifft(prod)
        return cres[res_idx] / return cropped, real part

        4) multiply_type = circ_inv: compute C^{-1} v
        """
        # reshape each vector so that it is bsz x N1 x N2 ... x Nd
        bsz = vec.shape[0]

        if multiply_type == "Rv":
            expanded_dims = tuple(self.C.shape)
            cvec = self.make_complex(vec.reshape((vec.shape[0],) + expanded_dims))
        else:
            cvec = self.zero_pad_vec_batch_comp(vec)  # batch_shape + embeded_dims + (2, )

        # if Kv = (Ft D F)v, do the Fv at the tail end. multiply by D
        Fv = torch.fft(cvec, signal_ndim=self.ndim)
        if multiply_type == "gram":
            prod = self.complex_mult(self.D, Fv)
        elif multiply_type == "RTv":  # compute R^T v
            prod = self.complex_mult(self.D_sqrt, Fv)
        elif multiply_type == "Rv":
            prod = self.complex_mult(self.D_sqrt, Fv)
        elif multiply_type == "circ_inv":
            prod = self.complex_mult(self.Di, Fv)
        else:
            raise NotImplementedError("gram|RTv|Rv|circ_inv")
        # prod shape is batch_shape + embeded_dims + (2, )

        # if Kv = Ft (DFv), do the inverse FFT
        cres = torch.ifft(prod, signal_ndim=self.ndim)  # (batch_size, dims, 2)

        if multiply_type == "RTv":
            return cres[..., 0].view(bsz, -1)
            # remove all the padding
        return cres[self.res_idx].reshape(bsz, -1)

    def circulant_embed(self, Ktoe):
        dims = Ktoe.shape
        for d in range(len(dims)):
            Krev = torch.flip(Ktoe, dims=(d,))
            idx = [slice(None)]*d + [slice(1, -1, 1)]
            Krev = Krev[idx]
            # concat onto correct dimension
            Ktoe = torch.cat([Ktoe, Krev], dim=d)
        return Ktoe

    def zero_pad_vec_batch_comp(self, vec):
        cvec = torch.zeros(self.cvec_shape, dtype=self.K.dtype, device=self.device)
        cvec[self.res_idx] = vec.reshape((vec.shape[0],) + self.dims)
        return cvec

    def zero_pad_vec_batch(self, vec):
        """ pads a batch of vectors to be multiplied
            - vec: bsz x M tensor (where M = N1 * N2 * ... * Nd)

        Returns:
            - padded vec: bsz x (2N1-2, 2N2-2, ..., 2Nd-2)

        """
        # reshape into bsz x N1 x N2 x ... x Nd
        vec = vec.view((vec.shape[0],) + self.dims)
        dims = vec.shape
        for d in range(1, len(dims)):
            sz    = list(vec.shape)
            sz[d] = sz[d]-2
            pad   = torch.zeros(sz, dtype=vec.dtype, device=self.device)
            vec = torch.cat([vec, pad], dim=d)

        return vec

    def make_complex(self, vec):
        """add a zero compelx part to vec in last dimension"""
        return torch.stack([vec, torch.zeros_like(vec)], dim=-1)

    def complex_mult(self, t1, t2):
        real1, imag1 = t1[...,0], t1[...,1]
        real2, imag2 = t2[...,0], t2[...,1]
        if self.batch_shape is None:
            return torch.stack([real1 * real2 - imag1 * imag2,
                                real1 * imag2 + imag1 * real2], dim = -1)
        else:
            tmp = torch.zeros(self.cvec_shape, dtype=self.K.dtype, device=self.device)
            #tmp = self.tmp.zero_()
            tmp[...,0] = real1*real2 - imag1*imag2
            tmp[...,1] = real1*imag2 + imag1*real2
            return tmp

    def toeplitz_gram(self, xgrids, kernel):
        dims = [len(xg) for xg in xgrids]
        xxs  = torch.meshgrid(*xgrids)
        xs   = torch.stack([x.reshape(-1) for x in xxs], dim=-1)  # (M, ndim)
        Krow = kernel(xs[0][None,:], xs)  # (1, ndim) by (M, ndim)  -> (1, M)
        # Note: here is where we would add a nugget --- is that a good idea?
        # Krow[0,0] += 1e-4 # nugget
        Ktoe = Krow.view(dims)
        return Ktoe


#def blk_toeplitz_solve(xinduce_grids, kernel_fun, vec, sqrt_solve=False):
#    """
#    Matrix multiplication for a Toeplitz Block Block Toeplitz matrix
#    (or higher levels of recursion) in O(n log n) time
#        Works by circulant embedding and FFT
#    """
#    # size of the mesh along each dimension, total size of the problem
#    dims = [len(x) for x in xinduce_grids]
#    Ntotal = np.prod(dims)
#
#    # toeplitz representation of the gram matrix (see below)
#    Ktoe = toeplitz_gram(xinduce_grids, kernel_fun)
#
#    # now embed the block toeplitz representation in a block circulant
#    # matrix representaion.
#    Ctoe = circulant_embed(Ktoe)
#
#    # similarly embed the vector
#    vmat = np.reshape(vec, Ktoe.shape, order='C')
#    cvec = circulant_embed(vmat, zero_padding=True)
#
#    # now apply N-d FFT, invert, chop, and return
#    res = ifftn(fftn(Ctoe) * fftn(cvec))
#
#    # subselect dims
#    idx = [slice(0, d, 1) for d in dims]
#    res = np.real(res[idx])
#
#    # chop out the appropriate bits?
#    return np.reshape(res, (-1,), order='C')
#
#
#
#def solve_circulant_cg(C, vec, sqrt_solve=False):
#    pass
#
#
#def circ_mvm(C, vec):
#    """ circ ... """
#
#    res = ifftn(fftn(Ctoe) * fftn(cvec))
#
#    # subselect dims
#    idx = [slice(0, d, 1) for d in dims]
#    res = np.real(res[idx])
#
#
#def sqrt_circ_mvm(C, vec):
#    pass
#
#
#def solve_circulant_cg(C, vec): #xinduce_grids, kernel_fun, vec):
#    pass
#
#
#
#
#
##############################################
## Numpy implementation for debugging        #
##############################################
#import numpy as np
#from numpy.fft import fft, ifft, fftn, ifftn
#
#
#def blk_toeplitz_mvm(xinduce_grids, kernel_fun, vec):
#    """
#    Matrix multiplication for a Toeplitz Block Block Toeplitz matrix
#    (or higher levels of recursion) in O(n log n) time
#
#    Works by circulant embedding and FFT
#    """
#    # size of the mesh along each dimension, total size of the problem
#    dims = [len(x) for x in xinduce_grids]
#    #Ntotal = np.prod(dims)
#
#    # toeplitz representation of the gram matrix (see below)
#    Ktoe = toeplitz_gram(xinduce_grids, kernel_fun)
#
#    # now embed the block toeplitz representation in a block circulant
#    # matrix representaion.
#    Ctoe = circulant_embed(Ktoe)
#
#    # similarly embed the vector
#    vmat = np.reshape(vec, Ktoe.shape, order='C')
#    cvec = circulant_embed(vmat, zero_padding=True)
#
#    # now apply N-d FFT, invert, chop, and return
#    res = ifftn(fftn(Ctoe) * fftn(cvec))
#
#    # subselect dims
#    idx = [slice(0, d, 1) for d in dims]
#    res = np.real(res[idx])
#
#    # chop out the appropriate bits?
#    return np.reshape(res, (-1,), order='C')
#
#
#def sqrt_mvm(xinduce_grids, kernel_fun, vec):
#    """ Left multiplies vector by sqrt
#
#        S = Ft D F v = ifft( fft(c) * fft(v) )
#
#    Where D = (1/N) diag(Fs)
#
#    so if S = RR^t, then R is a matrix
#
#    we construct the square root of the covariance matrix VIA diagonalization
#    https://en.wikipedia.org/wiki/Square_root_of_a_matrix
#
#    """
#    dims = [len(x) for x in xinduce_grids]
#    Ntotal = np.prod(dims)
#
#    # toeplitz representation of the gram matrix (see below)
#    Ktoe = toeplitz_gram(xinduce_grids, kernel_fun)
#
#    # now embed the block toeplitz representation in a block circulant
#    # matrix representaion.
#    Ctoe = circulant_embed(Ktoe)
#
#    # similarly embed the vector
#    vmat = np.reshape(vec, Ktoe.shape, order='C')
#    cvec = circulant_embed(vmat, zero_padding=True)
#    res = ifftn(np.sqrt(fftn(Ctoe)) * fftn(cvec))
#
#    # circulant matrix has diagonalization of ....
#    #   C = F D F^t where D = diag(Fc) where s is the circulant embedding
#    #
#    # So the sqrt of S can be written
#    #
#    #    Crt = (F D^{1/2})(F D^{1/2})^T  where R = FD^{1/2}
#    #
#    # So a left multiply of the sqrt of S can be done
#    #    R^t v = (Fc)^{1/2} F^t v
#    #          = fft(c)^{1/2} * ifft(v)
#    #stilde = fftn(Ctoe)
#    #lam_sqrt = np.sqrt(stilde)# / Ntotal)
#    #lam_sqrt_vec = lam_sqrt*cvec
#    #res = ifftn(lam_sqrt_vec)
#
#    # subselect dims
#    idx = [slice(0, d, 1) for d in dims]
#    rres = np.real(res[idx])
#    return np.reshape(rres, (-1,), order='C')
#
#
#def circulant_embed(Ktoe, zero_padding=False):
#    """ Embeds a representation of a block-toeplitz matrix into a block
#    circulant matrix
#
#        - zero_padding: if true, just pad with zeros (for the vector being multiplied)
#            --- doesn't actually result in a circulant matrix
#
#    """
#    # TODO add padding option here
#    dims = Ktoe.shape
#    for dim, size in enumerate(dims):
#        if zero_padding:
#            Ksize = list(Ktoe.shape)
#            Ksize[dim] = Ksize[dim]-1
#            Krev = np.zeros(Ksize)
#        else:
#            # reverse specific dim, chopping off the first element
#            idx = [slice(None)]*dim + [slice(-1, 0, -1)]
#            Krev = Ktoe[idx]
#
#        Ktoe = np.concatenate([Ktoe, Krev], axis=dim)
#
#    return Ktoe
#
#
#def toeplitz_gram(xinduce_grids, kernel_fun):
#    """ creates a minimal toeplitz representation of the symmetric gram
#    matrix that results from applying a stationary kernel_fun to a N-d mesh
#    of evenly spaced points (along each dimension)
#
#    Args:
#        - xinduce_grids
#        - kernel_fun
#
#    Returns:
#        - Ktoe: array of size (Nx x Ny x Nz x ...) that is a minimal
#            representation of the block toeplitz Gram Matrix
#
#    For a toeplitz (or block toeplitz) gram matrix, the first row
#    characterizes the whole covariance matrix.  compute this using the
#    covariance kernel_fun.
#    Reshape the top covariance row so that the first dimension corresponds
#    to the delta-x along the first grid (the x grid).
#    e.g. in two dimension, we have Ktoe = (Nx, Ny)-size array where #
#    Nx is the number of points along the xgrid (Ny similar).
#    Because we traverse along the first dimension (x) first, and then the
#    second dimension (y) and so on, a row of Ktoe will characterize
#    a toeplitz block within the covariance, and the column ordering reflects
#    the overall block toeplitz structure (that is, the blocks are laid out
#    along constant block-diagonals, and each block itself is toeplitz. fml).
#    """
#    dims = [len(xg) for xg in xinduce_grids]
#    xxs = np.meshgrid(*xinduce_grids, indexing='ij')
#    xs = np.column_stack([x.flatten(order='C') for x in xxs])
#    Krow = kernel_fun(xs[0][None,:], xs)
#    Ktoe = np.reshape(Krow, dims, order='C')
#    return Ktoe
#
