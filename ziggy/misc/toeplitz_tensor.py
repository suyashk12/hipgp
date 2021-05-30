import torch
import numpy as np
from ziggy.misc._inv_matmul import InvMatmul
from ziggy.misc.cg import conj_grad2


class ToeplitzTensor:
    # TODO: set batch_shape
    def __init__(self, xgrids, kernel, batch_shape=None, jitter_val=1e-3):
        # suppose we have formed the column

        self.column = self.toeplitz_gram(xgrids, kernel, jitter_val)

        self.device = xgrids[0].device
        self.dims = tuple(len(xg) for xg in xgrids)
        self.ndim = len(self.dims)

        self.M = np.prod(self.dims)

        self.C = self.circulant_embed(self.column.view(self.dims))

        # Diagonal of the Circulant Embedding + sqrt
        Cc = self.make_complex(self.C)

        D0 = torch.fft(Cc, signal_ndim=self.ndim)
        D0 = D0[..., 0].clamp(min=1e-6)
        D1 = torch.zeros_like(D0, dtype=D0.dtype, device=D0.device)
        self.D = torch.stack([D0, D1], dim=-1)
        self.D_sqrt = torch.stack([torch.sqrt(self.D[..., 0]), D1], dim=-1)
        Di0 = 1. / self.D[..., 0]
        self.Di = torch.stack([Di0, D1], dim=-1)

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

    def inv_matmul(self, right_tensor, do_precond=True, maxiter=20, tol=1e-8):
        """compute A^{-1}R, where self = A"""

        func = InvMatmul

        return func.apply(self, self.column, right_tensor, do_precond, maxiter, tol)

    def _solve(self, vec, do_precond=True, maxiter=100, tol=1e-8, callback=None):
        """vec: (bsz, M)
        return: d -- (bsz, M)
        """
        # compute K^{-1}v using CG
        # use matmul_by_Cinv as preconditioner
        assert len(vec.shape) == 2
        self.set_batch_shape(vec.shape[:-1])
        precond = None
        if do_precond:
            precond = self._matmul_by_Cinv

        d = conj_grad2(self._matmul_by_K, vec, precond=precond, maxiter=maxiter, tol=tol, callback=callback) # (bsz, M)

        return d

    def _matmul_by_K(self, vec):
        """Compute Kv
        C = [K, Ktilde;
            Ktilde.T, K']
        C [v;0]^T = [Kv; Ktilde.T v]^T
        """
        bsz, M = vec.shape

        cvec = self.zero_pad_vec_batch_comp(vec)  # batch_shape + embeded_dims + (2, )
        Fv = torch.fft(cvec, signal_ndim=self.ndim)

        prod = self.complex_mult(self.D, Fv)
        cres = torch.ifft(prod, signal_ndim=self.ndim)  # (batch_size, dims, 2)
        return cres[self.res_idx].reshape(bsz, -1)

    def _matmul_by_RT(self, vec):
        """Compute R.T v = [Av; B.T v], where R = [A, B]
        C^{1/2} = [A   B;
                   B.T D]
        C^{1/2} [v; 0]^T = [Av; B.T v]^T
        """
        bsz, M = vec.shape

        cvec = self.zero_pad_vec_batch_comp(vec)  # batch_shape + embeded_dims + (2, )
        Fv = torch.fft(cvec, signal_ndim=self.ndim)
        prod = self.complex_mult(self.D_sqrt, Fv)
        cres = torch.ifft(prod, signal_ndim=self.ndim)  # (batch_size, dims, 2)
        return cres[..., 0].view(bsz, -1)

    def _matmul_by_R(self, vec):
        """Compute Rv = [ A, B] v
        C^{1/2} = [A   B;
                   B.T D]
        C^{1/2} v = [Rv; ...]
        """
        bsz = vec.shape[0]

        expanded_dims = tuple(self.C.shape)
        cvec = self.make_complex(vec.reshape((vec.shape[0],) + expanded_dims))
        Fv = torch.fft(cvec, signal_ndim=self.ndim)
        prod = self.complex_mult(self.D_sqrt, Fv)
        cres = torch.ifft(prod, signal_ndim=self.ndim)  # (batch_size, dims, 2)
        return cres[self.res_idx].reshape(bsz, -1)

    def _matmul_by_Cinv(self, vec):
        """Compute C.inv.upperleft v
        C.inv = [C.inv.uppperleft, C.inv.upperight;
                C.inv.lowerleft,   C.inv.lowerright]
        """
        bsz, M = vec.shape

        cvec = self.zero_pad_vec_batch_comp(vec)  # batch_shape + embeded_dims + (2, )
        Fv = torch.fft(cvec, signal_ndim=self.ndim)
        prod = self.complex_mult(self.Di, Fv)
        cres = torch.ifft(prod, signal_ndim=self.ndim)  # (batch_size, dims, 2)
        return cres[self.res_idx].reshape(bsz, -1)

    def toeplitz_gram(self, xgrids, kernel, jitter_val):
        xxs  = torch.meshgrid(*xgrids)
        xs   = torch.stack([x.reshape(-1) for x in xxs], dim=-1)  # (M, ndim)
        Krow = kernel(xs[0][None,:], xs)  # (1, ndim) by (M, ndim)  -> (1, M)
        # Note: here is where we would add a nugget --- is that a good idea?
        Krow[0,0] += jitter_val # nugget
        return Krow.squeeze()

    def circulant_embed(self, Ktoe):
        dims = Ktoe.shape
        for d in range(len(dims)):
            Krev = torch.flip(Ktoe, dims=(d,))
            idx = [slice(None)]*d + [slice(1, -1, 1)]
            Krev = Krev[idx]
            # concat onto correct dimension
            Ktoe = torch.cat([Ktoe, Krev], dim=d)
        return Ktoe

    def make_complex(self, vec):
        """add a zero compelx part to vec in last dimension"""
        return torch.stack([vec, torch.zeros_like(vec)], dim=-1)

    def zero_pad_vec_batch_comp(self, vec):
        cvec = torch.zeros(self.cvec_shape, dtype=self.column.dtype, device=self.device)
        cvec[self.res_idx] = vec.reshape((vec.shape[0],) + self.dims)
        return cvec

    def complex_mult(self, t1, t2):
        real1, imag1 = t1[...,0], t1[...,1]
        real2, imag2 = t2[...,0], t2[...,1]
        if self.batch_shape is None:
            return torch.stack([real1 * real2 - imag1 * imag2,
                                real1 * imag2 + imag1 * real2], dim = -1)
        else:
            tmp = torch.zeros(self.cvec_shape, dtype=self.column.dtype, device=self.device)
            #tmp = self.tmp.zero_()
            tmp[...,0] = real1*real2 - imag1*imag2
            tmp[...,1] = real1*imag2 + imag1*real2
            return tmp

    def set_batch_shape(self, batch_shape):
        self.batch_shape = batch_shape
        self.cvec_shape = tuple(batch_shape) + self.Cc_shape
