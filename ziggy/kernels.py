"""
Integrated kernels as pytorch modules
"""
import torch
from torch import nn
import numpy as np
from ziggy.misc.stats import normal_cdf

# TODO: make sure backprop through the integrated kernel makes sense
# TODO: for now we only assume ell is a scalar for rest of files. However in this file, ell has been adapted to 1) a scalar 2) a tensor of shape (D, )
class Kernel(nn.Module):
    """ base pytorch kernel_fun class """
    def __init__(self):
        super(Kernel, self).__init__()

    def k_semi(self, xpoint, xintegrated, params):
        raise NotImplementedError

    def k_semi_mc(self, xpoint, xintegrated, params, npts=5):
        """ monte carlo approximation of the Semi Integrated Kernel """
        Np, D = xpoint.shape
        Ni, D = xintegrated.shape

        # create random grid
        delta  = 1./npts
        alphas = torch.arange(npts, dtype=self.dtype, device=xpoint.device)/npts + \
                 torch.rand(1, dtype=self.dtype, device=xpoint.device)*delta

        # create points between O and xintegrated along the random grid
        # creates Ni x npts x 2 tensor
        xgrid = xintegrated[:,None,:] * alphas[None,:,None]

        # stacked Kpi matrix
        Kpis = self.forward(xpoint, xgrid.reshape(-1, D), params=params)
        Kpis = Kpis.reshape(Np, Ni, npts)

        # average sample, multiply by distance
        dists = xintegrated.pow(2.).sum(dim=-1).sqrt()
        return torch.mean(Kpis, dim=-1)*dists[None,:]

    def k_semi_num(self, xpoint, xintegrated, params):
        """ numeric version of k semi """
        #from ziggy.kernels import semi_integrated_kernel
        def kfun(xp, xi):
            return self.forward(torch.Tensor(xp).to(torch.double),
                                torch.Tensor(xi).to(torch.double),
                                params=params).detach().numpy()
        Kpi = semi_integrated_kernel(
            xpoint.numpy(), xintegrated.numpy(), kfun)
        return Kpi

    def k_doubly_diag_num(self, x, params):
        print("numerically integrating diagonal terms!")
        # numpy callable of kernel_fun
        #from ziggy.kernels import doubly_integrated_diag
        def kfun(x, y):
            return self.forward(torch.Tensor(x).to(self.dtype),
                                torch.Tensor(y).to(self.dtype),
                                params=params).detach().numpy()
        knn = doubly_integrated_diag(x.numpy(), kfun)
        return torch.Tensor(knn).to(self.dtype)


class SqExp(Kernel):
    """ squared exponential kernel_fun (implements analytic SQ Exp """
    def __init__(self, dtype=torch.double, Ndiag=50, dmax=5):
        super(SqExp, self).__init__()
        self.dtype = dtype
        self.diag_interp = \
            KernelDoublyDiagInterpolator(self, N=Ndiag, dmax=dmax)
        self.has_k_semi = True

    def forward(self, x, y, params):
        assert x.shape[-1] == y.shape[-1], \
            "last dimension should match, but got x.shape = {}, y.shape = {}".format(x.shape, y.shape)
        assert x.ndimension() == 2 and y.ndimension() == 2, "x shape = {}, y shape = {}".format(x.shape, y.shape)
        sig2, ell = params # ell should be a scalar, or a tensor that matches the dimension of data
        sqdist = torch.sum(((x[:,None,:] - y[None,:,:])/ell)**2, dim=-1)
        return sig2*torch.exp(-sqdist / 2)

    def diag(self, x, params):
        sig2, ell = params
        return sig2*torch.ones(x.shape[0], dtype=self.dtype, device=x.device)

    def k_semi(self, xpoint, xintegrated, params):
        sig2, ell = params # ell is either a scalar, of a tensor of shape (D, )
        Npoint, D = xpoint.shape
        Sinv = (1./(ell**2))*torch.eye(D, dtype=self.dtype, device=xpoint.device)
        Kip = semi_integrated_sqe(xintegrated, xpoint, sig2, Sinv)
        return Kip.transpose(0,1)

    def k_doubly_diag(self, x, params):
        return self.diag_interp(x, params)


class Gneiting(Kernel):
    def __init__(self, alpha=1., length_scale=1., dtype=torch.double,
                 Ndiag=50, dmax=5.):
        super(Gneiting, self).__init__()
        self.dtype = dtype
        self.alpha = alpha
        self.length_scale = length_scale
        self.anisotropic = False
        self.diag_interp = \
            KernelDoublyDiagInterpolator(self, N=Ndiag, dmax=dmax)
        self.has_k_semi = False

    def forward(self, x, y, params):
        assert x.shape[-1] == y.shape[-1], \
            "last dimension should match, but got x.shape = {}, y.shape = {}".format(x.shape, y.shape)
        sig2, ell = params
        dist = torch.sqrt(torch.sum(((x[:,None,:]-y[None,:,:])/ell)**2, dim=-1))
        t = dist
        cterms = (1-t)*torch.cos(np.pi*t) + (1/np.pi)*torch.sin(np.pi*t)
        cij = (1+t**self.alpha)**(-3) * cterms
        cij[t>1.] = 0.
        return sig2*cij

    def diag(self, x, params):
        sig2, ell = params
        t = torch.zeros(x.shape[0], dtype=self.dtype, device=x.device)
        cterms = torch.ones(x.shape[0], dtype=self.dtype, device=x.device)
        cij = (1)**(-3)*cterms
        cij[t>1.] = 0.
        return sig2*cij

    def k_doubly_diag(self, x, params):
        return self.diag_interp(x, params)


class Matern(Kernel):
    def __init__(self, nu=0.5, length_scale=1., dtype=torch.double,
                 Ndiag=50, dmax=5.):
        super(Matern, self).__init__()
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        self.nu = nu
        self.dtype = dtype
        self.length_scale = length_scale
        self.anisotropic = False
        self.diag_interp = \
            KernelDoublyDiagInterpolator(self, N=Ndiag, dmax=dmax)
        self.has_k_semi = False

    def forward(self, x, y, params):
        assert x.shape[-1] == y.shape[-1], \
            "last dimension should match, but got x.shape = {}, y.shape = {}".format(x.shape, y.shape)
        sig2, ell = params # ell is either a scalar or a tensor of shape (D, )
        sqdist = torch.sum((x[:,None,:]-y[None,:,:])**2, dim=-1) # / (ell*ell)
        if self.nu == .5:
            kmat = torch.exp(-torch.sqrt(sqdist)/ell)
        elif self.nu == 1.5:
            dp = np.sqrt(3)*torch.sqrt(sqdist)/ell
            kmat = (1+dp)*torch.exp(-dp)
        elif self.nu == 2.5:
            dp = np.sqrt(5)*torch.sqrt(sqdist)/ell
            kmat = (1+dp + (5./3.)*sqdist/(ell**2))*torch.exp(-dp)
        return sig2*kmat

    def diag(self, x, params):
        sig2, ell = params
        return sig2*x.new_ones(x.shape[0]) #, dtype=self.dtype)

    def k_doubly_diag(self, x, params):
        return self.diag_interp(x, params)


class KernelDoublyDiagInterpolator(nn.Module):
    """ general module for linear interpolation of doubly integrated
    diagonal term """
    def __init__(self, kernel, N=50, dmax=5, dtype=None):
        super(KernelDoublyDiagInterpolator, self).__init__()
        if dtype is None:
            dtype = kernel.dtype

        # numpy callable of kernel_fun
        def kfun(x, y):
            return kernel(torch.Tensor(x).to(dtype),
                          torch.Tensor(y).to(dtype),
                          params=(1., 1.))

        # create distance grid
        dgrid  = np.linspace(0, dmax, N)
        ddelta = dgrid[1] - dgrid[0]

        # numerically compute "true" values
        xs = np.column_stack([dgrid, np.zeros(N)])
        knn = doubly_integrated_diag(xs, kfun)

        # linear interpolation as matrix multiplication
        slopes = (knn[1:] - knn[:-1]) / (dgrid[1:] - dgrid[:-1])
        slopes = np.concatenate([ slopes, [slopes[-1]] ])

        # store tensors used in forward
        self.distance_grid = torch.Tensor(dgrid).to(dtype)
        self.slopes = torch.Tensor(slopes).to(dtype)
        self.knn    = torch.Tensor(knn).to(dtype)

    def forward(self, x, params):
        # unpack marginal variance and lengthscale parameters
        sig2, ell = params  # ell is either a scalar or a tensor of shape (D, )

        # make sure everything is on the correct device
        distance_grid = self.distance_grid.to(device=x.device)
        slopes = self.slopes.to(device=x.device)
        knn = self.knn.to(device=x.device)

        # convert inputs into distances from the origin
        # dists = torch.sqrt(torch.sum(x ** 2, dim=-1)) / ell
        dists = torch.sqrt(torch.sum((x/ell)**2, dim=-1))

        # compute index of lower bound
        lower_i = torch.sum(dists[:,None] > distance_grid, dim=-1) - 1

        # interpolate
        diff = dists - distance_grid[lower_i]
        ivals = knn[lower_i] + slopes[lower_i]*diff
        return ell*ell*sig2*ivals


sqrt2pi = np.sqrt(2*np.pi)

def semi_integrated_sqe(xintegrated, x, sig2, Sinv):
    """ integrates over FIRST argument """
    xdists = torch.sqrt(torch.sum(xintegrated*xintegrated, dim=-1))
    a = torch.sum(torch.matmul(xintegrated, Sinv)*xintegrated, dim=-1)
    xint_Si = torch.matmul(xintegrated, Sinv)
    b = torch.matmul(xint_Si[:,None,None,:], x[None,:,:,None]).squeeze() # num_xi x num_x
    c = torch.sum(torch.matmul(x, Sinv)*x, dim=-1)

    # compute loc and scale for CDFs
    scale = torch.sqrt(1/a[:,None])
    loc   = b / a[:,None]
    coef = sig2*torch.exp((b**2)/(2*a[:,None]) - c/2) * sqrt2pi * scale
    ca = normal_cdf(1, loc, scale)
    cb = normal_cdf(0, loc, scale)
    return coef * (ca-cb) * xdists[:,None]


#####################
# Numeric methods   #
#####################
import numpy as np
from sklearn.gaussian_process import kernels
from scipy.stats import norm
from scipy import integrate, interpolate
import pyprind
from itertools import product

def semi_integrated_kernel(xpoint, xint, kern):
    Npoint, D = xpoint.shape
    Nint, D   = xint.shape
    Kpi = np.zeros((Npoint, Nint))
    for p, xp in enumerate(xpoint):
        for i, xi in enumerate(xint):
            di = np.sqrt(np.sum(xi**2))
            def rayfun(alpha):
                return kern(xp[None,:], (1-alpha)*xi[None,:])*di

            res = integrate.quad(rayfun, a=0, b=1)
            Kpi[p,i] = res[0]

    return Kpi


def doubly_integrated_diag(x, kern, return_errors=False):
    N, D = x.shape
    knn  = np.zeros(N)
    errs = np.zeros(N)

    gen = pyprind.prog_bar(range(N)) if N > 200 else range(N)
    for n in gen:
        xn = x[n,:]
        xn_dist = np.sqrt(np.sum(xn**2))
        def rayfun(alpha, alpha_p):
            return kern(alpha*xn[None,:], alpha_p*xn[None,:])

        res = integrate.dblquad(rayfun, a=0, b=1,
                                gfun=lambda a: 0, hfun=lambda b: 1,
                                epsrel=1.49e-5, epsabs=1.49e-1)
                                ##epsabs=1.49e-5)
        knn[n] = res[0]*(xn_dist*xn_dist)
        errs[n] = res[1]

    if return_errors:
        return knn, errs
    return knn
