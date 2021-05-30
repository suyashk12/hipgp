import torch
import numpy as np
import ziggy.misc.util as zutil
from ziggy.misc.toeplitz_tensor import ToeplitzTensor
from ziggy.misc import stats
from ziggy.kernels import SqExp


def k(x, y, sig2, ell):
    diff = x[:, None] - y[None,]
    Kxy = sig2 * torch.exp(-1 / 2 * diff ** 2 / ell ** 2)
    return Kxy


def k_1d(x, sig2, ell):
    return sig2


def kprime(x, y, sig2, ell):
    diff = x[:, None] - y[None,]
    Kxy = sig2 * torch.exp(-1 / 2 * diff ** 2 / ell ** 2)
    Kprime_xy = -diff / (ell ** 2) * Kxy
    return Kprime_xy


def kprime_double_1d(x, sig2, ell):
    ell_sq = ell ** 2
    Kprime_double_xy = sig2 / ell_sq
    return Kprime_double_xy


def kprime_double_full(x, y, sig2, ell):
    diff = x[:, None] - y[None,]
    diff_sq = diff ** 2
    ell_sq = ell ** 2
    Kxy = sig2 * torch.exp(-1 / 2 * diff_sq / ell_sq)
    Kprime_double_xy = Kxy / ell_sq * (1 - 1 / ell_sq * diff_sq)
    return Kprime_double_xy


def prediction(xprime, yprime, x, sig2, ell, jitter=1e-4):
    xprime = xprime.squeeze()
    x = x.squeeze()

    K_xprime_xprime = kprime_double_full(xprime, xprime, sig2, ell)
    K_xprime_x = kprime(xprime, x, sig2, ell)
    Kxx = k(x, x, sig2, ell)

    n = x.shape[0]
    I = torch.eye(n, dtype=Kxx.dtype, device=Kxx.device)
    predict_mu, _ = torch.solve(yprime, K_xprime_xprime + jitter * I)
    predict_mu = torch.matmul(K_xprime_x.t(), predict_mu)

    predict_cov, _ = torch.solve(K_xprime_x, K_xprime_xprime + jitter * I)
    predict_cov = torch.matmul(K_xprime_x.t(), predict_cov)
    predict_cov = Kxx - predict_cov
    return predict_mu, predict_cov


def prediction_for_prime(x, y, xprime, sig2, ell):
    xprime = xprime.squeeze()
    x = x.squeeze()

    K_xprime_xprime = kprime_double_full(xprime, xprime, sig2, ell)
    K_xprime_x = kprime(xprime, x, sig2, ell)
    Kxx = k(x, x, sig2, ell)

    predict_mu, _ = torch.solve(y, Kxx)
    predict_mu = torch.matmul(K_xprime_x, predict_mu)

    predict_cov, _ = torch.solve(K_xprime_x.t(), Kxx)
    predict_cov = torch.matmul(K_xprime_x, predict_cov)
    predict_cov = K_xprime_xprime - predict_cov
    return predict_mu, predict_cov


def svgp_batch_solve(u, xprime, yprime, x, y, sig2, ell, derivative_obs_noise_std, obs_noise_std, batch_size=-1,
                     whitened_type='ziggy', maxiter=20, precond=True, tol=1e-8):
    """compute the inducing point values for u"""

    M = u.shape[0]  # u is (m, )

    if whitened_type == 'cholesky':
        I = torch.eye(M, dtype=u.dtype, device=u.device)
        Kuu = k(u, u, sig2, ell)
        cKuu = torch.cholesky(Kuu + I * 1e-4, upper=False)
        big_lam = I
    else:
        # use Toeplitz tensor
        xgrids = [u]

        kern = SqExp()
        kernel = lambda x, y: kern(x, y, (sig2, ell))
        tt = ToeplitzTensor(xgrids=xgrids, kernel=kernel, batch_shape=None)
        big_lam = torch.eye(2*M-2, dtype=u.dtype, device=u.device)

    b = 0
    if xprime is not None:
        nprime = xprime.shape[0]
        if batch_size == -1:
            batch_size_prime = nprime
        else: batch_size_prime = batch_size

        num_batches = int(np.ceil(nprime / batch_size_prime))
        batches = [zutil.batch_indices(i, num_batches, batch_size_prime, nprime) for i in range(num_batches)]

        for bi in batches:
            xbatch = xprime[bi].to(u.device)
            ybatch = yprime[bi].to(u.device)

            Knm = kprime(xbatch, u, sig2, ell)
            if whitened_type == 'cholesky':
                kn = torch.triangular_solve(Knm.t(), cKuu, upper=False)[0]
                kn = kn.squeeze()  # (m, n1)
            else:
                d0 = tt.inv_matmul(Knm, do_precond=precond, maxiter=maxiter, tol=tol)  # (n1, m)
                kn = tt._matmul_by_RT(d0)  # (n1, m')
                kn = kn.t()
            ivar_noise = 1./ derivative_obs_noise_std**2
            knknt = (ivar_noise * kn).matmul(kn.t())  # (m,m)
            b_prime = torch.sum(ivar_noise * ybatch.squeeze() * kn, dim=-1)
            b = b + b_prime
            big_lam = big_lam + knknt

    if x is not None:
        nlatent = x.shape[0]
        if batch_size == -1:
            batch_size_latent = nlatent
        else:
            batch_size_latent = batch_size

        num_batches = int(np.ceil(nlatent / batch_size_latent))
        batches = [zutil.batch_indices(i, num_batches, batch_size_latent, nlatent) for i in range(num_batches)]

        for bi in batches:
            xbatch = x[bi].to(u.device)
            ybatch = y[bi].to(u.device)

            Knm = k(xbatch, u, sig2, ell)
            if whitened_type == 'cholesky':
                kn = torch.triangular_solve(Knm.t(), cKuu, upper=False)[0]
                kn = kn.squeeze()  # (m, n2)
            else:
                d0 = tt.inv_matmul(Knm, do_precond=precond, maxiter=maxiter, tol=tol)  # (n1, m)
                kn = tt._matmul_by_RT(d0)  # (n1, m')
                kn = kn.t()
            ivar_noise = 1./ obs_noise_std**2
            knknt = (ivar_noise * kn).matmul(kn.t())  # (m,m)
            b_latent = torch.sum(ivar_noise * ybatch.squeeze() * kn, dim=-1)
            b = b + b_latent
            big_lam = big_lam + knknt

    S = torch.inverse(big_lam)
    m = torch.matmul(S, b)
    return m, S


def posterior_prediction(x, u, m, S, sig2, ell, domain='latent', batch_size=-1, whitened_type='ziggy',
                         maxiter=20, precond=True, tol=1e-8):

    if whitened_type == 'cholesky':
        Kuu = k(u, u, sig2, ell)
        I = torch.eye(Kuu.shape[0], dtype=Kuu.dtype, device=Kuu.device)
        cKuu = torch.cholesky(Kuu + I * 1e-4, upper=False)
    else:
        # use Toeplitz tensor
        xgrids = [u]
        kern = SqExp()
        kernel = lambda x, y: kern(x, y, (sig2, ell))
        tt = ToeplitzTensor(xgrids=xgrids, kernel=kernel, batch_shape=None)

    nobs = x.shape[0]
    if batch_size == -1:
        batch_size = nobs
    num_batches = int(np.ceil(nobs / batch_size))
    batches = [zutil.batch_indices(i, num_batches, batch_size, nobs) for i in range(num_batches)]

    predict_mu = torch.empty(nobs, dtype=u.dtype)
    predict_sig2 = torch.empty(nobs, dtype=u.dtype)

    for bi in batches:
        xbatch = x[bi].to(u.device)

        if domain == 'latent':
            Knm = k(xbatch, u, sig2, ell)
            Knn = k_1d(xbatch, sig2, ell)
        else:
            Knm = kprime(xbatch, u, sig2, ell)
            Knn = kprime_double_1d(xbatch, sig2, ell)

        if whitened_type == 'cholesky':
            kn = torch.triangular_solve(Knm.t(), cKuu, upper=False)[0]
            kn = kn.squeeze()  # (m, n)
        else:
            d0 = tt.inv_matmul(Knm, do_precond=precond, maxiter=maxiter, tol=tol)  # (n1, m)
            kn = tt._matmul_by_RT(d0)  # (n1, m')
            kn = kn.t()

        predict_mu_batch = torch.matmul(kn.t(), m)
        predict_mu[bi] = predict_mu_batch

        kntkn = torch.sum(kn*kn, dim=0)
        kntSkn = torch.sum(kn * torch.matmul(S, kn), dim=0)
        predict_sig2_batch = Knn - kntkn + kntSkn
        predict_sig2[bi] = predict_sig2_batch
    return predict_mu, predict_sig2


def exact_gp_prediction(xtest, xprime, yprime, xlatent, ylatent, sig2, ell, derivative_obs_noise_std, obs_noise_std, batch_size=-1):

    # put up a big K matrix
    device = xtest.device
    nprime = 0 if xprime is None else xprime.shape[0]
    nlatent = 0 if xlatent is None else xlatent.shape[0]
    ntotal = nprime + nlatent
    ytotal = torch.empty(ntotal, dtype=xtest.dtype, device=device)
    K = torch.empty(ntotal, ntotal, dtype=xtest.dtype, device=device)

    if nprime > 0:
        nprime = xprime.shape[0]
        I = torch.eye(nprime, dtype=xtest.dtype, device=device)
        K[:nprime, :nprime] = kprime_double_full(xprime, xprime, sig2, ell) + derivative_obs_noise_std**2 * I
        ytotal[:nprime] = yprime
        if nlatent > 0:
            corr = kprime(xprime, xlatent, sig2, ell)
            K[:nprime, nprime:] = corr
            K[nprime:, :nprime] = corr.t()

    if nlatent > 0:
        I = torch.eye(nlatent, dtype=xtest.dtype, device=device)
        Kxx = k(xlatent, xlatent, sig2, ell)
        K[nprime:, nprime:] = Kxx + obs_noise_std**2 * I
        ytotal[nprime:] = ylatent

    ntest = xtest.shape[0]
    if batch_size == -1:
        batch_size = ntest
    num_batches = int(np.ceil(ntest / batch_size))
    batches = [zutil.batch_indices(i, num_batches, batch_size, ntest) for i in range(num_batches)]

    predict_mu = torch.empty(ntest, dtype=xtest.dtype)
    predict_sig2 = torch.empty(ntest, dtype=xtest.dtype)

    for bi in batches:
        xbatch = xtest[bi].to(device)

        ktest = torch.empty(len(xbatch), ntotal, dtype=xtest.dtype, device=device)
        if nprime > 0:
            ktest[:, :nprime] = kprime(xprime, xbatch, sig2, ell).t()
        if nlatent > 0:
            ktest[:, nprime:] = k(xlatent, xbatch, sig2, ell).t()

        predict_mu_batch, _ = torch.solve(ytotal.unsqueeze(-1), K)
        predict_mu_batch = torch.matmul(ktest, predict_mu_batch)
        predict_mu[bi] = predict_mu_batch.squeeze()

        predict_sig2_batch, _ = torch.solve(ktest.t(), K) # (ntotal, n_xbatch)
        predict_sig2_batch = sig2 - torch.sum(ktest.t() * predict_sig2_batch, dim=0) # (n_xbatch)
        predict_sig2[bi] = predict_sig2_batch

    return predict_mu, predict_sig2



def compute_elbo(u, m, S, xprime, yprime, x, y, sig2, ell, derivative_obs_noise_std, obs_noise_std, batch_size=-1,
                     whitened_type='ziggy', maxiter=20, precond=True, tol=1e-8, print_debug_info=False):
    M = u.shape[0]  # u is (m, )

    if whitened_type == 'cholesky':
        I = torch.eye(M, dtype=u.dtype, device=u.device)
        Kuu = k(u, u, sig2, ell)
        cKuu = torch.cholesky(Kuu + I * 1e-4, upper=False)
    else:
        # use Toeplitz tensor
        xgrids = [u]

        kern = SqExp()
        kernel = lambda x, y: kern(x, y, (sig2, ell))
        tt = ToeplitzTensor(xgrids=xgrids, kernel=kernel, batch_shape=None)

    elbo = 0
    if xprime is not None:
        nprime = xprime.shape[0]
        if batch_size == -1:
            batch_size_prime = nprime
        else:
            batch_size_prime = batch_size

        num_batches = int(np.ceil(nprime / batch_size_prime))
        batches = [zutil.batch_indices(i, num_batches, batch_size_prime, nprime) for i in range(num_batches)]

        for bi in batches:
            xbatch = xprime[bi].to(u.device)
            ybatch = yprime[bi].to(u.device)

            Knm = kprime(xbatch, u, sig2, ell)
            if whitened_type == 'cholesky':
                kn = torch.triangular_solve(Knm.t(), cKuu, upper=False)[0]
                kn = kn.squeeze()  # (m, n1)
                kn = kn.t()
            else:
                d0 = tt.inv_matmul(Knm, do_precond=precond, maxiter=maxiter, tol=tol)  # (n1, m)
                kn = tt._matmul_by_RT(d0)  # (n1, m')
            batch_elbo = compute_batch_an(m=m, S=S,  Knn_diag=sig2, kn=kn, y=ybatch,
                                          noise_std_batch=derivative_obs_noise_std,
                                          print_debug_info=print_debug_info)
            elbo += torch.sum(batch_elbo)

    if x is not None:
        nlatent = x.shape[0]
        if batch_size == -1:
            batch_size_latent = nlatent
        else:
            batch_size_latent = batch_size

        num_batches = int(np.ceil(nlatent / batch_size_latent))
        batches = [zutil.batch_indices(i, num_batches, batch_size_latent, nlatent) for i in range(num_batches)]

        for bi in batches:
            xbatch = x[bi].to(u.device)
            ybatch = y[bi].to(u.device)

            Knm = k(xbatch, u, sig2, ell)
            if whitened_type == 'cholesky':
                kn = torch.triangular_solve(Knm.t(), cKuu, upper=False)[0]
                kn = kn.squeeze()  # (m, n2)
                kn = kn.t()
            else:
                d0 = tt.inv_matmul(Knm, do_precond=precond, maxiter=maxiter, tol=tol)  # (n1, m)
                kn = tt._matmul_by_RT(d0)  # (n1, m')
            batch_elbo = compute_batch_an(m=m, S=S, Knn_diag=sig2, kn=kn, y=ybatch,
                                          noise_std_batch=obs_noise_std,
                                          print_debug_info=print_debug_info)
            elbo += torch.sum(batch_elbo)

    kl_to_prior = stats.kl_to_standard(m, S)
    #elbo = elbo / xobs.shape[0] - kl_to_prior / self.N
    elbo = elbo - kl_to_prior
    return elbo


def compute_batch_an(m, S, Knn_diag, kn, y, noise_std_batch, print_debug_info=False):
    """y: (n, ), Knn: (n, ), kn: (n, m')"""
    knt_kn = torch.sum(kn * kn, dim=-1).squeeze()  # (n, )

    # (n, m') * (m', 1) -> (n, 1) -> (n, )
    knt_m = kn.matmul(m.squeeze())

    knSkn = torch.sum(kn.matmul(S) * kn, dim=-1).squeeze()
    ivar_noise = (1 / (noise_std_batch ** 2))

    ivar_noise = (1 / (noise_std_batch ** 2))


    # compute the loss
    mse = (knt_m - y) ** 2
    variance = Knn_diag - knt_kn + knSkn
    if print_debug_info:
        print("mse = {:.4f}".format(torch.mean(mse)))
        print("variance = {:.4f}".format(torch.mean(variance)))
    batch_an = -0.5 * ivar_noise * (mse + variance) \
               - torch.log(noise_std_batch) \
               - 0.5 * np.log(2 * np.pi)
    return batch_an
