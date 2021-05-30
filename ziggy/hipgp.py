"""
Hierarchical Inducing point GP with prior defined in expanded space (2M, 1), where M = number of inducing points
along each dimension

"""
from ziggy.svi_gp import SviGP
import torch
from torch import nn
import numpy as np
from ziggy.misc import stats
from ziggy.misc import util as zutil
from ziggy.misc.toeplitz_tensor import ToeplitzTensor


class ToeplitzInducingGP(SviGP):

    def __init__(self, kernel, xgrids, num_obs,
                 sig2_init=1.,
                 ell_init=.05,
                 noise2_init=1.,
                 learn_kernel=True,
                 learn_noise=True,
                 dtype=torch.float,
                 whitened_type='ziggy',
                 parameterization='expectation-family',
                 jitter_val=1e-3):
        """ SVGP init
        Args:
            - kernel_fun  : svgp.Kernel object
            - xgrids : list of one-dimensional grids that define
                multi-dimensional inducing point mesh
            - num_obs : (scalar) number of total observations (to ensure
                we can balance the data and entropy terms in the bound)
            - dtype : defaults to float64
            - whitened_type: choose between ziggy and cholesky
            - parameterization: choose between standard and expectation-family
        """
        super(ToeplitzInducingGP, self).__init__()
        # sig2, ell for stationary kernels --- todo, generalize this  # TODO: xgrids, dtype check
        #cov_params = [sig2_init_val, ell_init]
        self.learn_kernel = learn_kernel
        self.learn_noise = learn_noise
        self.jitter_val = jitter_val

        self.ell = torch.tensor(ell_init, dtype=dtype)
        self.log_ell = nn.Parameter(torch.log(torch.tensor(ell_init, dtype=dtype)), requires_grad=self.learn_kernel)

        self.sig2 = torch.tensor(sig2_init, dtype=dtype)
        self.log_sig2 = nn.Parameter(torch.log(torch.tensor(sig2_init, dtype=dtype)), requires_grad=self.learn_kernel)

        self.noise2 = torch.tensor(noise2_init, dtype=dtype) # likelihood noise covar
        self.log_noise2 = nn.Parameter(torch.log(torch.tensor(noise2_init, dtype=dtype)), requires_grad=self.learn_noise)

        self.kernel = kernel
        self.dtype = dtype
        self.N = num_obs
        print(f"Model initialization: sig2 = {sig2_init:.2f}, ell_init = {ell_init:.2f}, noise2 = {noise2_init:.2f}")

        # set up toeplitz prior stuff
        assert len(xgrids) > 1, len(xgrids)
        self.xgrids = xgrids
        # dims = [len(xg) for xg in self.grids]
        xxs  = torch.meshgrid(*self.xgrids)
        self.xinduce = torch.stack([x.reshape(-1) for x in xxs], dim=-1)
        self.M = len(self.xinduce)

        self.whitened_type = whitened_type
        if self.whitened_type == 'cholesky':
            self.Mprime = self.M
        else:
            assert self.whitened_type == 'ziggy', self.whitened_type
            self.Mprime = int(np.prod([2 * len(xg) - 2 if len(xg) > 1 else len(xg) for xg in self.xgrids]))

        self.parameterization = parameterization

    @property
    def name(self):
        raise NotImplementedError

    def cuda_params(self, cuda_num=0):
        """ make sure inducing points are moved to GPU"""
        device = torch.device('cuda:{}'.format(cuda_num))
        self.to(device)
        self.ell.to(device)
        self.sig2.to(device)
        self.noise2.to(device)
        self.xinduce = self.xinduce.to(device)
        self.kernel  = self.kernel.to(device)
        self.xgrids  = [x.to(device) for x in self.xgrids]
        return self

    def update_kernel_params(self, sig2=None, ell=None, sig2_grad=None, ell_grad=None):
        assert not self.learn_kernel
        if sig2_grad is not None:
            self.sig2 -= sig2_grad
        else:
            if sig2 is not None:
                self.sig2 = torch.tensor(sig2, dtype=self.dtype, device=self.sig2.device)
        if ell_grad is not None:
            self.ell -= ell_grad
        else:
            if ell is not None:
                self.ell = torch.tensor(ell, dtype=self.dtype, device=self.ell.device)
        print("Updated kernel params: sig2 = {:.5f}, length scale = {:.5f}".format(self.sig2, self.ell))

    def get_kernel_params(self):
        if not self.learn_kernel:
           return self.sig2, self.ell
        return torch.exp(self.log_sig2), torch.exp(self.log_ell)

    def sample(self, x, n, integrated_obs=False):
        # TODO: sample observations from the GP model
        # TODO: check if sampling GP observations then integrating = sampling from integrated GP -- prior checking
        # TODO: sample from SVGP v.s. sample from GP?
        raise NotImplementedError

    def compute_kn(self, Knm, maxiter_cg=10, tol=1e-8, Kmm=None):
        """
        compute kn = R^T Kmm^{-1} Kmn if whitened_type == 'ziggy',
        kn = L^{-1} Kmn if whitened_type == 'cholesky'
        Args:
            - Knm: bsz x M matrix, where M=m1*m2*...*MD
            - maxiter_cg: maximum number of conjugate gradient iterations

        Returns:
            - kn:   bsz x M' matrix of results, where M' = (2m1-2)*(2m2-2)*...*(2mD-2) if whitened_type == 'ziggy'
                    or, bsz x M matrix of results, if whitened_type == 'cholesky'

        """
        # compute the Kmm^{-1} Kmn terms for each batch
        cov_params = self.get_kernel_params()
        if self.whitened_type == 'cholesky':
            if Kmm is None:
                Kmm = self.kernel(self.xinduce, self.xinduce, cov_params)
            I = torch.eye(Kmm.shape[0], dtype=Knm.dtype, device=Knm.device)
            cKmm = torch.cholesky(Kmm + I * self.jitter_val, upper=False)
            kn = torch.triangular_solve(Knm.t(), cKmm, upper=False)[0].t()
        else:
            kfun = lambda x, y: self.kernel(x, y, params=cov_params)
            # return bsz x M matrix
            # first solve d = Kmm^{-1} K_mn using PCG, then solve kn = R^T d
            if Kmm is None:
                Kmm = ToeplitzTensor(xgrids=self.xgrids, kernel=kfun, batch_shape=None, jitter_val=self.jitter_val)
            d0 = Kmm.inv_matmul(Knm, do_precond=True, maxiter=maxiter_cg, tol=tol)  # (bsz, M)
            kn = Kmm._matmul_by_RT(d0)  # (bsz, M')
        return kn

    def compute_knSkn(self, kn, qS):
        raise NotImplementedError

    def get_identity_for_lam(self):
        raise NotImplementedError

    def get_lam(self, ivar_noise, kn, bscale=None, add_identity=True):
        raise NotImplementedError

    def get_kl_to_prior(self, qm, qS):
        raise NotImplementedError

    def elbo(self, xbatch, ybatch, noise_std_batch=None,
             maxiter_cg=10,
             integrated_obs=False,
             semi_integrated_estimator="analytic",
             semi_integrated_samps=10,
             Kmm=None,
             print_debug_info=False):
        """computes elbo and/or natural gradient for global params
        xbatch shape: (bsz, D), ybatch shape: (bsz, 1). noise_std_batch shape: None or (bsz, 1)
        """
        # data-to-inducing cross covariances
        Knm, Knn_diag = self._make_grams(xbatch,
            integrated_obs=integrated_obs,
            semi_integrated_estimator=semi_integrated_estimator,
            semi_integrated_samps=semi_integrated_samps)

        # return bsz x M' matrix of R^T Kmm^{-1]Kmn, where M' = (2m1-2)*(2m2-)*...*(2mD-2)
        kn = self.compute_kn(Knm, maxiter_cg=maxiter_cg, Kmm=Kmm)
        qm, qS = self.standard_variational_params()  # with gradient

        batch_an = self.compute_batch_an(xbatch, ybatch, noise_std_batch, qm=qm, qS=qS,
                                           Knm=Knm, Knn_diag=Knn_diag, kn=kn,
                                           maxiter_cg=maxiter_cg, integrated_obs=integrated_obs,
                                           semi_integrated_estimator=semi_integrated_estimator,
                                           semi_integrated_samps=semi_integrated_samps,
                                            print_debug_info=print_debug_info)
        kl_to_prior = self.get_kl_to_prior(qm, qS)
        elbo_estimate = torch.mean(batch_an) - (kl_to_prior / self.N)

        if print_debug_info:
            print("an = {:.4f}".format(torch.mean(batch_an)))
            print("kl = {:.4f}".format(kl_to_prior/self.N))
        return elbo_estimate

    def elbo_and_grad(self, xbatch, ybatch, noise_std_batch=None,
                      maxiter_cg=10,
                      integrated_obs=False,
                      semi_integrated_estimator="analytic",
                      semi_integrated_samps=10,
                      print_debug_info=False,
                      Kmm=None):
        """computes elbo and natural gradient for global params
        xbatch shape: (bsz, D), ybatch shape: (bsz, 1). noise_std_batch shape: None or (bsz, 1)
        """
        assert self.parameterization == 'expectation-family', "need parameterization=expectation-family " \
                                                      "when performing natural gradient descent"

        # data-to-inducing cross covariances
        Knm, Knn_diag = self._make_grams(xbatch,
            integrated_obs=integrated_obs,
            semi_integrated_estimator=semi_integrated_estimator,
            semi_integrated_samps=semi_integrated_samps)

        # return bsz x M' matrix of R^T Kmm^{-1]Kmn, where M' = (2m1-2)*(2m2-)*...*(2mD-2)
        kn = self.compute_kn(Knm, maxiter_cg=maxiter_cg, Kmm=Kmm)
        # var params
        with torch.no_grad():
            qm, qS = self.standard_variational_params()  # without gradient since we compute grad for theta1, theta2 manually.
        # However we still trace kernel grads if learn_kernel_grads=True

        batch_an = self.compute_batch_an(xbatch, ybatch, noise_std_batch, qm=qm, qS=qS,
                                         Knm=Knm, Knn_diag=Knn_diag, kn=kn,
                                         maxiter_cg=maxiter_cg, integrated_obs=integrated_obs,
                                         semi_integrated_estimator=semi_integrated_estimator,
                                         semi_integrated_samps=semi_integrated_samps,
                                         print_debug_info=print_debug_info)
        kl_to_prior = self.get_kl_to_prior(qm, qS)
        elbo_estimate = torch.mean(batch_an) - (kl_to_prior / self.N)

        bscale = self.N/xbatch.shape[0]

        # compute big lambda
        if noise_std_batch is not None:
            ivar_noise = (1 / (noise_std_batch ** 2))  # (bsz, 1)
        else:
            ivar_noise = torch.exp(-self.log_noise2)

        # computing the gradient:
        # dL/deta1 = dL/dm -2dL/dS
        # dL/deta2 = dL/dS

        knt_m = kn.matmul(qm)  # (bsz, 1)
        bdiff = ivar_noise * (knt_m - ybatch)  # (bsz, 1)
        data_dm = -torch.matmul(bdiff.t(), kn).t()  # M x 1 vector
        dm = bscale * data_dm - qm

        # nat grad
        if self.name == 'mean-field':
            lam_diag = bscale * torch.sum(ivar_noise * kn * kn, dim=0) + 1
            dS = -.5 * lam_diag[:, None] - self.global_theta2.data
            deta1 = dm + dS * (-2*qm)
        elif self.name == 'block':
            blk_kn = self.to_blocks(kn)[..., None]
            knkn_t = torch.matmul(blk_kn, blk_kn.transpose(-1, -2))  # bsz x num_blocks x block_size x block_size
            batch_knkn_t = torch.sum(ivar_noise.unsqueeze(-1).unsqueeze(-1) * knkn_t,
                                     dim=0)  # num_blocks x block_size x block_size
            _, num_blks, blk_size, _ = knkn_t.shape
            blk_I = torch.eye(blk_size, device=batch_knkn_t.device)[None, :, :].repeat(num_blks, 1, 1)
            lam_block = bscale * batch_knkn_t + blk_I
            dS = -.5 * lam_block - self.global_theta2.data
            dSdeta1 = self.block_diag_multiply(dS, -2*qm[None, :, 0])  # dL/dS * dS/deta1, shape (1, Mprime_
            deta1 = dm + dSdeta1.squeeze().unsqueeze(-1)
        else:
            lam = bscale * (ivar_noise * kn).t().matmul(kn) + torch.eye(self.Mprime, device=kn.device)
            dS = -.5 * lam - self.global_theta2.data
            b = torch.sum(ivar_noise * ybatch * kn, dim=0, keepdim=True)
            deta1 = b - self.global_theta1.data
        deta2 = dS
        # (M', 1) for mf, (M', M') for fullrank, (num_blocks, block_size, block_size for block model)

        self.global_theta1.grad = -deta1
        self.global_theta2.grad = -deta2
        if print_debug_info:
            for name, vec in zip(["Theta1", "Theta2", "dTheta1", "dTheta2"],
                                 [self.global_theta1, self.global_theta2, deta1, deta2]):
                zutil.print_vec(name, vec)
        return elbo_estimate

    def batch_solve(self, xobs, yobs, noise_std=None, batch_size=-1, maxiter_cg=10,
                    integrated_obs=False, semi_integrated_estimator="analytic",semi_integrated_samps=10,
                    compute_elbo=False, Kmm=None, print_debug_info=False):
        if xobs.shape[0] != self.N:
            print("x obs shape = {}, total_num_obs = {}".format(xobs.shape[0], self.N))

        if batch_size == -1:
            batch_size = xobs.shape[0]

        num_batches = int(np.ceil(len(xobs) / batch_size))
        batches = [zutil.batch_indices(i, num_batches, batch_size, len(xobs)) for i in range(num_batches)]

        if self.whitened_type == 'cholesky':
            cov_params = self.get_kernel_params()
            Kmm = self.kernel(self.xinduce, self.xinduce, cov_params)
        else:
            Kmm = None

        b = 0
        lam = self.get_identity_for_lam()

        if self.name != 'full-rank':
            big_lam = torch.eye(self.Mprime, dtype=self.dtype, device=self.xgrids[0].device)

        with torch.no_grad():
            for bi in batches:
                xbatch = xobs[bi].to(self.xgrids[0].device)
                ybatch = yobs[bi].to(self.xgrids[0].device)
                Knm, Knn_diag = self._make_grams(xbatch,
                                                 integrated_obs=integrated_obs,
                                                 semi_integrated_estimator=semi_integrated_estimator,
                                                 semi_integrated_samps=semi_integrated_samps)

                # return bsz x M' matrix
                kn = self.compute_kn(Knm, maxiter_cg=maxiter_cg, Kmm=Kmm)

                if noise_std_batch is not None:
                    noise_std_batch = noise_std[bi].to(self.xgrids[0].device)
                    ivar_noise = (1/(noise_std_batch**2))  # (bsz, 1)
                else:
                    ivar_noise = torch.exp(-self.log_noise2)

                # (M', L ) * (L, M') -> (M' M')
                lam = lam + self.get_lam(ivar_noise=ivar_noise, kn=kn, bscale=1.0, add_identity=False)
                b = b + torch.sum(ivar_noise * ybatch * kn, dim=0)

                if self.name != 'full-rank':
                    big_lam += (ivar_noise * kn).t().matmul(kn)

            if self.parameterization == 'standard':
                self.global_S.data[:] = self.get_S_from_lam(lam)
                if self.name == 'full-rank':
                    self.global_m.data[:] = torch.matmul(self.global_S, b[:,None])
                else:
                    mhat, _ = torch.solve(b[:, None], big_lam)  # (M', 1)
                    self.global_m.data[:] = mhat
            else:
                self.global_theta2.data[:] = -.5 * lam
                if self.name == 'mean-field':
                    mhat, _ = torch.solve(b[:, None], big_lam)  # (M', 1)
                    nhat = mhat.squeeze() * lam.squeeze()
                    self.global_theta1.data[:] = nhat[:, None]  # b[:, None]
                elif self.name == 'block':
                    mhat, _ = torch.solve(b[:, None], big_lam)  # (M', 1)
                    nhat = self.block_diag_multiply(lam, mhat.t()).t() # TODO: check this
                    self.global_theta1.data[:] = nhat
                else:
                    self.global_theta1.data[:] = b[:, None]

        if compute_elbo: # we may want to trace gradient here
            qm, qS = self.standard_variational_params()
            # qS = torch.inverse(lam)
            # qm = torch.matmul(qS, b[:, None])
            elbo = 0
            for bi in batches:
                xbatch = xobs[bi].to(self.xgrids[0].device)
                ybatch = yobs[bi].to(self.xgrids[0].device)
                if noise_std_batch is not None:
                    noise_std_batch = noise_std[bi].to(self.xgrids[0].device)
                batch_elbo = self.compute_batch_an(xbatch, ybatch, noise_std_batch, qm=qm, qS=qS,
                                                   maxiter_cg=maxiter_cg,
                                                   integrated_obs=integrated_obs,
                                                   semi_integrated_estimator=semi_integrated_estimator,
                                                   semi_integrated_samps=semi_integrated_samps,
                                                   Kmm=Kmm,
                                                   print_debug_info=print_debug_info)
                elbo += torch.sum(batch_elbo)

            kl_to_prior = self.get_kl_to_prior(qm, qS)
            elbo = elbo / xobs.shape[0] - kl_to_prior / self.N
            return elbo

    def compute_batch_an(self, xbatch, ybatch, noise_std_batch=None, qm=None, qS=None, Knm=None, Knn_diag=None,
                         kn=None, maxiter_cg=10,
                         integrated_obs=False,
                         semi_integrated_estimator=None,
                         semi_integrated_samps=None,
                         cache_K_matmul=None,
                         print_debug_info=False,
                         Kmm=None):
        """compute an = -1/2 \ln 2pi sigma_n^2 - 1/(2\sigma_n)^2 [K_nn - kn^Tkn + kn^TSkn + (kn^Tm-y)^2]"""
        if qm is None or qS is None:
            qm, qS = self.standard_variational_params()

        if Knm is None or Knn_diag is None:
            Knm, Knn_diag = self._make_grams(xbatch,
                                             integrated_obs=integrated_obs,
                                             semi_integrated_estimator=semi_integrated_estimator,
                                             semi_integrated_samps=semi_integrated_samps)

        # return bsz x M matrix of Knm Kuu^{-1/2}
        if kn is None:
            kn = self.compute_kn(Knm, maxiter_cg=maxiter_cg, Kmm=Kmm)

        # Equation 36 in the writeup
        y = ybatch.squeeze()
        Knn = Knn_diag.squeeze()
        knt_kn = torch.sum(kn * kn, dim=-1).squeeze()
        knt_m = kn.matmul(qm).squeeze()
        knSkn = self.compute_knSkn(kn, qS)
        if noise_std_batch is not None:
            ivar_noise = (1 / (noise_std_batch ** 2)).squeeze()
            log_noise_std = torch.log(noise_std_batch)
        else:
            ivar_noise = torch.exp(-self.log_noise2)
            log_noise_std = 0.5 * self.log_noise2

        # compute the loss
        mse = (knt_m - y)**2
        variance = Knn - knt_kn + knSkn
        if print_debug_info:
            print("mse = {:.4f}".format(torch.mean(mse)))
            print("variance = {:.4f}".format(torch.mean(variance)))
        batch_an = -0.5 * ivar_noise * (mse + variance) \
                     - log_noise_std \
                     - 0.5 * np.log(2 * np.pi)
        return batch_an

    def predict(self, x, integrated_obs=False,
                semi_integrated_estimator="analytic",
                semi_integrated_samps=10,
                maxiter_cg=50,
                Kmm=None):
        """ return E[ f(x) ] and Var[ f(x) ] """
        # make sure x lives on the device of the kernel_fun
        x = x.to(self.xgrids[0].device)
        Knm, Knn_diag = self._make_grams(x,
            integrated_obs=integrated_obs,
            semi_integrated_estimator=semi_integrated_estimator,
            semi_integrated_samps=semi_integrated_samps)

        # return bsz x M matrix of Knm Kuu^{-1/2}
        kn = self.compute_kn(Knm, maxiter_cg=maxiter_cg, Kmm=Kmm)

        # var params
        qm, qS = self.standard_variational_params()

        # predictive mean
        mu_star = torch.matmul(kn, qm)

        # predictive marginal variance
        ktilde_star = Knn_diag - torch.sum(kn*kn, dim=-1)
        #print("Less than zero?", torch.sum(ktilde_star < 0.))
        ktilde_star = ktilde_star.clamp_min(1e-5)
        knSkn = self.compute_knSkn(kn, qS)
        sig_star = torch.sqrt(ktilde_star + knSkn)[:,None]
        #print(" ... less than zero:", torch.sum( (ktilde_star+knSkn)<0.))

        return mu_star.cpu().detach(), sig_star.cpu().detach()


class MeanFieldToeplitzGP(ToeplitzInducingGP):

    def __init__(self, kernel, xgrids, num_obs,
                 sig2_init=1.,
                 ell_init=.05,
                 noise2_init=1.,
                 init_Svar=.1,
                 learn_kernel=False,
                 learn_noise=False,
                 dtype=torch.float,
                 whitened_type='ziggy',
                 parameterization='expectation-family',
                 jitter_val=1e-3):
        """ SVGP init
        Args:
            - kernel_fun  : svgp.Kernel object
            - xinduce_grids : list of one-dimensional grids that define
                multi-dimensional inducing point mesh
            - num_obs : (scalar) number of total observations (to ensure
                we can balance the data and entropy terms in the bound)
            - init_Svar : initial value for the diagonal of the variational
                covariance parameter diagonal
            - dtype : defaults to float64
        """
        super(MeanFieldToeplitzGP, self).__init__(kernel, xgrids, num_obs,
                                                  sig2_init=sig2_init, ell_init=ell_init, noise2_init=noise2_init,
                                                  learn_kernel=learn_kernel,
                                                  learn_noise=learn_noise,
                                                  dtype=dtype,
                                                  whitened_type=whitened_type,
                                                  parameterization=parameterization,
                                                  jitter_val=jitter_val)

        # global parameters --- variational params over inducing points for whitened variable
        if self.parameterization == 'standard':
            self.global_m = nn.Parameter(torch.nn.init.xavier_normal_(torch.zeros(self.Mprime, 1, dtype=self.dtype)),
                                         requires_grad=True)
            self.global_S = nn.Parameter(init_Svar*torch.ones(self.Mprime, 1, dtype=self.dtype), requires_grad=True)
        else:
            self.global_theta1 = nn.Parameter(torch.nn.init.xavier_normal_(torch.zeros(self.Mprime, 1, dtype=self.dtype)),
                                              requires_grad=True)
            self.global_theta2 = nn.Parameter(
                (-.5/init_Svar)*torch.ones(self.Mprime, 1, dtype=self.dtype), requires_grad=True)

    @property
    def name(self):
        return 'mean-field'

    def standard_variational_params(self):
        if self.parameterization == 'standard':
            return self.global_m, self.global_S
        else:
            # convert ExpFam mvn params to standard mean, cov
            S = -0.5 * 1 / self.global_theta2  # (M', 1)
            m = S * self.global_theta1  # (M', 1)
            return m, S

    def get_kl_to_prior(self, qm=None, qS=None):
        if qm is None or qS is None:
            qm, qS = self.standard_variational_params()
        return stats.diag_kl_to_standard(qm, qS)

    def get_identity_for_lam(self):
        return 1

    def get_lam(self, ivar_noise, kn, bscale=1, add_identity=True):
        """Lambda = ivar_noise * kn * kn^T"""
        identity = self.get_identity_for_lam() if add_identity else 0
        lam_diag = bscale*torch.sum(ivar_noise * kn * kn, dim=0) + identity  # (M', )
        return lam_diag[:, None]

    def get_S_from_lam(self, lam):
        return 1. / lam

    def compute_knSkn(self, kn, qS):
        return torch.sum((kn * qS.t()) * kn, dim=-1).squeeze()


class BlockToeplitzGP(ToeplitzInducingGP):

    def __init__(self, kernel, xgrids, num_obs,
                 xblock_size=10,
                 block_sizes=None,
                 sig2_init=1.,
                 ell_init=.05,
                 noise2_init=1.,
                 init_Svar=.1,
                 learn_kernel=False,
                 learn_noise=False,
                 dtype=torch.float,
                 whitened_type='ziggy',
                 parameterization='expectation-family',
                 jitter_val=1e-3):
        """ BlockToeplitzGP

        creates a block-diagonal variational approximation

        Args:
            - kernel_fun     : svgp.Kernel object
            - grids     : list of one-dimensional grids that define
                multi-dimensional inducing point mesh
            - xblock_size: size of each block along one dimension -- for 2d
                inputs this implies a block size of xblock_size^2, for 3d,
                xblock_size^3.
                NOTE: If block_sizes is not None, use block_sizes instead
            -block_sizes: list of integers which specify the sizes of blocks along each dimension
            - num_obs    : (scalar) number of total observations (to ensure
                we can balance the data and entropy terms in the bound)
            - init_Svar  : initial value for the diagonal of the variational
                covariance parameter diagonal
            - block_size :
            - dtype : defaults to float64
        """
        super(BlockToeplitzGP, self).__init__(kernel, xgrids, num_obs,
                                              sig2_init=sig2_init, ell_init=ell_init, noise2_init=noise2_init,
                                              learn_kernel=learn_kernel,
                                              learn_noise=learn_noise,
                                              dtype=dtype,
                                              whitened_type=whitened_type,
                                              parameterization=parameterization,
                                              jitter_val=jitter_val)

        # number of xinduce_grids = input dimension
        input_dim = len(xgrids)
        if block_sizes is not None:
            block_dim = len(block_sizes)
            assert input_dim == block_dim, "xgrids ndim = {}, block ndim = {}".format(input_dim, block_dim)
        else:
            block_sizes = [xblock_size for _ in range(input_dim)]

        ############################################################
        # there are two orderings to juggle: toeplitz ordering (implied by
        # xinduce_grids) and block ordering, implied by a neighbor grouping.
        #
        #   - qm (mean): stored in Toeplitz ordering --- easier.
        #   - qS (block cov): stored in BLOCK ordering --- no other way
        #     when we compute matrix-vector products we need to
        #         from_block(S*to_block(v))
        #     which corresponds to computing something like
        #           P^{-1} Sb P v
        #     where S v = PSbP^T v = P Sb (P^T v)
        # create block index ordering --- this is a matrix of size
        # (num_blocks x block_size) that converts the ordering of the
        # TOEPLITZ inducing points into a block-diagonal inducing
        # point ordering

        if self.whitened_type == 'cholesky':
            self.block_idx, self.to_blocks, self.from_blocks = \
                zutil.define_block_chunks(xgrids, block_sizes)
        else:
            expanded_xgrids = self.get_expanded_xgrids(xgrids)
            self.block_idx, self.to_blocks, self.from_blocks = \
                zutil.define_block_chunks(expanded_xgrids, block_sizes)
        self.num_blocks, self.block_size = self.block_idx.shape
        # block_size = xblock_size * yblock_size (*zblock_size)
        # num_blocks = M' / block_size

        if self.parameterization == 'standard':
            self.global_m = nn.Parameter(torch.nn.init.xavier_normal_(
            torch.zeros(self.Mprime, 1, dtype=self.dtype)), requires_grad=True)
            qSinit = torch.stack([
                init_Svar * torch.eye(self.block_size, dtype=self.dtype) for _ in range(self.num_blocks)
            ])
            self.global_S = nn.Parameter(qSinit, requires_grad=True)

        else:
            # global parameters --- variational params over inducing points
            self.global_theta1 = nn.Parameter(torch.nn.init.xavier_normal_(
                torch.zeros(self.Mprime, 1, dtype=self.dtype)), requires_grad=True
            )
            # initial variational covariance matrix is ....
            qSinit = torch.stack([
                (-.5 / init_Svar) * torch.eye(self.block_size, dtype=self.dtype) for _ in range(self.num_blocks)
            ])  # (num_blocks, block_sz, block_sz)
            self.global_theta2 = nn.Parameter(qSinit, requires_grad=True)

    @property
    def name(self):
        return 'block'

    def get_expanded_xgrids(self, xgrids):
        expanded_xgrids = []
        for x in xgrids:
            m = len(x)
            expanded_xgrids.append(torch.arange(2*m-2))
        return expanded_xgrids

    def standard_variational_params(self):
        if self.parameterization == 'standard':
            return self.global_m, self.global_S
        else:
            # convert ExpFam mvn params to standard mean, cov
            S = torch.inverse(-2*self.global_theta2)  # block-inverse
            m = self.block_diag_multiply(S, self.global_theta1.t()).t()
            return m, S

    def block_diag_multiply(self, S_block, v):
        """ compute a batch of matrix-vector products where
             - S_block is a n_block x D x D block diagonal representation
             - v is a N x (n_block x D) stack of vectors
        """
        bsz, _ = v.shape
        # (num_blocks, block_size, block_size) * (bsz, num_blocks, block_size, 1) -> (bsz, num_blocks, block_size, 1)
        Sv_block = S_block.matmul(self.to_blocks(v)[...,None])
        assert Sv_block.shape == (bsz, self.num_blocks, self.block_size, 1)
        Sv = self.from_blocks(Sv_block)
        assert Sv.shape == (bsz, self.num_blocks*self.block_size), Sv.shape
        return Sv

    def get_S_from_lam(self, lam):
        return torch.inverse(lam)

    def compute_knSkn(self, kn, qS):
        Skn = self.block_diag_multiply(qS, kn)
        knSkn = torch.sum(kn * Skn, dim=-1).squeeze()
        return knSkn

    def get_identity_for_lam(self):
        return torch.eye(self.block_size, device=self.xgrids[0].device, dtype=self.dtype)

    def get_lam(self, ivar_noise, kn, bscale=1, add_identity=True):
        """
        Lambda = \sum_n 1/sigma_n^2 kn kn^T + I
        :param ivar_noise: (bsz, 1)
        :param kn: (bsz, M')
        :param bscale: a scalar
        :param add_identity: whether to add identity matrix
        :return: (num_blocks, block_size, block_size)
        """
        blk_kn = self.to_blocks(kn)  # (bsz, num_blocks, block_size)
        blk_kn = blk_kn.transpose(0, 1)  # (num_blocks, bsz, block_size)
        batch_knkn_t = torch.matmul(blk_kn.transpose(1, 2), ivar_noise * blk_kn)  # (num_blocks, block_size, block_size)

        blk_I = self.get_identity_for_lam() if add_identity else 0

        lam_block = bscale * batch_knkn_t + blk_I
        return lam_block

    def get_kl_to_prior(self, qm, qS):
        if qm is None or qS is None:
            qm, qS = self.standard_variational_params()
        return stats.block_kl_to_standard(qm, qS)


class FullRankToeplitzGP(ToeplitzInducingGP):

    def __init__(self, kernel, xgrids, num_obs,
                 sig2_init=1.,
                 ell_init=.05,
                 noise2_init=1.,
                 init_Svar=.1,
                 learn_kernel=False,
                 learn_noise=False,
                 dtype=torch.float,
                 whitened_type='ziggy',
                 parameterization='expectation-family',
                 jitter_val=1e-3):
        """ SVGP init

        Args:
            - kernel_fun  : svgp.Kernel object
            - xinduce_grids : list of one-dimensional grids that define
                multi-dimensional inducing point mesh
            - num_obs : (scalar) number of total observations (to ensure
                we can balance the data and entropy terms in the bound)
            - init_Svar : initial value for the diagonal of the variational
                covariance parameter diagonal
            - dtype : defaults to float64

        """
        super(FullRankToeplitzGP, self).__init__(kernel, xgrids, num_obs,
                                                 sig2_init=sig2_init, ell_init=ell_init, noise2_init=noise2_init,
                                                 learn_kernel=learn_kernel,
                                                 learn_noise=learn_noise,
                                                 dtype=dtype, whitened_type=whitened_type,
                                                 parameterization=parameterization,
                                                 jitter_val=jitter_val)

        # global parameters --- variational params over inducing points
        if self.parameterization == 'standard':
            self.global_m = nn.Parameter(
            torch.zeros(self.Mprime, 1, dtype=self.dtype), requires_grad=True)
            self.global_S = nn.Parameter(
                init_Svar * torch.eye(self.Mprime, dtype=self.dtype), requires_grad=True)
        else:
            self.global_theta1 = nn.Parameter(
                torch.zeros(self.Mprime, 1, dtype=self.dtype), requires_grad=True)
            self.global_theta2 = nn.Parameter(
                (-.5/init_Svar) * torch.eye(self.Mprime, dtype=self.dtype), requires_grad=False)

    @property
    def name(self):
        return 'full-rank'

    def standard_variational_params(self):
        if self.parameterization == 'standard':
            return self.global_m, self.global_S
        else:
            # convert ExpFam mvn params to standard mean, cov
            S = -0.5 * torch.inverse(self.global_theta2)
            m = torch.matmul(S, self.global_theta1)
        return m, S

    def get_S_from_lam(self, lam):
        return torch.inverse(lam)

    def get_kl_to_prior(self, qm, qS):
        if qm is None or qS is None:
            qm, qS = self.standard_variational_params()
        return stats.kl_to_standard(qm, qS)

    def compute_knSkn(self, kn, qS):
        return torch.sum(kn.matmul(qS) * kn, dim=-1).squeeze()

    def get_identity_for_lam(self):
        return torch.eye(self.Mprime, device=self.xgrids[0].device, dtype=self.dtype)

    def get_lam(self, ivar_noise, kn, bscale=1.0, add_identity=True):
        """
        lam = [\sum_n 1/sigma_n^2 kn kn^T] + I
        :param ivar_noise: (bsz, 1)
        :param kn: (bsz, M')
        :param bscale: a scalar
        :param add_identity: whether to add I
        :return: lam: a M' by M' matrix
        """
        identity = self.get_identity_for_lam() if add_identity else 0
        # (M', bsz ) * (bsz, M') -> (M' M')
        lam = bscale * (ivar_noise * kn).t().matmul(kn) + identity
        return lam

    def get_inducing_S(self, Kmm=None):
        """return RSR^T"""
        S = -0.5 * torch.inverse(self.global_theta2)
        if Kmm is None:
            cov_params = self.get_kernel_params()
            kfun = lambda x, y: self.kernel(x, y, params=cov_params)

            Kmm = ToeplitzTensor(self.xgrids, kfun, jitter_val=self.jitter_val)

        Kmm.set_batch_shape(S.shape[:-1])

        # compute v= RS
        v = Kmm._matmul_by_R(S)  # (M', M)

        # compute tilde_S.t() = Rv.t() = RSR^T
        Kmm.set_batch_shape(v.shape[1:])
        tilde_S = Kmm._matmul_by_R(v.t()).t()   # (M, M)
        return tilde_S



