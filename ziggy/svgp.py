"""
PyTorch Implementation of SVI for Sparse
Gaussian Process Graphical Models
"""
from ziggy.svi_gp import SviGP
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal, Normal, kl_divergence
import numpy as np
from copy import deepcopy
from ziggy.misc import stats as zstats
from ziggy.misc import util as zutil

#########################################################
# Stochastic Variational Gaussian Process model class   #
#########################################################


class SVGP(SviGP):

    def __init__(self, kernel, xinduce, num_obs,
                 whitened   = False,
                 sig2_init  = 1.,
                 ell_init   = 1.,
                 learn_kernel=False,
                 init_Svar  = .1,
                 prior_ell  = (.1, .025), # (mean, scale)
                 prior_sig2 = (1., 10), # (mean, scale)
                 dtype=torch.float64,
                 jitter_val=1e-3):
        """ SVGP init

        Args:
            - kernel_fun  : ziggy.kernels.Kernel object
            - xinduce : M x D Tensor of inducing point locations
            - num_obs : (scalar) number of total observations (to ensure
                we can balance the data and entropy terms in the bound)
            - whitened (optional) : optimize in the whitened space --- shouldn't
                make a difference for
            - init_Svar : initial value for the diagonal of the variational
                covariance parameter diagonal
            - dtype : defaults to float64

        """
        super(SVGP, self).__init__()

        # for some reason only doubles work with direct solves...
        assert dtype==torch.float64, "SVGP needs doubles..."
        assert kernel.dtype==torch.float64, "SVGP kernel_fun needs doublse ..."

        self.learn_kernel = learn_kernel
        self.jitter_val = jitter_val
        self.ell = nn.Parameter(torch.tensor(ell_init, dtype=dtype), requires_grad=False)
        self.log_ell = nn.Parameter(torch.log(torch.tensor(ell_init, dtype=dtype)), requires_grad=self.learn_kernel)

        self.sig2 = nn.Parameter(torch.tensor(sig2_init, dtype=dtype), requires_grad=False)
        self.log_sig2 = nn.Parameter(torch.log(torch.tensor(sig2_init, dtype=dtype)), requires_grad=self.learn_kernel)

        #kern_params = torch.Tensor([sig2_init_val, ell_init]).to(dtype).log()
        #self.kernel_params = nn.Parameter(kern_params)
        self.kernel = kernel
        self.dtype = dtype
        self.prior_ell = prior_ell
        self.prior_sig2 = prior_sig2

        # global parameters --- variational params over inducing points
        self.xinduce = xinduce.to(dtype)

        # natural parameterization 
        self.whitened = whitened
        self.global_theta1 = nn.Parameter(
            torch.zeros(len(xinduce), 1, dtype=self.dtype))
        self.global_theta2 = nn.Parameter(
            (-.5/init_Svar)*torch.eye(len(xinduce), dtype=self.dtype))
        self.N  = num_obs
        self.IM = torch.eye(len(xinduce), dtype=dtype)

    def cuda_params(self, cuda_num=0):
        """ make sure inducing points are moved to GPU"""
        device = torch.device('cuda:{}'.format(cuda_num))
        self.to(device)
        self.xinduce.to(device)
        self.kernel.to(device)
        self.xinduce.to(device)
        self.IM.to(device)
        return self

    def standard_variational_params(self):
        """ convert ExpFam mvn params to standard mean, cov """
        S = torch.inverse(-2*self.global_theta2)
        m = torch.matmul(S, self.global_theta1)
        return m, S

    def get_kernel_params(self):
        if not self.learn_kernel:
            return self.sig2, self.ell
        return torch.exp(self.log_sig2), torch.exp(self.log_ell)

    def _make_inducing_grams(self):
        kern_params = self.get_kernel_params()
        #kern_params = torch.exp(self.kernel_params)
        Kmm = self.kernel(self.xinduce, self.xinduce, kern_params)
        return Kmm

    def _make_kn_vectors(self, Knm, Kmm=None, return_Kmm=False):
        if Kmm is None:
            Kmm = self._make_inducing_grams()
        if self.whitened:
            # here we compute Kuu^{-1} Kuu^{1/2}, which can be simplified 
            # using cholesky decomposotion Kuu = L L^T and the fact that
            # Kuu^{-1} = (L^{-1})^T L^{-1}, which means that
            #
            #       Kuu^{-1} Kuu^{1/2} = Kuu^{-1} L
            #                          = (L^{-1})^T L^{-1} L 
            #                          = (L^{-1})^T
            #
            # So we need to compute Knu L^{-1}^T = Knu Lt^{-1}
            # which is equal to L^{-1} Kun = tri_solve(Kun, L)
            I = torch.eye(Kmm.shape[0]).to(Kmm.dtype)
            cKmm = torch.cholesky(Kmm + I*self.jitter_val, upper=False)
            kn = torch.triangular_solve(Knm.t(), cKmm, upper=False)[0].t()
        else:
            I = torch.eye(Kmm.shape[0]).to(Kmm.dtype).cuda()
            kn = torch.solve(Knm.t(), Kmm+self.jitter_val*I)[0].t() # N x M
        if return_Kmm:
            return kn, Kmm
        return kn

    def _make_inducing_chol_and_inv(self, Kmm):
        I = torch.eye(Kmm.shape[0], dtype=self.dtype)
        cKmm = torch.cholesky(Kmm + I*1e-5, upper=False)
        cKmm_i, _ = torch.triangular_solve(I, cKmm, upper=False)
        Kmm_i = torch.matmul(cKmm_i.t(), cKmm_i)
        return cKmm, Kmm_i

    def predict(self, x,
                integrated_obs=False,
                semi_integrated_estimator="analytic",
                semi_integrated_samps=10,
                **kwargs):
        """ return E[ f(x) ] and St-dev[ f(x) ] """
        x = x.to(self.xinduce.device)
        Knm, Knn_diag = self._make_grams(x,
            integrated_obs=integrated_obs,
            semi_integrated_estimator=semi_integrated_estimator,
            semi_integrated_samps=semi_integrated_samps)

        # create N x M matrix of kn = Knm Kmm^{-1} values
        kn = self._make_kn_vectors(Knm, return_Kmm=False)

        # standard variational parametesr m, S
        qm, qS = self.standard_variational_params()

        # mean
        fmu = torch.matmul(kn, qm)

        # variance --- differs slightly betweened whitened and non whitened
        if self.whitened:
            Ktilde_diag = Knn_diag - torch.sum(kn * kn, dim=1)
        else:
            Ktilde_diag = Knn_diag - torch.sum(kn * Knm, dim=1)
        Stilde_diag = torch.sum(kn.matmul(qS) * kn, dim=1)
        fsig = torch.sqrt(Ktilde_diag + Stilde_diag)

        # return posterior mean + standard deviation
        return fmu.detach().cpu(), fsig.detach().cpu()

    def batch_solve(self, xobs, yobs, noise_std, batch_size=-1,
                    integrated_obs=False,
                    semi_integrated_estimator="analytic",
                    semi_integrated_samps=10,
                    compute_elbo=False, **kwargs):
        print("Integrated obs? solve:", integrated_obs)

        if xobs.shape[0] != self.N:
            print("x obs shape = {}, total_num_obs = {}".format(xobs.shape[0], self.N))

        if batch_size == -1:
            batch_size = xobs.shape[0]

        num_batches = int(np.ceil(len(xobs) / batch_size))
        batches = [zutil.batch_indices(i, num_batches, batch_size, len(xobs)) for i in range(num_batches)]

        b = 0

        Kmm = self._make_inducing_grams()
        Lam = self.IM if self.whitened else torch.solve(self.IM, Kmm)[0]
        for bi in batches:
            xbatch = xobs[bi].to(self.xinduce.device)
            ybatch = yobs[bi].to(self.xinduce.device)
            noise_std_batch = noise_std[bi].to(self.xinduce.device)
            Knm, Knn_diag = self._make_grams(xbatch,
                                             integrated_obs=integrated_obs,
                                             semi_integrated_estimator=semi_integrated_estimator,
                                             semi_integrated_samps=semi_integrated_samps)

            # Kmm and kn = Knm Kmm^{-1} values
            kn = self._make_kn_vectors(Knm, Kmm=Kmm, return_Kmm=False)  # (batch_size, M)

            # covariance
            kn_tilde = (1/noise_std_batch)*kn   # (batch_size, M)

            Lam = Lam + torch.matmul(kn_tilde.t(), kn_tilde)  # (M, M)

            # mean
            y_tilde = (1/noise_std_batch)*ybatch  # (batch_size, 1)
            b += torch.matmul(kn_tilde.t(), y_tilde)  # (M, 1)

        # set information form parameters
        self.global_theta1.data[:] = b
        self.global_theta2.data[:] = -.5*Lam[:]

        if compute_elbo:
            with torch.no_grad():
                qm, qS = self.standard_variational_params()
                elbo = 0
                for bi in batches:
                    xbatch = xobs[bi].to(self.xinduce.device)
                    ybatch = yobs[bi].to(self.xinduce.device)
                    noise_std_batch = noise_std[bi].to(self.xinduce.device)

                    an = self.compute_batch_an(xbatch, ybatch, noise_std_batch,
                                               qm=qm, qS=qS, Kmm=Kmm,
                                               integrated_obs=integrated_obs,
                                               semi_integrated_estimator=semi_integrated_estimator,
                                               semi_integrated_samps=semi_integrated_samps)
                    elbo = elbo + torch.sum(an)

                if self.whitened:
                    kl_qp = zstats.kl_to_standard(qm, qS)
                else:
                    kl_qp = zstats.kl_mvn(qm, qS, torch.zeros_like(qm), Kmm)
                elbo = elbo / (xobs.shape[0]) - (kl_qp / self.N)
            return elbo

    def compute_batch_an(self, xbatch, ybatch, noise_std_batch, qm=None, qS=None, Knm=None, Knn_diag=None,
                         kn=None,
                         integrated_obs=False,
                         semi_integrated_estimator=None,
                         semi_integrated_samps=None,
                         Kmm=None,
                         print_debug_info=False):
        """compute an = -1/2 \ln 2pi sigma_n^2 - 1/(2\sigma_n)^2 [K_nn - kn^Tkn + kn^TSkn + (kn^Tm-y)^2]"""
        if qm is None or qS is None:
            qm, qS = self.standard_variational_params()

        if Knm is None or Knn_diag is None:
            Knm, Knn_diag = self._make_grams(xbatch,
                                             integrated_obs=integrated_obs,
                                             semi_integrated_estimator=semi_integrated_estimator,
                                             semi_integrated_samps=semi_integrated_samps)
        if Kmm is None:
            Kmm = self._make_inducing_grams()
        # return bsz x M matrix of Knm Kuu^{-1/2}
        if kn is None:
            kn = self._make_kn_vectors(Knm, Kmm=Kmm, return_Kmm=False)

        y = ybatch.squeeze()
        Knn = Knn_diag.squeeze()
        if self.whitened:
            knt_kn = torch.sum(kn * kn, dim=-1).squeeze()
        else:
            knt_kn = torch.sum(kn * Knm, dim=-1).squeeze()
        knt_m = kn.matmul(qm).squeeze()
        knSkn = torch.sum(kn.matmul(qS) * kn, dim=-1).squeeze()
        ivar_noise = (1 / (noise_std_batch ** 2)).squeeze()

        # compute the loss
        mse = (knt_m - y) ** 2
        variance = Knn - knt_kn + knSkn
        if print_debug_info:
            print("mse = {:.4f}".format(torch.mean(mse)))
            print("variance = {:.4f}".format(torch.mean(variance)))
        batch_an = -0.5 * ivar_noise * (mse + variance) \
                   - torch.log(noise_std_batch.squeeze()) \
                   - 0.5 * np.log(2 * np.pi)
        return batch_an

    def elbo_and_grad(self, xbatch, ybatch, noise_std_batch,
                      integrated_obs=False,
                      semi_integrated_estimator="analytic",
                      semi_integrated_samps=10,
                      compute_elbo=True,
                      compute_natgrad=True,
                      compute_kernelgrads=False,
                      **kwargs):
        """ computes elbo and/or natural gradient for global params """

        # data-to-inducing cross covariances
        Knm, Knn_diag = self._make_grams(xbatch,
            integrated_obs=integrated_obs,
            semi_integrated_estimator=semi_integrated_estimator,
            semi_integrated_samps=semi_integrated_samps)

        # kn = Knm Kmm^{-1} (or Knm Kmm^{-1/2} if whitened)
        kn, Kmm = self._make_kn_vectors(Knm)

        # make variational parameters --- be sure to whiten/uniwhiten
        qm, qS = self.standard_variational_params()
        noise_var = noise_std_batch**2

        # rescale to data of size N=1
        bscale = self.N/xbatch.shape[0]

        # compute L3 in GPs for Big Data
        elbo_estimate = None
        if compute_elbo:

            # elbo in whitened/non whitened parameterization
            if self.whitened:
                sn = Knn_diag - torch.sum(kn*kn, dim=1)
                kl_qp = zstats.kl_to_standard(qm, qS)
            else:
                sn = Knn_diag - torch.sum(kn*Knm, dim=1)
                kl_qp = zstats.kl_mvn(qm, qS, torch.zeros_like(qm), Kmm)

            y = ybatch.squeeze()
            kn_qm = kn.matmul(qm).squeeze()
            data_term = y**2 + sn + \
                torch.sum(kn.matmul(qS)*kn, dim=1).squeeze() + \
                kn_qm**2 - \
                2*y*kn_qm
            data_term = -.5 * data_term / (noise_std_batch.squeeze()**2)

            elbo_estimate = torch.mean(data_term) - (kl_qp/self.N)

            # compute kernel_fun gradients if called   # TODO: fix kenrel params part
            if compute_kernelgrads:
                theta_logprior = self.kernel_param_prior()
                #print("Theta log prior:", theta_logprior)
                eloss = -(elbo_estimate + theta_logprior)
                eloss.backward()
                # make sure kernel_fun param grads are not nan
                assert np.all(~np.isnan(self.kernel_params.grad.data.cpu().numpy()))

        # natural gradient compute
        if compute_natgrad:

            # compute natgrad for S (global_theta2)
            kn_tilde = (1/noise_std_batch)*kn
            if self.whitened:
                Lam = bscale*torch.matmul(kn_tilde.t(), kn_tilde) + self.IM
            else:
                Lam = bscale*torch.matmul(kn_tilde.t(), kn_tilde) + \
                    torch.solve(self.IM, Kmm)[0]
            dS = -.5*Lam - self.global_theta2.data

            # compute natgrad for m (global_theta1)
            y_tilde = (1/noise_std_batch)*ybatch
            kn_tilde = (1/noise_std_batch)*kn
            uhat = torch.matmul(kn_tilde.t(), y_tilde) # M x 1
            dm = bscale*uhat - self.global_theta1.data

            # manually set grad to natural grad (NEGATIVE!)
            self.global_theta2.grad = -(dS / self.N) * 1000
            self.global_theta1.grad = -(dm / self.N) * 1000

        return elbo_estimate

    def kernel_param_prior(self):
        kernel_params = self.get_kernel_params()
        ln_sig2, ln_ell = torch.log(kernel_params)
        #ln_sig2, ln_ell = self.kernel_params

        # gamma prior on ell, with mean/scale given
        ell_mu, ell_sig = self.prior_ell
        alpha, beta = zstats.gamma_params(ell_mu, ell_sig**2)
        ll_ell = zstats.lngamma_pdf_lnx(ln_ell, alpha, beta)

        # TODO --- make this an inverse gamma w/ this mean and variance
        # so that it avoids zero --- push up on ell
        #ell_mu, ell_sig = self.prior_ell
        #ll_ell = -.5*(1/(ell_sig*ell_sig))*(ell-ell_mu)**2
        return ll_ell


#####################
# simple 1-d test!  #
#####################

if __name__=="__main__":

    # create synthetic data
    import numpy as np
    np.random.seed(40)
    fun = lambda x: np.sin(x)
    xgrid = np.linspace(0, 12, 100)
    fgrid = fun(xgrid)

    Nobs = 1000
    noise_std_train = 2.5*np.ones(Nobs)[:,None]
    xtrain = np.random.rand(Nobs)[:,None]*xgrid.max()
    ytrain = fun(xtrain) + np.random.randn(*xtrain.shape)*noise_std_train

    #####################
    # create model      #
    #####################
    from ziggy.kernels import SqExp
    dtype = torch.float64
    xinduce = torch.linspace(xgrid.min(), xgrid.max(), 20,
                             dtype=dtype)[:,None]
    kern = SqExp(dtype=dtype)
    mod  = SVGP(kernel=kern,
                xinduce=xinduce,
                num_obs=len(xtrain),
                dtype=dtype,
                whitened=True,
                init_Svar=1,
                ell_init=.3,
                sig2_init=10.)

    # torch version of data
    xt, yt, nt = [torch.Tensor(a).to(mod.dtype)
                  for a in [xtrain, ytrain, noise_std_train]]
    def batch_elbo():
        # elbo val?
        lval = mod.elbo_and_grad(xt, yt, nt,
                                 integrated_obs=False,
                                 compute_elbo=True,
                                 compute_natgrad=False,
                                 compute_kernelgrads=False)
        return lval.item()

    #########
    # FIT!  #
    #########
    print("pre fit lval: ", batch_elbo())

    # fit!
    res = mod.fit(xtrain, ytrain, noise_std_train,
                  epochs=100,
                  natgrad_rate=.1,
                  kernel_rate=1e-2,
                  kernel_mom=.99,
                  batch_size=200)
    print("post fit lval: ", batch_elbo())
    print("  ... post fit kernels params: ", mod.kernel_params.exp())
    fmu, fsig = mod.predict(torch.Tensor(xgrid).to(mod.dtype)[:,None])
    fmu = fmu.detach().numpy().squeeze()
    fsig = fsig.detach().numpy().squeeze()

    # do analytic batch_solve
    mod.batch_solve(xt, yt, nt, integrated_obs=False)
    print("optimal lval?", batch_elbo())
    opt_val = batch_elbo()
    fmu_opt, fsig_opt = mod.predict(torch.DoubleTensor(xgrid)[:,None])
    fmu_opt = fmu_opt.detach().numpy().squeeze()
    fsig_opt = fsig_opt.detach().numpy().squeeze()

    #########
    # Plots #
    #########
    import matplotlib.pyplot as plt; plt.ion()
    plt.close("all")
    fig, ax = plt.figure(figsize=(8,6)), plt.gca()
    ax.plot(res['trace'][1:])
    xlim = ax.get_xlim()
    ax.plot(xlim, (opt_val, opt_val))

    fig, axarr = plt.subplots(1, 2, figsize=(12,4)) #plt.figure(figsize=(8,6)), plt.gca()
    for ax, (fm, fs), name in zip(axarr.flatten(),
                              [(fmu, fsig), (fmu_opt, fsig_opt)],
                              ["sgd", "opt"]):
        ax.plot(xgrid, fgrid)
        ax.scatter(xtrain[:100], ytrain[:100])

        ax.plot(xgrid, fm, label=name)
        ax.fill_between(xgrid, y1=fm-2*fs, y2=fm+2*fs, alpha=.25)
        ax.legend()

