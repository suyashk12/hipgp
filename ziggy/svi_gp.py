"""
Base class for a GP that is fit using Minibach SVI updates.
acm
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import pandas as pd
import time


class SviGP(nn.Module):
    """ abstract class for a GP that can be fit using stochastic variational
    inference.

    Implements generic batch_predict, and fit.
    """
    def __init__(self):
        super(SviGP, self).__init__()
        self.pred_scale_factor = 1.

    def torch(self, npy_array):
        if isinstance(npy_array, np.ndarray):
            return torch.Tensor(npy_array).to(self.dtype)
        elif isinstance(npy_array, torch.Tensor):
            assert npy_array.dtype == self.dtype, "model dtype = {}, data dtype = {}".format(self.dtype,
                                                                                             npy_array.dtype)
            return npy_array
        else:
            raise ValueError("Only accepts np.ndarry or torch.Tensor, but got {}".format(type(np.ndarray)))

    def cuda_params(self, cuda_num=0):
        """ send necessary model attributes to device """
        raise NotImplementedError

    def elbo_and_grad(self, xbatch, ybatch, noise_std_batch, **kwargs):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        """ returns two vectors --- fmu, fisg """
        raise NotImplementedError

    def batch_solve(self, xbatch, ybatch, noise_std_batch, **kwargs):
        raise NotImplementedError

    def _make_grams(self, xbatch,
                    integrated_obs=False,
                    semi_integrated_estimator="analytic",
                    semi_integrated_samps=10):
        """ make gram matrices needed for elbo, predict, etc """
        kern_params = self.get_kernel_params()
        # data-to-inducing point covariances
        if integrated_obs:
            # only the sq exponential kernel_fun has an analytic solution ---
            # others have to be estimated (e.g. using monte carlo on the fly)
            if semi_integrated_estimator == "analytic":
                Knm = self.kernel.k_semi(
                    self.xinduce, xbatch, kern_params).transpose(0,1)
            elif semi_integrated_estimator == "mc-biased":
                Knm = self.kernel.k_semi_mc(
                    self.xinduce, xbatch, kern_params,
                    npts=semi_integrated_samps).transpose(0,1)
            elif semi_integrated_estimator == "numerical":
                Knm = self.kernel.k_semi_num(
                    self.xinduce, xbatch, kern_params).transpose(0,1)
            else:
                raise NotImplementedError
            Knn_diag = self.kernel.k_doubly_diag(xbatch, kern_params)
        else:
            Knm = self.kernel(xbatch, self.xinduce, kern_params)
            Knn_diag = self.kernel.diag(xbatch, kern_params)

        # inducing point covariance
        return Knm, Knn_diag

    def batch_predict(self, x, batch_size, verbose=True, **kwargs):
        """ Wraps mod.predict(...) into smaller batch predictions """
        # batch up the data
        num_batches = int(np.ceil(len(x) / batch_size))
        def batch_indices(it):
            idx = it % num_batches
            return slice(idx*batch_size, min((idx+1)*batch_size, len(x)))
        batches = [batch_indices(i) for i in range(num_batches)]

        # predict batches, concat and return
        res = []
        for bi, b in enumerate(batches):
            res.append( self.predict(x[b], **kwargs))
            if bi % 100 == 0 and verbose:
                print(" ... batch_predict %d / %d batches"%(bi, len(batches)))

        # un-bundle, and return
        fmu_batches  = [fmu_b for fmu_b, _ in res]
        fsig_batches = [fsig_b for _, fsig_b in res]
        return torch.cat(fmu_batches, dim=0), torch.cat(fsig_batches, dim=0)

    def fit(self, odir,
            xtrain, ytrain, noise_std_train, xtest, ftest, etest, xgrid, fgrid, egrid,
            xvalid=None, fvalid=None, evalid=None,
            batch_callback=None, epoch_callback=None,
            **kwargs):
        """ fit variational parameters of SVGP
        Args:
            - xtrain: process observations (star locations)
            - ytrain: observed values
            - itrain: is the value observed integrated? defaults to all false.

        Kwargs (see svgp_fit)
        """
        return svigp_fit(self, odir, xtrain, ytrain, noise_std_train, xtest, ftest, etest, xgrid, fgrid, egrid,
                         xvalid, fvalid, evalid,
                         batch_callback, epoch_callback, **kwargs)

    def ell_fit(self, mod, odir, xobs, yobs, sobs, **fit_kwargs):
        return ell_fit(mod, odir, xobs, yobs, sobs, **fit_kwargs)

    def estimate_predictive_variance_correction(self, xobs, aobs, sobs,
                                                **kwargs):
        self.pred_scale_factor = 1.
        fmu, fsig = self.batch_predict(xobs, batch_size=100, **kwargs)
        deltas = (aobs - fmu).squeeze()
        s2obs  = (sobs**2).squeeze()
        self.pred_scale_factor = torch.sqrt(
            (torch.sum(deltas**2) - torch.sum(sobs**2)) / torch.sum(fsig**2)
          ).item()
        print("changing pred_scale_factor to {}".format(self.pred_scale_factor))


def ell_fit(mod, odir, xobs, yobs, sobs, **fit_kwargs):
    best_ell = -1
    best_elbo = -1e10
    elbo_list = []
    ell_min, ell_max, ell_step_size = fit_kwargs['ell_min'], fit_kwargs['ell_max'], fit_kwargs['ell_step_size']
    ell_range = np.arange(ell_min, ell_max+ell_step_size, ell_step_size)
    print("Annealing ell among", list(ell_range))
    for ell in ell_range:
        # set params
        mod.update_kernel_params(ell=ell)
        elbo = mod.batch_solve(mod.torch(xobs),
                               mod.torch(yobs),
                               mod.torch(sobs),
                               batch_size=fit_kwargs['batch_solve_bsz'],
                               integrated_obs=fit_kwargs['integrated_obs'],
                               semi_integrated_estimator=fit_kwargs['ksemi_method'],
                               semi_integrated_samps=fit_kwargs['ksemi_samps'],
                               maxiter_cg=fit_kwargs['maxiter_cg'],
                               compute_elbo=True)
        elbo_list.append(elbo.detach().cpu().numpy())
        if elbo > best_elbo:
            best_ell = ell
            best_elbo = elbo
        print("ell={} elbo={:.5f} Best ell={} Best elbo={:.5f} \n".format(ell, elbo, best_ell, best_elbo))

    # batch solve model with best elbo
    ell_list = list(ell_range)
    mod.update_kernel_params(ell=best_ell)
    elbo = mod.batch_solve(mod.torch(xobs),
                           mod.torch(yobs),
                           mod.torch(sobs),
                           batch_size=fit_kwargs['batch_solve_bsz'],
                           integrated_obs=fit_kwargs['integrated_obs'],
                           semi_integrated_estimator=fit_kwargs['ksemi_method'],
                           semi_integrated_samps=fit_kwargs['ksemi_samps'],
                           maxiter_cg=fit_kwargs['maxiter_cg'],
                           compute_elbo=True)
    assert best_elbo == elbo, "best elbo = {}, elbo = {}".format(best_elbo, elbo)
    return ell_list, best_ell, elbo_list, best_elbo


def svigp_fit(mod, odir,
              xtrain, ytrain, noise_std_train, xtest, ftest, etest, xgrid, fgrid, egrid,
              xvalid, fvalid, evalid,
              batch_callback, epoch_callback,
              **fit_kwargs):
    """ Stochastic Variational Gaussian Process updates
        --- implements natural gradient updates
    Args:
        - Xtrain, Ytrain, Noise_std_train
    """
    do_cuda        = fit_kwargs.get("do_cuda", torch.cuda.is_available())
    cuda_num       = fit_kwargs.get("cuda_num", 0)
    device = torch.device('cuda:{}'.format(cuda_num))

    # natgrad parameters
    fit_method = fit_kwargs.get("fit_method", "natgrad")
    assert fit_method in ['natgrad', 'gd'], "got fit_method = {}, must choose from natgrad and gd".format(fit_method)
    lr = fit_kwargs.get("lr", 1e-2)
    schedule_lr = fit_kwargs.get("schedule_lr", True)
    step_decay = fit_kwargs.get("step_decay", .99)
    batch_size     = fit_kwargs.get("batch_size", 256)
    epochs         = fit_kwargs.get("epochs", 50)

    # kernel learning
    learn_kernel = fit_kwargs.get("learn_kernel", False)
    kernel_lr = fit_kwargs.get("kernel_lr", 1e-3)

    # noise learning  -- we will use the same lr for learning kernel parameters
    learn_noise = fit_kwargs.get("learn_noise", False)
    noise_std_init = fit_kwargs.get("noise_std_init", 0.01)

    print_debug_info = fit_kwargs.get("print_debug_info", False)
    epoch_log_interval = fit_kwargs.get("epoch_log_interval", 1)
    batch_log_interval = fit_kwargs.get("batch_log_interval", 1)  # print every X batches

    # GP stuff
    maxiter_cg     = fit_kwargs.get("maxiter_cg", 5)
    integrated_obs = fit_kwargs.get("integrated_obs", False)
    do_integrated_predictions = fit_kwargs.get("do_integrated_predictions", False)
    semi_integrated_estimator = fit_kwargs.get("semi_integrated_estimator", "analytic")
    num_semi_mc_samples = fit_kwargs.get("num_semi_mc_samples", 10)

    # regarding predictions
    predict_ksemi_method = fit_kwargs.get("predict_ksemi_method", "analytic")
    predict_ksemi_samples = fit_kwargs.get("predict_ksemi_samps", 200)
    predict_maxiter_cg = fit_kwargs.get("predict_maxiter_cg", 50)

    # misc
    eval_train = fit_kwargs.get("eval_train", False)
    only_eval_last_epoch = fit_kwargs.get("only_eval_last_epoch", False)

    print("\n-------------- Start training ---------------")

    # semi-integrated estimator/comp
    needs_semi = integrated_obs and \
        (semi_integrated_estimator=="analytic") and \
        (not mod.kernel.has_k_semi)
    if needs_semi:
        print("kernel_fun %s does not have k_semi --- doing MC estimate"%str(mod.kernel))
        semi_integrated_estimator="mc-biased"

    # set up data/batching
    assert len(xtrain.shape) == len(ytrain.shape) == 2
    xtrain       = mod.torch(xtrain)
    ytrain       = mod.torch(ytrain)
    if not learn_noise:
        assert len(noise_std_train.shape) == 2
        noise_train  = mod.torch(noise_std_train)
        dataset      = TensorDataset(xtrain, ytrain, noise_train)
    else:
        dataset = TensorDataset(xtrain, ytrain)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size, shuffle=False)

    # set up optimizer --- SGD with little to no momentum ideally
    trace = []
    optimizer = torch.optim.SGD([mod.global_theta1, mod.global_theta2], lr=lr)
    sig2_list = ell_list = noisesq_list = None
    if learn_kernel:
        sig2_list = []
        ell_list = []
        if learn_noise:
            hyper_param_optimizer = torch.optim.Adam([mod.log_ell, mod.log_sig2, mod.log_noise2], lr=kernel_lr)
            noisesq_list = []
        else:
            hyper_param_optimizer = torch.optim.Adam([mod.log_ell, mod.log_sig2], lr=kernel_lr)
    else:
        if learn_noise:
            hyper_param_optimizer = torch.optim.Adam([mod.log_noise2], lr=kernel_lr)
            noisesq_list = []
    learn_hyper = learn_kernel or learn_noise

    if schedule_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=step_decay)

    if do_cuda:
        print("Fitting SVI GP with CUDA!")
        print("device: cuda:{}".format(cuda_num))
        mod = mod.cuda_params(cuda_num)

    # track best epoch-level loss
    best_elbo, best_state = -np.inf, None
    fitting_time_list = []
    ftest_eval_time_list = []
    fgrid_eval_time_list = []
    etest_eval_time_list = []
    egrid_eval_time_list = []
    fvalid_eval_time_list = []
    evalid_eval_time_list = []
    for epoch in range(epochs):
        print("\n------- epoch {} -----------".format(epoch))

        epoch_start = time.time()
        # iterate over each batch, keeping track of number of data
        epoch_loss, nbatch, ndata, ntracked = 0., 0, 0, 0
        for train_batch in train_loader:
            if learn_noise:
                xbatch, ybatch = train_batch
                noise_std_batch = None
            else:
                xbatch, ybatch, noise_std_batch = train_batch
            batch_start = time.time()

            nbatch += 1
            ndata += xbatch.shape[0]

            if do_cuda:
                xbatch = xbatch.to(device)
                ybatch = ybatch.to(device)
                noise_std_batch = noise_std_batch.to(device)

            # batch callback --- before anything happens
            if batch_callback is not None:
                batch_callback(mod, xbatch, ybatch, noise_std_batch)

            # call loss and natural grad --- populates the natural
            # gradient into parameter grad field
            # only compute loss when logging
            compute_loss = (batch_log_interval!=False) and \
                           (nbatch % batch_log_interval == 0)

            optimizer.zero_grad()
            if learn_hyper:
                hyper_param_optimizer.zero_grad()
            lval = mod.elbo_and_grad(xbatch=xbatch, ybatch=ybatch, noise_std_batch=noise_std_batch,
                                     maxiter_cg=maxiter_cg,
                                     integrated_obs=integrated_obs,
                                     semi_integrated_estimator=semi_integrated_estimator,
                                     semi_integrated_samps=num_semi_mc_samples,
                                     print_debug_info=print_debug_info)
            if learn_hyper:
                    loss = -lval
                    loss.backward()
                    hyper_param_optimizer.step()

            # grad step at each batch
            optimizer.step()
            # make sure we decrease learning rate
            if schedule_lr:
                scheduler.step()

            # trace losses
            if compute_loss:
                batch_elapsed = time.time() - batch_start
                trace.append(lval.item())
                epoch_loss += lval.item()
                ntracked += 1
                if learn_hyper:

                    sig2, ell = mod.get_kernel_params()
                    sig2, ell = sig2.detach().cpu().numpy(), ell.detach().cpu().numpy()
                    if learn_kernel:
                        sig2_list.append(sig2)
                        ell_list.append(ell)
                    if learn_noise:
                        noisesq = torch.exp(log_noise_std).detach().cpu().numpy() ** 2
                        noisesq_list.append(noisesq)

                        print(
                            ' ... [{cb}/{tb} ({frac:.0f}%)] ELBO: {loss:.4f} sig2={sig2:.4f} ell={ell:.4f} '
                            'noisesq={noisesq:.4f} takes {t:.4f}'. \
                            format(cb=ndata,
                                   tb=len(train_loader.dataset),
                                   frac=100 * ndata / len(train_loader.dataset),
                                   loss=epoch_loss / ntracked,
                                   sig2=sig2,
                                   ell=ell,
                                   noisesq=noisesq,
                                   t=batch_elapsed))
                    else:
                        print(
                            ' ... [{cb}/{tb} ({frac:.0f}%)] ELBO: {loss:.4f} sig2={sig2:.4f} ell={ell:.4f} '
                            'takes {t:.4f}'. \
                                format(cb=ndata,
                                       tb=len(train_loader.dataset),
                                       frac=100 * ndata / len(train_loader.dataset),
                                       loss=epoch_loss / ntracked,
                                       sig2=sig2,
                                       ell=ell,
                                       t=batch_elapsed))
                else:
                    print(' ... [{cb}/{tb} ({frac:.0f}%)] ELBO: {loss:.4f} takes {t:.4f}'. \
                          format(cb=ndata,
                                 tb=len(train_loader.dataset),
                                 frac=100 * ndata / len(train_loader.dataset),
                                 loss=epoch_loss / ntracked,
                                 t=batch_elapsed))

        epoch_elbo = epoch_loss / ntracked
        epoch_elapsed = time.time() - epoch_start
        fitting_time_list.append(epoch_elapsed)
        if (epoch_log_interval != False) and (epoch % epoch_log_interval == 0):
            print("Epoch {epoch:5}: {loss:10} ({batches:4} batches) takes {t:.4f}". \
                  format(epoch=epoch, loss="%2.3f" % epoch_elbo,
                         batches="%d" % nbatch,
                         t=epoch_elapsed))

        if epoch_elbo > best_elbo:
            # best_state = deepcopy(mod.state_dict())
            best_elbo = epoch_elbo
        del xbatch
        del ybatch
        del noise_std_batch
        torch.cuda.empty_cache()
            # callback run per epoch --- call predict and plot functions
        if epoch_callback is not None:
            if (only_eval_last_epoch and epoch == epochs - 1) or not only_eval_last_epoch:
                epoch_odir = os.path.join(odir, "epoch{}".format(epoch))
                elbo_trace = trace
                save_model, save_etrace = True, True
                print("------- epoch {} -----------\n".format(epoch))
                ftest_eval_time, etest_eval_time, fgrid_eval_time, egrid_eval_time, fvalid_eval_time, evalid_eval_time \
                    = epoch_callback(epoch_odir, mod,
                               eval_train, xtrain, ytrain, noise_std_train, xtest, ftest, etest, xgrid, fgrid, egrid,
                               cuda_num, predict_maxiter_cg,
                               do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                               elbo_trace, save_model, save_etrace, elbo_trace[-1], sig2_list=sig2_list, ell_list=ell_list, noisesq_list=noisesq_list,
                               xvalid=xvalid, fvalid=fvalid, evalid=evalid)

                ftest_eval_time_list.append(ftest_eval_time)
                etest_eval_time_list.append(etest_eval_time)
                fgrid_eval_time_list.append(fgrid_eval_time)
                egrid_eval_time_list.append(egrid_eval_time)
                fvalid_eval_time_list.append(fvalid_eval_time)
                evalid_eval_time_list.append(evalid_eval_time)
        else:
            ftest_eval_time_list.append(None)
            etest_eval_time_list.append(None)
            fgrid_eval_time_list.append(None)
            egrid_eval_time_list.append(None)
            fvalid_eval_time_list.append(None)
            evalid_eval_time_list.append(None)

    time_report_df = pd.DataFrame(dict(fitting=fitting_time_list,
                                       ftest_eval=ftest_eval_time_list,
                                       etest_eval=etest_eval_time_list,
                                       fgrid_eval=fgrid_eval_time_list,
                                       egrid_eval=egrid_eval_time_list,
                                       fvalid_eval=fvalid_eval_time_list,
                                       evalid_eval=evalid_eval_time_list),
                                  index=['epoch{}'.format(i) for i in range(epochs)])
    time_report_df.loc['Total'] = time_report_df.sum()
    print("\n##############################\n")
    print("Finish training and evaluating")
    pd.options.display.float_format = '{:.4f}'.format
    print("Time report")
    print(time_report_df)
    time_report_df.to_csv(os.path.join(odir, "time_report.csv"))

    return



# check if we do not have any regularizer