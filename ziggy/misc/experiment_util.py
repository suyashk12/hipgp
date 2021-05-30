from ziggy import svgp, hipgp
from ziggy import kernels as zkern
import torch
import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
import time

def svigp_fit_predict_and_save(
    name,
    xobs, yobs, sobs,
    xinduce_grids,
    model_class = "SVGP",
    init_Svar=1.,
    xtest=None, etest=None, ftest=None,
    xvalid=None, evalid=None, fvalid=None,
    xgrid=None, egrid=None, fgrid=None,
    output_dir="./model-output/",
    epoch_callback=None,
    **fit_kwargs):
    """ Run a SVGP experiment and save output to directory
    Args:
        - name : name for this experiment/run
        - xobs, yobs, sobs : data
        - xinduce : inducing points --- same second dimension as xobs
        - init_Svar: initial value of variational covariance (mean defaults to 0)
        - xtest (optional): test location to save predictions
        - xgrid (optional): alternative test locations to save predictions
        - etest/ftest/egrid/fgrid (optional): true values being estimated
            --- save here for plotting later
        - output_directory: saves model stuff in directory
        - predict_ksemi_method : semi integrated kernel_fun method for prediction
            (don't use Monte Carlo here, it'll be inaccurate)
        - xblock_size : size of var approx block covariance along
            one dimension, e.g. xblock_size=15 and xdim=2 implies block size
            is M=15x15=225
        - fit_kwargs: arguments to pass to the SGD routine

    Example fit_kwargs:
        fit_kwargs = {'epochs'        : 10,
                  'natgrad_rate'  : .1,
                  'kernel_rate'   : 0., #1e-5,
                  'inaegrated_obs': True,
                  'batch_callback': batch_callback, ## touches all data!
                  'batch_size'    : 1000,
                  'step_decay'    : .9 }
    Saves extinction predictions, pointwise predictions, and a 
    grid of predictions to the directory `<output_dir>/<name>`
    """
    print(json.dumps(fit_kwargs))
    # require 2-dim inputs
    assert len(xobs.shape) == len(yobs.shape)
    if sobs is not None:
        assert  len(sobs.shape) == 2

    # save everything into its own output directory
    odir = os.path.join(output_dir, name)
    if not os.path.exists(odir):
        os.makedirs(odir)
    print("Saving to {}".format(odir))

    # empirically initialize cov function sig2 based on distance slope
    # compute observation value vs distance
    # slope for obs vs distance is a denoised version of the average process
    # value.  Use this as like, 1 sig unit, give or take.
    if fit_kwargs["sig2_init_val"] == "empirical":
        dobs = np.sqrt(np.sum(xobs**2, axis=-1))
        slope, res, _, _ = np.linalg.lstsq(dobs[:,None], yobs[:,None])
        sig2_init = slope[0,0]**2
        fit_kwargs['sig2_init_val'] = sig2_init

    sig2_init, ell_init = fit_kwargs['sig2_init_val'], fit_kwargs['ell_init']

    # make kernel
    kernel = fit_kwargs.get("kernel")
    dtype = torch.float64 if model_class == "SVGP" else torch.float32
    if kernel == "Mat12":
        kern = zkern.Matern(.5, dtype=dtype)
    elif kernel == "Mat32":
        kern = zkern.Matern(1.5, dtype=dtype)
    elif kernel == "Mat52":
        kern = zkern.Matern(2.5, dtype=dtype)
    elif kernel == "SqExp":
        kern = zkern.SqExp(dtype=dtype)
    else:
        raise NotImplementedError

    # fit method
    fit_method = fit_kwargs['fit_method']
    assert fit_method in ['natgrad', 'full-batch'], fit_method

    # predict k semi method
    fit_kwargs['ksemi_method'] = "analytic" if kern.has_k_semi else "mc-biased"
    fit_kwargs['ksemi_samps'] = 200
    fit_kwargs['predict_ksemi_method'] = "analytic" if kern.has_k_semi else "mc-biased"
    fit_kwargs['predict_ksemi_samps'] = 200

    # create model
    if model_class == "mean-field":
        parameterization = "expectation-family"
        mod = hipgp.MeanFieldToeplitzGP(
            kernel=kern,
            xgrids=xinduce_grids,
            num_obs=xobs.shape[0],
            sig2_init=sig2_init,
            ell_init=ell_init,
            init_Svar=init_Svar,
            whitened_type=fit_kwargs.get('whitened_type', 'ziggy'),
            parameterization=parameterization,
            learn_kernel=fit_kwargs.get('learn_kernel', False),
            jitter_val=fit_kwargs.get('jitter_val', 1e-3))
    elif "block-diagonal" in model_class:
        parameterization = "expectation-family"
        xblock_size = fit_kwargs['xblock_size']
        yblock_size = fit_kwargs['yblock_size']
        zblock_size = fit_kwargs.get('zblock_size', None)
        if zblock_size is None:
            block_sizes = [xblock_size, yblock_size]
        else:
            block_sizes = [xblock_size, yblock_size, zblock_size]
        mod = hipgp.BlockToeplitzGP(
            kernel=kern,
            xgrids=xinduce_grids,
            block_sizes=block_sizes,
            num_obs=xobs.shape[0],
            sig2_init=sig2_init,
            ell_init=ell_init,
            init_Svar=init_Svar,
            whitened_type=fit_kwargs.get('whitened_type', 'ziggy'),
            parameterization=parameterization,
            learn_kernel=fit_kwargs.get('learn_kernel', False),
            jitter_val=fit_kwargs.get('jitter_val', 1e-3)
        )
    elif model_class == "full-rank":
        mod = hipgp.FullRankToeplitzGP(
            kernel=kern,
            xgrids=xinduce_grids,
            num_obs=xobs.shape[0],
            sig2_init=sig2_init,
            ell_init=ell_init,
            whitened_type=fit_kwargs.get('whitened_type', 'ziggy'),
            parameterization='standard',
            learn_kernel=fit_kwargs.get('learn_kernel', False),
            jitter_val=fit_kwargs.get('jitter_val', 1e-3))
    elif model_class == "SVGP":
        xxs = torch.meshgrid(*xinduce_grids)
        xinduce = torch.stack([xxs[0].flatten(), xxs[1].flatten(), xxs[2].flatten()], dim=1)
        mod = svgp.SVGP(kernel=kern,
                        xinduce=xinduce,
                        num_obs=len(xobs),
                        whitened=False,
                        sig2_init=sig2_init,
                        ell_init=ell_init,
                        init_Svar=init_Svar,
                        dtype=torch.float64,
                        learn_kernel=fit_kwargs.get('learn_kernel', False),
                        jitter_val=fit_kwargs.get('jitter_val', 1e-3))
    else:
        raise NotImplementedError( "model_class=mean-field | " +
            "block-diagonal | hierarchical-vi | full-rank | SVGP")

    # keep track of ELBO trace statistics --- hacky, should just wrap up 
    # into the model
    mod.trace = []
    #################
    # Fit Model     #
    #################
    # saving fit params

    start = time.time()
    do_cuda = fit_kwargs.get("do_cuda", torch.cuda.is_available())
    if fit_method == "natgrad":
        mod.fit(odir, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                xvalid, fvalid, evalid,
                batch_callback=None, epoch_callback=epoch_callback,
                **fit_kwargs)

    elif fit_method == "full-batch":
        print("Fitting w/ Full Batch.")
        if do_cuda and torch.cuda.is_available():
            print("Batch solve uisng cuda.")
            mod = mod.cuda_params(fit_kwargs.get("cuda_num", 0))
        mod.whitened = True
        elbo = mod.batch_solve(mod.torch(xobs),
                               mod.torch(yobs),
                               mod.torch(sobs),
                               batch_size=fit_kwargs['batch_solve_bsz'],
                               integrated_obs=fit_kwargs['integrated_obs'],
                               semi_integrated_estimator=fit_kwargs['ksemi_method'],
                               semi_integrated_samps=fit_kwargs['ksemi_samps'],
                               maxiter_cg=fit_kwargs['maxiter_cg'],
                               compute_elbo=True)
        fitting_time = time.time() - start
        print("Batch solve time = {}".format(fitting_time))

        print("Elbo = {}\n".format(elbo))

        ftest_eval_time, fgrid_eval_time, etest_eval_time, egrid_eval_time, fvalid_eval_time, evalid_eval_time \
            = None, None, None, None, None, None
        if epoch_callback is not None:
            ftest_eval_time, fgrid_eval_time, etest_eval_time, egrid_eval_time, fvalid_eval_time, evalid_eval_time \
                = epoch_callback(odir, mod, fit_kwargs.get("eval_train", False),
                           xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                           fit_kwargs.get("cuda_num", 0),
                           fit_kwargs['predict_maxiter_cg'],
                           fit_kwargs.get("do_integrated_predictions", False),
                           fit_kwargs.get("ksemi_method", "analytic"),
                           fit_kwargs.get("ksemi_samples", 200),
                           elbo_trace=None,
                           save_model=True,
                           save_trace=False,
                           elbo=elbo.detach().cpu().numpy(),
                           xvalid=xvalid, evalid=evalid, fvalid=fvalid)
        time_report_df = pd.DataFrame(dict(fitting=[fitting_time], ftest_eval=[ftest_eval_time], etest_eval=[etest_eval_time],
                               fgrid_eval=[fgrid_eval_time], egrid_eval=[egrid_eval_time],
                               fvalid_eval=[fvalid_eval_time], evalid_eval=[evalid_eval_time]))
        print("\n Finish training and evaluating")
        pd.options.display.float_format = '{:,.4f}'.format
        print("Time report")
        print(time_report_df)
        time_report_df.to_csv(os.path.join(odir, "time_report.csv"))
    else:
        raise NotImplementedError
    end = time.time()
    elapsed = end - start
    print("Total fitting and evaluation time = {:.4f}".format(elapsed))

    return mod


def make_noise_comparison_dataframe(model_name, dstd, integrated_obs, train_elbo=None,
                                    max_fsig_grid=None, max_esig_grid=None, max_esig_test=None, eval_valid=False):
    # load in the error data frame
    df = make_error_dataframe([model_name], data_type='test')
    # compare integrated or non integrated obs errors
    etype = 'f'
    eresids = df['%stest'%etype].values - df['%smu_test'%etype].values
    post_rmse = np.sqrt(np.nanmean(eresids**2))
    mae = np.nanmean(np.abs(eresids))
    loglike = np.nanmean(df['%s loglike' % etype].values)


    ndict = {'post-rmse'        : post_rmse,
             'post-mae'        : mae,
             'data-noise'      : dstd,
             'noise-reduction' : 100*(dstd-post_rmse) / dstd,  # ideally, in [0,100]
             'rmse-to-std'      : post_rmse/dstd,  # ideally, in [0,1]
             'loglike'         : loglike}
    if eval_valid:
        df_valid = make_error_dataframe([model_name], data_type='valid')

        eresids_valid = df_valid['%svalid' % etype].values - df_valid['%smu_valid' % etype].values
        post_rmse_valid = np.sqrt(np.nanmean(eresids_valid ** 2))
        mae_valid = np.nanmean(np.abs(eresids_valid))
        loglike_valid = np.nanmean(df_valid['%s loglike' % etype].values)
        ndict['post-rmse-valid'] = post_rmse_valid,
        ndict['post-mae-valid'] =  mae_valid,
        ndict['loglike-valid'] = loglike_valid

    for name, val in zip(['train_elbo', 'max-fsig-grid'],
                         [train_elbo, max_fsig_grid]):
        if val is not None:
            ndict[name] = val

    if integrated_obs:
        etype = 'e'
        eresids = df['%stest' % etype].values - df['%smu_test' % etype].values
        post_rmse = np.sqrt(np.nanmean(eresids ** 2))
        mae = np.nanmean(np.abs(eresids))
        loglike = np.nanmean(df['%s loglike' % etype].values)

        ndict_e = {'post-rmse': post_rmse,
                   'post-mae': mae,
                   'data-noise': dstd,
                   'noise-reduction': 100 * (dstd - post_rmse) / dstd,  # ideally, in [0,100]
                   # 'rmse-to-std': post_rmse / dstd,  # ideally, in [0,1]
                   'loglike': loglike
                   }
        if eval_valid:
            eresids_valid = df['%stest' % etype].values - df['%smu_test' % etype].values
            post_rmse_valid = np.sqrt(np.nanmean(eresids_valid ** 2))
            mae_valid = np.nanmean(np.abs(eresids_valid))
            loglike_valid = np.nanmean(df['%s loglike valid' % etype].values)


            ndict['post-rmse-valid'] = post_rmse_valid,
            ndict['post-mae-valid'] = mae_valid,
            ndict['loglike-valid'] = loglike_valid

        for name, val in zip(['train_elbo', 'max-esig-grid', 'max-esig-test'],
                             [train_elbo, max_esig_grid, max_esig_test]):
            if val is not None:
                ndict_e[name] = val

        return pd.DataFrame(dict(fobs=ndict, eobs=ndict_e))
    return pd.DataFrame(ndict, index=['%s-obs'%etype]).T


##############################################################################
# Functions to analyze the error of the model above --- take in lists of 
# model_names = [/path/to/model/SqExp/, /path/to/model/Matern/], etc
##############################################################################

def make_error_dataframe(model_names, pretty_names=None, data_type='test'):
    """ Takes a collection of fit models (using the above method) and 
    outputs a dataframe of prediction statistics
    columns: model,
             (etest, emu_test, esig_test),
             (ftest, fmu_test, fsig_test), 
             (e mse, mae, loglike, chisq, zscore)
    """
    if pretty_names is None:
        pretty_names = [os.path.split(m)[-1] for m in model_names]

    # collect all model predictions into a dataframe
    if data_type == 'test':
        subs = ['etest', 'emu_test', 'esig_test',
                'ftest', 'fmu_test', 'fsig_test',
                'xtest_dist']
    elif data_type == 'valid':
        subs = ['evalid', 'emu_valid', 'esig_valid',
                'fvalid', 'fmu_valid', 'fsig_valid',
                'xvalid_dist']

    # for each model, creat results dictionary
    dfs = []
    for mod, pname in zip(model_names, pretty_names):
        pdict = torch.load(os.path.join(mod, "predictions.pkl"))
        mdf   = {}
        for sub in subs:
            if sub in pdict.keys() and pdict[sub] is not None:
                mdf[sub] = pdict[sub].squeeze()
            else:
                mdf[sub] = np.nan
        mdf['model'] = pname
        dfs.append(pd.DataFrame(mdf))

    df = pd.concat(dfs, axis=0)

    # compute all the errors!
    # mse
    df['e mse'] = (df['e{}'.format(data_type)]-df['emu_{}'.format(data_type)])**2
    df['f mse'] = (df['f{}'.format(data_type)]-df['fmu_{}'.format(data_type)])**2

    # mae
    df['e mae'] = np.abs(df['e{}'.format(data_type)] - df['emu_{}'.format(data_type)])
    df['f mae'] = np.abs(df['f{}'.format(data_type)] - df['fmu_{}'.format(data_type)])

    # log like
    from scipy.stats import norm
    df['e loglike'] = norm.logpdf(df['e{}'.format(data_type)],
                                  loc=df['emu_{}'.format(data_type)],
                                  scale=df['esig_{}'.format(data_type)])

    df['f loglike'] = norm.logpdf(df['f{}'.format(data_type)],
                                  loc=df['fmu_{}'.format(data_type)],
                                  scale=df['fsig_{}'.format(data_type)])

    # zscored variables
    df['e zscore'] = (df['e{}'.format(data_type)]-df['emu_{}'.format(data_type)]) / df['esig_{}'.format(data_type)]
    df['f zscore'] = (df['f{}'.format(data_type)]-df['fmu_{}'.format(data_type)]) / df['fsig_{}'.format(data_type)]

    # chi squared
    df['e chisq'] = df['e zscore']**2
    df['f chisq'] = df['f zscore']**2

    return df


def make_qq_plots(model_names, pretty_names=None, extinction=True):
    """ takes directory for model (saved above) and outputs a
    QQ plot comparing them

    Args:
        - extinction (bool) : use etest (if true) or ftest (if false)
    """
    from scipy.stats import norm

    if pretty_names is None:
        pretty_names = [os.path.split(m)[-1] for m in model_names]

    def make_qq_xy(zscores):
        pgrid = np.arange(1, len(zscores)+1) / (len(zscores)+1)
        znorm = norm.ppf(pgrid)
        return znorm

    # plot qq's
    fig, ax = plt.figure(figsize=(6,6)), plt.gca()
    ax.plot([-3, 3], [-3, 3], "--", c='grey', linewidth=2, zorder=-1)

    # for each model, plot qqs
    markers=['o', 's', 'd', '^', '3', '4', '8']
    pdicts = []
    for mod, pname, mark in zip(model_names, pretty_names, markers):
        pdict = torch.load(os.path.join(mod, "predictions.pkl"))
        ptrace = torch.load(os.path.join(mod, "elbo_trace.pkl"))
        if extinction:
            zscores = (pdict['etest'] - pdict['emu_test']) / (pdict['esig_test'])
        else:
            zscores = (pdict['ftest'] - pdict['fmu_test']) / pdict['fsig_test']
        znorm = make_qq_xy(zscores)

        ax.scatter(znorm[::5], np.sort(zscores)[::5], s=25, label=pname, marker=mark)

    ax.legend(fontsize=12, frameon=True, framealpha=.8)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)

    return fig, ax


def make_error_plots(output_dir, model_names, pretty_names=None):
    """ visualizes different test sample errors 
        - mean squared error (rho, e)
        - log like (rho, e)
        - chisq (rho, e)
    """
    df = make_error_dataframe(model_names, pretty_names)

    # plots
    sns.set_context("paper")
    error_types = ["e mse", "f mse",
                   "e mae", "f mae",
                   "e loglike", "f loglike",
                   "e chisq", "f chisq",
                   "e mse valid", "f mse valid",
                   "e mae valid", "f mae valid",
                   "e loglike valid", "f loglike valid",
                   "e chisq valid", "f chisq valid"]

    # save plot of each error type
    for etype in error_types:
        #with sns.plotting_context(font_scale=10):
        sns.set(font_scale = 1.2, style='whitegrid')

        fig, ax = plt.figure(figsize=(4,3)), plt.gca()
        ax = sns.pointplot(x="model", y=etype, data=df, join=False, ax=ax)

        if "chisq" in etype:
            xlim = ax.get_xlim()
            ax.plot(xlim, (1, 1), '--', c='grey', linewidth=1.5, alpha=.75)
            ax.set_xlim(xlim)

        # update tick fontsizes
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(11)
            tick.label.set_rotation(40)

        if "f " in etype:
            elab = etype.replace("f ", r"$\rho_*$ ")
            ax.set_ylabel(elab, fontsize=14)
        elif "e " in etype:
            elab = etype.replace("e ", r"$e_*$ ")
            ax.set_ylabel(elab, fontsize=14)

        fig.savefig(os.path.join(output_dir, "test-error-%s.pdf"%etype), bbox_inches='tight')
        plt.close("all")


def plot_posterior_grid(pdict, output_dir, ticklabels=True):
    """ Visualize posteiror grid slice
    """
    from ziggy import viz

    xlo, xhi = pdict['xgrid'][:, 0].min(), pdict['xgrid'][:, 0].max()
    ylo, yhi = pdict['xgrid'][:, 1].min(), pdict['xgrid'][:, 1].max()
    nx, ny = pdict['fgrid'].shape
    vmin, vmax = pdict['fgrid'].min(), pdict['fgrid'].max()

    # plot posterior mean on xgrid
    sns.set(font_scale = 1.2, style='white')

    pkwargs = {'xlim': (xlo, xhi),
               'ylim': (ylo, yhi),
               'ticklabels': ticklabels}

    fig, ax = plt.figure(figsize=(6,6)), plt.gca()
    cm = viz.plot_smooth(ax, pdict['fmu_grid'].reshape(nx, ny),
                         vmin=vmin, vmax=vmax, **pkwargs)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    fig.savefig(os.path.join(output_dir, "posterior-fmu.pdf"), bbox_inches='tight')
    plt.close()

    fig, ax = plt.figure(figsize=(6,6)), plt.gca()
    cm = viz.plot_smooth(ax, 2*pdict['fsig_grid'].reshape(nx, ny),
                         vmin=0., **pkwargs)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    fig.savefig(os.path.join(output_dir, "posterior-fsig.pdf"), bbox_inches='tight')
    plt.close()

    # plot the true grid as well
    fig, ax = plt.figure(figsize=(6,6)), plt.gca()
    cm = viz.plot_smooth(ax, pdict['fgrid'],
                         vmin=vmin, vmax=vmax, **pkwargs)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    fig.savefig(os.path.join(output_dir, "true-fgrid.pdf"), bbox_inches='tight')
    plt.close()

    # plot residual and zscore
    fig, ax = plt.figure(figsize=(6,6)), plt.gca()
    resid = pdict['fgrid'] - pdict['fmu_grid'].reshape(nx, ny)
    cm = viz.plot_smooth(ax, resid, **pkwargs)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    fig.savefig(os.path.join(output_dir, "residual-fgrid.pdf"), bbox_inches='tight')
    plt.close()

    fig, ax = plt.figure(figsize=(6,6)), plt.gca()
    zs = resid / pdict['fsig_grid'].reshape(nx, ny)
    cm = viz.plot_smooth(ax, zs.reshape(nx, ny), **pkwargs)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    fig.savefig(os.path.join(output_dir, "zscore-fgrid.pdf"), bbox_inches='tight')
    plt.close()


def make_model_distance_plots(model_name, pretty_name=None):
    """ make plots that show error as a function of distance """
    df = make_error_dataframe([model_name], None)

    # plot posterior mean on xgrid
    error_types = ["e mse", "f mse",
                   "e mae", "f mae",
                   "e loglike", "f loglike",
                   "e chisq", "f chisq",
                   "esig_test", "fsig_test", 
                   "etest", "ftest"]
    for error_type in error_types:
        fig, ax = plt.figure(figsize=(6,4)), plt.gca()
        ax.scatter(df['xtest_dist'], df[error_type], s=10)
        ax.set_xlabel("distance", fontsize=12)
        ax.set_ylabel(error_type, fontsize=12)
        fig.savefig(os.path.join(model_name, "dist-by-%s.pdf"%error_type), bbox_inches='tight')
        plt.close("all")

    # for zscore, show 1/2 sd, 1 sd, and 2 sd
    error_types = ["e zscore", "f zscore"]
    for error_type in error_types:
        fig, ax = plt.figure(figsize=(6,4)), plt.gca()

        xlo, xhi = df['xtest_dist'].min(), df['xtest_dist'].max()
        for sig in [.5, 1., 2.]:
            ax.plot([xlo, xhi], [sig, sig], '--', linewidth=1.5, c='grey')
            ax.plot([xlo, xhi], [-sig, -sig], '--', linewidth=1.5, c='grey')
            ax.fill_between([xlo, xhi], [sig, sig], [-sig, -sig], color='grey', alpha=.2)

        ax.scatter(df['xtest_dist'], df[error_type], s=15)
        ax.set_xlabel("distance", fontsize=12)
        ax.set_ylabel(error_type, fontsize=12)
        fig.savefig(os.path.join(model_name, "dist-by-%s.pdf"%error_type), bbox_inches='tight')
        plt.close("all")


def make_zscore_histogram(model_name, pretty_name=None, target='f'):
    if pretty_name is None:
        pretty_name = os.path.split(model_name)[-1]
    df = make_error_dataframe([model_name], [pretty_name])

    colname = "%s zscore"%target
    nz = pd.isnull(df[colname])
    fig, ax = plt.figure(figsize=(6,4)), plt.gca()
    ax.hist(df[colname][~nz].values, bins=30, density=True, alpha=.5,
            label=pretty_name)

    from scipy.stats import norm
    xlim = ax.get_xlim()
    xlim = -3, 3
    xgrid = np.linspace(xlim[0], xlim[-1], 100)
    ax.plot(xgrid, norm.pdf(xgrid), label="$\mathcal{N}(0,1)$")

    ax.set_xlabel("z score", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.legend(fontsize=12, frameon=True, loc='upper left')
    ax.set_xlim(xlim)
    fig.savefig(os.path.join(model_name, "%s-zscore-histogram.pdf"%target),
                bbox_inches='tight')
    plt.close("all")


def make_coverage_table(model_names, pretty_names=None, target="f"):
    print(" ... coverage table for prediction type %s (in f or e)"%target)

    # all error
    if pretty_names is None:
        pretty_names = [os.path.split(m)[-1] for m in model_names]
    df = make_error_dataframe(model_names, pretty_names)

    def zscore_to_coverage_vec(zs, sigs=[.5, 1., 2, 3]):
        fracs = np.array([np.mean(np.abs(zs) < s) for s in sigs ])
        return fracs

    import pandas as pd
    sigs = [.5, 1., 2., 3.]
    zstd = [.382924, .682694, .954997, .997300]

    if target != 'fe':
        cname = "%s zscore"%target
        mdict = {pname: zscore_to_coverage_vec(
                            df[cname][df['model']==pname].values, sigs)
                 for pname in pretty_names}
        covdf = pd.DataFrame({**mdict, **{r"$\mathcal{N}(0, 1)$": zstd}},
                             index=sigs)
    else:
        cname_f = 'f zscore'
        mdict_f = {"{} f".format(pname): zscore_to_coverage_vec(
            df[cname_f][df['model'] == pname].values, sigs)
            for pname in pretty_names}
        cname_e = 'e zscore'
        mdict_e = {"{} e".format(pname): zscore_to_coverage_vec(
            df[cname_e][df['model'] == pname].values, sigs)
            for pname in pretty_names}
        covdf = pd.DataFrame({**mdict_f, **mdict_e, **{r"$\mathcal{N}(0, 1)$": zstd}},
                             index=sigs)

    #covdf = pd.DataFrame(
    #    {**{pname: zscore_to_coverage_vec(df[cname][df['model']==pname].values, sigs)
    #        for pname in pretty_names},
    #     **{r"$\mathcal{N}(0, 1)$": zstd}},
    #    index=sigs)
    covdf.index.name = r"$\sigma$"
    return covdf


def batch_callback(mod, xbatch, ybatch, noise_std_batch):
    raise NotImplementedError


def standard_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                            cuda_num, predict_maxiter_cg,
                            do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                            elbo_trace, save_model=True, save_trace=True, elbo=None,
                            sig2_list=None, ell_list=None, noisesq_list=None, return_pdict=False,
                            xvalid=None, fvalid=None, evalid=None):
    ftest_eval_time = fgrid_eval_time = etest_eval_time = egrid_eval_time = fvalid_eval_time = evalid_eval_time = None

    eval_valid = True if xvalid is not None else False

    if not os.path.exists(epoch_odir):
        os.makedirs(epoch_odir)
    ################################
    # save, predict, summary, plot #
    ################################

    # save model + trace information
    if save_model:
        torch.save(mod.state_dict(), os.path.join(epoch_odir, "model.pkl"))
    if save_trace:
        torch.save(elbo_trace, os.path.join(epoch_odir, "elbo_trace.pkl"))

    if elbo_trace is not None:
            fig, ax = plt.subplots(1, 1)
            ax.plot(elbo_trace, '-o')
            ax.set_title("elbo")
            plt.savefig(os.path.join(epoch_odir, "elbo.jpg"))
            plt.close()

    for name, ll in zip(['sig2', 'ell', 'noisesq'], [sig2_list, ell_list, noisesq_list]):
        if ll is not None:
            torch.save(ll, os.path.join(epoch_odir, "{}_trace.pkl".format(name)))
            fig, ax = plt.subplots(1, 1)
            ax.plot(ll, '-o')
            ax.set_title(name)
            plt.savefig(os.path.join(epoch_odir, "{}.jpg".format(name)))
            plt.close()

    do_cuda = torch.cuda.is_available()
    if do_cuda:
        print("Evaluating GP with CUDA!")
        print("device: cuda:{}".format(cuda_num))
        mod = mod.cuda_params(cuda_num)

    #####################
    # Make predictions  #
    #####################
    pdict = {}

    if eval_train:
        print("\nEvaluating on training set...")
        train_predict = mod.batch_predict(mod.torch(xobs), maxiter_cg=predict_maxiter_cg, batch_size=100,
                                          integrated_obs=do_integrated_predictions,
                                          semi_integrated_estimator=predict_ksemi_method,
                                          semi_integrated_samps=predict_ksemi_samples)
        train_predict_np = train_predict[0]
        print("max sq error", torch.max((train_predict_np - yobs) ** 2))
        print("max abs error", torch.max(np.abs(train_predict_np - yobs)))
        print("msq", torch.mean((train_predict_np - yobs) ** 2))

    if xvalid is not None:
        print("evaluating on valid set...")

        start = time.time()
        fmu, fsig = mod.batch_predict(mod.torch(xvalid),
                                      batch_size=100,
                                      integrated_obs=False,
                                      maxiter_cg=predict_maxiter_cg)
        fvalid_eval_time = time.time() - start
        print("Predictions on valid set takes {:.4f}".format(fvalid_eval_time))
        pdict['fmu_valid'] = fmu.detach().numpy().squeeze()
        pdict['fsig_valid'] = fsig.detach().numpy().squeeze()
        pdict['fvalid'] = fvalid
        pdict['xvalid'] = xvalid
        #pdict['xtest_dist'] = np.sqrt(np.sum(xtest ** 2, axis=-1))

        if do_integrated_predictions:
            start = time.time()
            print("Integrated predictions")
            emu, esig = mod.batch_predict(mod.torch(xvalid),
                                          batch_size=10,  # 100,
                                          integrated_obs=True,
                                          semi_integrated_estimator=predict_ksemi_method,
                                          semi_integrated_samps=predict_ksemi_samples,
                                          maxiter_cg=predict_maxiter_cg)
            evalid_eval_time = time.time() - start
            print("Integrated predictions on valid set takes {:.4f}".format(evalid_eval_time))
            print("  Integrated predictions: %d are nan" % np.sum(pd.isnull(esig)))
            pdict['emu_valid'] = emu.detach().numpy().squeeze()
            pdict['esig_valid'] = esig.detach().numpy().squeeze()
            pdict['evalid'] = evalid

    if xtest is not None:
        print("evaluating on test set...")

        start = time.time()
        fmu, fsig = mod.batch_predict(mod.torch(xtest),
                                      batch_size=100,
                                      integrated_obs=False,
                                      maxiter_cg=predict_maxiter_cg)
        ftest_eval_time = time.time() - start
        print("Predictions on test set takes {:.4f}".format(ftest_eval_time))
        pdict['fmu_test'] = fmu.detach().numpy().squeeze()
        pdict['fsig_test'] = fsig.detach().numpy().squeeze()
        pdict['ftest'] = ftest
        pdict['xtest'] = xtest
        pdict['xtest_dist'] = np.sqrt(np.sum(xtest ** 2, axis=-1))

        if do_integrated_predictions:
            start = time.time()
            print("Integrated predictions")
            emu, esig = mod.batch_predict(mod.torch(xtest),
                                          batch_size=10, #100,
                                          integrated_obs=True,
                                          semi_integrated_estimator=predict_ksemi_method,
                                          semi_integrated_samps=predict_ksemi_samples,
                                          maxiter_cg=predict_maxiter_cg)
            etest_eval_time = time.time() - start
            print("Integrated predictions on test set takes {:.4f}".format(etest_eval_time))
            print("  Integrated predictions: %d are nan" % np.sum(pd.isnull(esig)))
            pdict['emu_test'] = emu.detach().numpy().squeeze()
            pdict['esig_test'] = esig.detach().numpy().squeeze()
            pdict['etest'] = etest
            pdict['xtest'] = xtest

    if xgrid is not None:
        print("\nEvaluating on grid points...")
        start = time.time()
        fmu, fsig = mod.batch_predict(mod.torch(xgrid),
                                      batch_size=100,
                                      integrated_obs=False,
                                      maxiter_cg=predict_maxiter_cg)
        fgrid_eval_time = time.time() - start
        print("Predictions on grid set takes {:.4f}".format(fgrid_eval_time))
        pdict['fmu_grid'] = fmu.detach().numpy().squeeze()
        pdict['fsig_grid'] = fsig.detach().numpy().squeeze()
        pdict['fgrid'] = fgrid
        pdict['xgrid'] = xgrid

        """
        if do_integrated_predictions:
            print("Integrated predictions...")
            start = time.time()
            emu, esig = mod.batch_predict(mod.torch(xgrid), batch_size=50, # 100 
                                          integrated_obs=True,
                                          semi_integrated_estimator=predict_ksemi_method,
                                          semi_integrated_samps=predict_ksemi_samples,
                                          maxiter_cg=predict_maxiter_cg)
            egrid_eval_time = time.time() - start
            print("Integrated predictions on grid set takes {:.4f}".format(egrid_eval_time))
            pdict['emu_grid'] = emu.detach().numpy().squeeze()
            pdict['esig_grid'] = esig.detach().numpy().squeeze()
            pdict['egrid'] = egrid
        """

    if len(pdict) > 0:
        print("saving to ", epoch_odir)
        torch.save(pdict, os.path.join(epoch_odir, "predictions.pkl"))


    ##########################
    # Report error reduction #
    ##########################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if do_integrated_predictions:
            max_esig_grid = None
           #max_esig_grid = np.max(pdict['esig_grid'])
            max_esig_test = np.max(pdict['esig_test'])
        else:
            max_esig_grid = None
            max_esig_test = None

        noise_df = make_noise_comparison_dataframe(
            model_name=epoch_odir,
            dstd=np.sqrt((sobs ** 2).mean().item()),
            integrated_obs=do_integrated_predictions,
            train_elbo=elbo,
            max_fsig_grid=np.max(pdict.get('fsig_grid', None)),
            max_esig_grid=max_esig_grid,
            max_esig_test=max_esig_test,
            eval_valid=eval_valid)
        if do_integrated_predictions:
            covdf = make_coverage_table([epoch_odir], target='fe')
        else:
            covdf = make_coverage_table([epoch_odir], target='f')
    print("\nNoise Reduction DataFrame")
    print(noise_df)

    print("\ncoverage table")
    print(covdf.T)

    noise_df.to_csv((os.path.join(epoch_odir, "noise_reduction.csv")))
    covdf.to_csv(os.path.join(epoch_odir, "coverage_table.csv"))

    eval_time_tuples = (ftest_eval_time, etest_eval_time, fgrid_eval_time, egrid_eval_time,
                        fvalid_eval_time, evalid_eval_time)
    if return_pdict:
        return pdict, eval_time_tuples
    return eval_time_tuples

