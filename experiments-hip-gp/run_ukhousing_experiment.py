"""
UK Housing
    (1) tune params on small data subset
    (2) fit to entire dataset
    (3) compare log likelihood
"""

import argparse, os
import matplotlib.pyplot as plt
import warnings

from ziggy.misc.util import NumpyEncoder
import json
parser = argparse.ArgumentParser(description='UK Housing Experiment')
parser.add_argument('--exp-name', default='null', type=str)

parser.add_argument("--fit-models", action="store_true")
parser.add_argument("--full-model", action="store_true")
parser.add_argument("--block-model", action="store_true")
parser.add_argument("--mf-model", action="store_true")
parser.add_argument("--whitened-type", default='ziggy', type=str, help='type of whitened approach, '
                                                                        'choose from "ziggy" and "cholesky"')
parser.add_argument("--jitter-val", default=1e-3, type=float, help='jitter value added to the diagonal of kernel matrix')
parser.add_argument("--batch-solve", action="store_true", help="if not batch-solve, use natural gradient")

# kernel learning
parser.add_argument("--learn-kernel", action="store_true", help='currently, only capatible with cholesky whitening')
parser.add_argument("--kernel-lr", default=1e-3, type=float)

# natgrad parameters, used when not batch-solve
parser.add_argument("--batch-size", default=200, type=int, help="batch size in natural gradient descent")
parser.add_argument("--epochs", default=10, type=int, help="number of epochs to run natural gradient descent")
parser.add_argument("--lr", default=1e-2, type=float, help='learning rate for natural gradient descent')
parser.add_argument("--schedule-lr", action="store_true")
parser.add_argument("--step_decay", default=0.99, help="used when schedule-lr")
parser.add_argument("--print-debug-info", action="store_true")
parser.add_argument("--epoch_log_interval", default=1, type=int)
parser.add_argument("--batch_log_interval", default=1, type=int)  # print every X batches

# batch-solve-parameters, used whn batch-solve
parser.add_argument("--batch-solve-bsz", default=-1, type=int)

# cuda
parser.add_argument("--cuda-num", default=0, type=int)

# kernel
parser.add_argument("--kernel", default="Mat32", type=str,
    help="Mat12 | Mat32 | Mat52 | SqExp")
parser.add_argument("--sig2-init",    default=-1, type=float)  # 1e-4
parser.add_argument("--ell-min", default=0.1, type=float)
parser.add_argument("--ell-max", default=0.1, type=float)
parser.add_argument("--ell-nsteps", default=1, type=int)

# data
parser.add_argument("--nobs", default=-1, type=int)
parser.add_argument("--ntest", default=20000, type=int)

# model structure
parser.add_argument("--num-inducing-x", default=20, type=int)
parser.add_argument("--num-inducing-y", default=20, type=int)
parser.add_argument("--xblock-size", default=10, help="block size along x dimension", type=int)
parser.add_argument("--yblock-size", default=10, help="block size along y dimension", type=int)

# misc
parser.add_argument('--maxiter-cg', default=20, type=int)
parser.add_argument("--predict-maxiter-cg", default=50, type=int)
parser.add_argument('--eval-train', action='store_true')
parser.add_argument("--only-eval-last-epoch", action="store_true")

args, _ = parser.parse_known_args()
print("Experiment script args: ", args)

# experiment directory params

experiment_name = "kern={kern}-M={numinducex}x{numinducey}-maxitercg={maxitercg}-nobs={nobs}-ntest={ntest}".format(
    kern=args.kernel,
    numinducex=args.num_inducing_x,
    numinducey=args.num_inducing_y,
    maxitercg=args.maxiter_cg,
    nobs=args.nobs,
    ntest=args.ntest)
if args.exp_name != "null":
    experiment_name = "{exp_name}/{experiment_name}".format(args.exp_name, experiment_name)


#####################
# start script      #
#####################
import torch; torch.manual_seed(42)
import numpy as np
import numpy.random as npr; npr.seed(42)
from scipy.stats import norm
import pandas as pd

from ziggy.misc import experiment_util as eu
from ziggy.misc.util import add_date_time


output_dir = os.path.join(os.getcwd(), "output-uk", add_date_time(experiment_name))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#############
# Make Data #
#############
import uk_housing_data as ukdata
data_dict = ukdata.make_data_dict(Ntrain=args.nobs, Ntest=args.ntest, gridnum=256)#256)
print(" ---- NTrain = %d"%data_dict['xobs'].shape[0])

args.nobs = data_dict['xobs'].shape[0]
args.test = data_dict['xtest'].shape[0]


#############################################
# error summary and plotting functions      #
#############################################
def summarize_error(mod_name):
    """ load in predictions, compute MAE, RMSE, ll """
    predfile = os.path.join(output_dir, mod_name, 'predictions.pkl')
    pdict = torch.load(predfile)

    fmu_test, fsig_test = pdict['fmu_test'], pdict['fsig_test']
    pdf = pd.DataFrame({
        'fmu_test': fmu_test,
        'fsig_test': fsig_test,
        'ysig_test': np.sqrt(fsig_test**2 + stest.squeeze()**2),
        'ytest' : ytest.squeeze()})

    pdf['lly'] = norm.logpdf(pdf['ytest'], loc=pdf['fmu_test'], scale=pdf['ysig_test'])
    pdf['abs-diff'] = np.abs(pdf['ytest'] - pdf['fmu_test'])
    pdf['sq-diff']  = (pdf['ytest'] - pdf['fmu_test'])**2
    return pdf


def uk_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                            cuda_num, predict_maxiter_cg,
                            do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                            elbo_trace, save_model=True, save_trace=True, elbo=None,
                            sig2_list=None, ell_list=None, noisesq_list=None, xvalid=None, fvalid=None, evalid=None):
    pdict, eval_time_tuples = eu.standard_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid,
                                                         egrid,
                                                         cuda_num, predict_maxiter_cg,
                                                         do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                                                         elbo_trace, save_model, save_trace, elbo, sig2_list, ell_list, noisesq_list,
                                                         return_pdict=True, xvalid=xvalid, fvalid=fvalid, evalid=evalid)

    plot_map(pdict, epoch_odir)
    return eval_time_tuples


def plot_map(pdict, output_dir):

    xx1, xx2 = data_dict['xx1'], data_dict['xx2']
    vmin, vmax = data_dict['vmin'], data_dict['vmax']

    plt.ion()
    from ziggy import viz
    pdf, sdf = hdata.pricedf, hdata.shapedf

    fig, ax = plt.figure(figsize=(6,8)), plt.gca()
    sdf.plot(ax=ax, facecolor='none', edgecolor='black', alpha=.75)
    cm = viz.plot_smooth(ax, pdict['fmu_grid'].reshape(xx1.shape),
                         xlim=hdata.roi_xlim, ylim=hdata.roi_ylim)
                         #vmin=vmin, vmax=vmax)

    #viz.colorbar(cm, ax)
    xlo = hdata.roi_xlim[0]+.2
    xhi = hdata.roi_xlim[1]-.2
    ylo = hdata.roi_ylim[0]+.2
    yhi = hdata.roi_ylim[1]-.2
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("auto")
    ax.grid(False)
    #ax.scatter(xobs[:100,0], xobs[:100,1], s=1, label="train")
    fig.savefig(os.path.join(output_dir, "fmu.pdf"), bbox_inches='tight')
    #fig.savefig(os.path.join(output_dir, "fmu.png"), bbox_inches='tight', dpi=110)
    plt.close()

    fig, ax = plt.figure(figsize=(6,8)), plt.gca()
    sdf.plot(ax=ax, facecolor='none', edgecolor='black', alpha=.75)
    cm = viz.plot_smooth(ax, pdict['fsig_grid'].reshape(xx1.shape),
                     xlim=hdata.roi_xlim, ylim=hdata.roi_ylim)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("auto")
    #ax.scatter(xobs[:100,0], xobs[:100,1], s=1, label="train")
    ax.grid(False)
    fig.savefig(os.path.join(output_dir, "fsig.pdf"), bbox_inches='tight')
    #fig.savefig(os.path.join(output_dir, "fsig.png"), bbox_inches='tight', dpi=110)
    plt.close()


#################
# Fit models
#################
if args.fit_models:

    # unpack data
    xobs, yobs, sobs = \
        data_dict['xobs'], data_dict['yobs'], data_dict['sobs']
    xtest, ytest, stest = \
        data_dict['xtest'], data_dict['ytest'], data_dict['stest']
    hdata = data_dict['hdata']

    if args.sig2_init == -1:
        args.sig2_init = data_dict['yobs'].var() - data_dict['noise_std'] ** 2

    xlo, xhi = hdata.roi_xlim
    ylo, yhi = hdata.roi_ylim
    delta_x = (xhi-xlo) / args.num_inducing_x
    delta_y = (yhi - ylo) / args.num_inducing_y

    print('Estimated length scale x: ', 2 * delta_x)
    print('Estimated length scale y: ', 2 * delta_y)

    print("Nobs = {}, Ntest = {}".format(xobs.shape[0], xtest.shape[0]))
    print("xlim = {}, ylim = {}".format(hdata.roi_xlim, hdata.roi_ylim))  # (-5.7, 1.8), (50, 55.5)

    fit_kwargs = {'kernel'            : args.kernel,
                  'learn_kernel'      : args.learn_kernel,
                  'kernel_lr'         : args.kernel_lr,
                  'sig2_init_val': args.sig2_init,
                  'ell_min': args.ell_min,
                  'ell_max': args.ell_max,
                  'ell_nsteps': args.ell_nsteps,

                  'whitened_type': args.whitened_type,
                  'jitter_val': args.jitter_val,

                  'batch_size': args.batch_size,
                  'epochs': args.epochs,
                  'lr': args.lr,
                  'schedule_lr': args.schedule_lr,
                  'step_decay': args.step_decay,
                  'print_debug_info': args.print_debug_info,
                  'batch_log_interval': args.batch_log_interval,

                  'maxiter_cg': args.maxiter_cg,  # training maxiter
                  'predict_maxiter_cg': args.predict_maxiter_cg,

                  'do_cuda': torch.cuda.is_available(),
                  'cuda_num': args.cuda_num,

                  'batch_solve_bsz': args.batch_solve_bsz,

                  'integrated_obs': False,

                  'num_inducing_x': args.num_inducing_x,
                  'num_inducing_y': args.num_inducing_y,
                  'xblock_size'       : args.xblock_size,
                  'yblock_size'       : args.yblock_size,

                  'eval_train'        : args.eval_train,
                  'only_eval_last_epoch': args.only_eval_last_epoch,
                  'output_dir'        : output_dir}

    with open(os.path.join(output_dir, 'fit_params.json'), 'w') as file:
        json.dump(fit_kwargs, file, indent=4, cls=NumpyEncoder)

    ell_list = np.linspace(args.ell_min, args.ell_max, args.ell_nsteps)

    # set up inducing point grids
    xgrids = [torch.linspace(*hdata.roi_xlim, args.num_inducing_x),
              torch.linspace(*hdata.roi_ylim, args.num_inducing_y)]
    num_inducing_in_each_dim = [len(s) for s in xgrids]
    total_num_inducing = np.prod(num_inducing_in_each_dim)
    print("Total number of inducing points m1 x m2 = M: {} x {} = {}".format(
        num_inducing_in_each_dim[0], num_inducing_in_each_dim[1], total_num_inducing))

    # set up args for data
    data_kwargs = {'xobs': xobs[:,:],
                   'yobs': yobs[:,:],
                   'sobs': sobs[:,:],
                   'xinduce_grids' : xgrids,
                   'xtest'    : xtest,
                   'etest'    : None,
                   'ftest'    : ytest,
                   'xgrid'    : data_dict['xgrid'],
                   'fgrid'    : data_dict['fgrid'],
                   'init_Svar': .1}

    vi_mods = []
    if args.mf_model:
        vi_mods.append('mean-field')
    if args.block_model:
        vi_mods.append('block-diagonal-{}{}'.format(args.xblock_size, args.yblock_size))
    if args.full_model:
        vi_mods.append('full-rank')
    if total_num_inducing > 4000:  # 61*61 = 3721
        print(" removing full rank --- too many inducing points ")
        if 'full-rank' in vi_mods:
            vi_mods.remove("full-rank")

    full_model_names = []
    pretty_names = []
    for mclass in vi_mods:
        print("\n----------------------------------------------------")
        print("--- variational type: %s"%mclass)
        print("------------------------------------------------------")

        if mclass == "full-rank":
            fit_kwargs['fit_method'] = 'full-batch'
        else:
            fit_kwargs['fit_method'] = 'full-batch' if args.batch_solve else 'natgrad'

        for ell in ell_list:
            mod_name = "vi={}_ell={}".format(mclass, ell)
            if mclass == 'full-rank':
                full_model_names.append(os.path.join(output_dir, mod_name))
                pretty_names.append(mod_name)
            else:
                pretty_names += ["{}-epoch{}".format(mod_name, epoch_idx) for epoch_idx in range(args.epochs)]
                if args.only_eval_last_epoch:
                    full_model_names.append(os.path.join(output_dir, mod_name, "epoch{}".format(args.epochs-1)))
                else:
                    full_model_names += [os.path.join(output_dir, mod_name, "epoch{}".format(epoch_idx))
                                         for epoch_idx in range(args.epochs)]

            fit_kwargs['ell_init'] = ell
            mod = eu.svigp_fit_predict_and_save(
                name=mod_name,
                **data_kwargs,
                model_class=mclass,
                epoch_callback=uk_epoch_callback,
                **fit_kwargs)

        plt.close("all")


    #################
    # do some plots #
    #################
    if len(vi_mods) != 0:
        # error + loglike
        print("\nerror and loglike comparison")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = eu.make_error_dataframe(full_model_names, pretty_names=pretty_names)
    dfsum = df.groupby("model")[['f loglike', 'f mae', 'f mse']].mean()
    print(dfsum)
    #df.to_csv(os.path.join(output_dir, "errordf.csv"))
    dfsum.to_csv(os.path.join(output_dir, "errordf-summary.csv"))