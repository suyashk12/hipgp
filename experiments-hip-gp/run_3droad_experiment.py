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
parser = argparse.ArgumentParser(description='UCI Experiment')
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
parser.add_argument("--learn-kernel", action="store_true", help='whether or not to learn kernel hyper-parameters')
parser.add_argument("--kernel-lr", default=1e-3, type=float)

# observation noise
parser.add_argument("--learn-noise", action="store_true", help='whether or not to learn observation noise')
parser.add_argument("--noise-std-init", default=0.01, type=float)

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
parser.add_argument("--sig2-init",    default=0.1, type=float)  # 1e-4
parser.add_argument("--ell-min", default=0.1, type=float)
parser.add_argument("--ell-max", default=0.1, type=float)
parser.add_argument("--ell-nsteps", default=1, type=int)

# data -- 64 / 16 / 20 split
parser.add_argument("--nobs", default=238319, type=int)
parser.add_argument("--nvalid", default=69580, type=int)
parser.add_argument("--ntest", default=86975, type=int)
parser.add_argument("--eval-valid", action="store_true")
parser.add_argument("--eval-grid", action="store_true")
parser.add_argument("--gridnum", default=256, type=int)
parser.add_argument("--data-split-seed", default=42, type=int)

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
if args.exp_name != "null":
    experiment_name = "{expname}/kern={kern}-M={numinducex}x{numinducey}-maxitercg={maxitercg}-nobs={nobs}-ntest={ntest}".format(
        expname = args.exp_name,
        kern=args.kernel,
        numinducex=args.num_inducing_x,
        numinducey=args.num_inducing_y,
        maxitercg=args.maxiter_cg,
        nobs=args.nobs,
        ntest=args.ntest,
    )
else:
    experiment_name = "kern={kern}-M={numinducex}x{numinducey}-maxitercg={maxitercg}-nobs={nobs}-ntest={ntest}".format(
        kern=args.kernel,
        numinducex=args.num_inducing_x,
        numinducey=args.num_inducing_y,
        maxitercg=args.maxiter_cg,
        nobs=args.nobs,
        ntest=args.ntest,
    )


#####################
# start script      #
#####################
import torch; torch.manual_seed(42)
import numpy.random as npr; npr.seed(42)

from ziggy.misc import experiment_util as eu
from ziggy.misc.util import add_date_time


output_dir = os.path.join(os.getcwd(), "output-3droad", add_date_time(experiment_name))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#############
# Make Data #
#############
from exp_utils import load_uci_data
data_dict = load_uci_data('./uci-data', '3droad_standardized.txt', nobs=args.nobs, nvalid=args.nvalid, ntest=args.ntest,
                          eval_valid=args.eval_valid, eval_grid=args.eval_grid, gridnum=args.gridnum,
                          noise_std=args.noise_std_init, seed=args.data_split_seed)
print(" ---- NTrain = %d"%data_dict['xobs'].shape[0])

##################
# Plot test data #
##################
import numpy as np
ymin, ymax = data_dict['ytest'].min(), data_dict['ytest'].max()
if len(data_dict['xtest']) > 10000:
    plot_idx = np.random.choice(len(data_dict['xtest']), 10000)
    xtest_to_plot = data_dict['xtest'][plot_idx]
    ytest_to_plot = data_dict['ytest'][plot_idx]
else:
    xtest_to_plot = data_dict['xtest']
    ytest_to_plot = data_dict['ytest']
fig, ax = plt.figure(figsize=(5,5)), plt.gca()
im = ax.scatter(xtest_to_plot[:,0], xtest_to_plot[:,1], c=ytest_to_plot.squeeze(), alpha=0.6, s=10, vmin=ymin, vmax=ymax)
ax.set_title("ytest")
fig.tight_layout()
fig.colorbar(im, ax=ax)
plt.savefig(os.path.join(output_dir, "ytest.png"))
plt.close()


def uci_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                            cuda_num, predict_maxiter_cg,
                            do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                            elbo_trace, save_model=True, save_trace=True, elbo=None,
                            sig2_list=None, ell_list=None, noisesq_list=None,
                            xvalid=None, fvalid=None, evalid=None):
    pdict, eval_time_tuples = eu.standard_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                                cuda_num, predict_maxiter_cg,
                                do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                                elbo_trace, save_model=save_model, save_trace=save_trace, elbo=elbo,
                                sig2_list=sig2_list, ell_list=ell_list, noisesq_list=noisesq_list, return_pdict=True,
                                xvalid=xvalid, fvalid=fvalid, evalid=evalid)
    xtest, fmu_test, fsig_test, ytest = pdict['xtest'], pdict['fmu_test'], pdict['fsig_test'], pdict['ftest']
    ymin, ymax = ytest.min(), ytest.max()
    residual = np.abs(fmu_test - ytest.squeeze())

    if len(xtest) > 10000:
        plot_idx = np.random.choice(len(xtest), 10000)
        xtest = xtest[plot_idx]
        fmu_test = fmu_test[plot_idx]
        fsig_test = fsig_test[plot_idx]
        residual = residual[plot_idx]

    fig, ax = plt.figure(figsize=(5, 5)), plt.gca()
    im = ax.scatter(xtest[:, 0], xtest[:, 1], c=fmu_test, alpha=0.6, s=10, vmin=ymin, vmax=ymax)
    ax.set_title("fmu-test")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(epoch_odir, "fmu-test.png"))
    plt.close()

    fig, ax = plt.figure(figsize=(5, 5)), plt.gca()
    im = ax.scatter(xtest[:, 0], xtest[:, 1], c=fsig_test.squeeze(), alpha=0.6, s=10)
    ax.set_title("fsig-test")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(epoch_odir, "fsig-test.png"))
    plt.close()

    fig, ax = plt.figure(figsize=(5, 5)), plt.gca()
    im = ax.scatter(xtest[:, 0], xtest[:, 1], c=residual, alpha=0.6, s=10)
    ax.set_title("fres-test")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(epoch_odir, "fres-test.png"))
    plt.close()

    return eval_time_tuples


#################
# Fit models
#################
if args.fit_models:

    if args.learn_kernel:
        assert args.whitened_type == 'cholesky'

    # unpack data
    xobs, yobs, sobs = data_dict['xobs'], data_dict['yobs'], data_dict['sobs']
    xtest, ytest = data_dict['xtest'], data_dict['ytest']
    xvalid, yvalid = data_dict['xvalid'], data_dict['yvalid']

    xlo, xhi, ylo, yhi = data_dict['xlo'], data_dict['xhi'], data_dict['ylo'], data_dict['yhi']

    delta_x = (xhi - xlo) / args.num_inducing_x
    delta_y = (yhi - ylo) / args.num_inducing_y

    print("xlo = {:.2f}, xhi = {:.2f}, ylo = {:.2f}, yhi = {:.2f}".format(xlo, xhi, ylo, yhi))
    print("xobs-1 lo = {:.2f}, hi = {:.2f}, xobs-2 lo = {:.2f}, hi = {:.2f}".format(
        xobs[:,0].min(), xobs[:,0].max(), xobs[:,1].min(), xobs[:,1].max()))
    print('theoretic length scale x: ', 2 * delta_x)
    print('theoretic length scale y: ', 2 * delta_y)

    print("Nobs = {}, Ntest = {}".format(xobs.shape[0], xtest.shape[0]))

    fit_kwargs = {'kernel'            : args.kernel,
                  'learn_kernel'      : args.learn_kernel,
                  'kernel_lr'         : args.kernel_lr,
                  'sig2_init_val': args.sig2_init,
                  'ell_min': args.ell_min,
                  'ell_max': args.ell_max,
                  'ell_nsteps': args.ell_nsteps,

                  'learn_noise': args.learn_noise,
                  'noise_std_init': args.noise_std_init,

                  'whitened_type': args.whitened_type,
                  'jitter_val': args.jitter_val,

                  # data
                  'nobs': args.nobs,
                  'nvalid': args.nvalid,
                  'ntest': args.ntest,
                  'eval_valid': args.eval_valid,
                  'data_split_seed': args.data_split_seed,

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
    xgrids = [torch.linspace(xlo, xhi, args.num_inducing_x),
              torch.linspace(ylo, yhi, args.num_inducing_y)]
    assert xobs[:, 0].min() > xlo and xobs[:, 0].max() < xhi, \
        "xlo = {}, xhi = {}, bug got xobs-0 min = {}, max = {}".format(xlo, xhi, xobs[:,0].min(), xobs[:,0].max())
    assert xtest[:, 0].min() > xlo and xtest[:, 0].max() < xhi, \
        "xlo = {}, xhi = {}, bug got xtest-0 min = {}, max = {}".format(xlo, xhi, xtest[:, 0].min(), xtest[:, 0].max())
    assert xobs[:, 1].min() > ylo and xobs[:, 1].max() < yhi, \
        "ylo = {}, yhi = {}, bug got xobs-1 min = {}, max = {}".format(xlo, xhi, xobs[:, 1].min(), xobs[:, 1].max())
    assert xtest[:, 1].min() > ylo and xtest[:, 1].max() < yhi, \
        "ylo = {}, yhi = {}, bug got xtest-1 min = {}, max = {}".format(xlo, xhi, xtest[:, 1].min(), xtest[:, 1].max())

    num_inducing_in_each_dim = [len(s) for s in xgrids]
    total_num_inducing = np.prod(num_inducing_in_each_dim)
    print("Total number of inducing points m1 x m2 = M: {} x {} = {}".format(
        num_inducing_in_each_dim[0], num_inducing_in_each_dim[1], total_num_inducing))

    # set up args for data
    data_kwargs = {'xobs': xobs,
                   'yobs': yobs,
                   'sobs': sobs,
                   'xinduce_grids' : xgrids,
                   'xtest'    : xtest,
                   'etest'    : None,
                   'ftest'    : ytest,
                   'xvalid'   : xvalid,
                   'evalid'   : None,
                   'fvalid'   : yvalid,
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
            if args.batch_solve:
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
                epoch_callback=uci_epoch_callback,
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
            df = eu.make_error_dataframe(full_model_names, pretty_names=pretty_names, data_type='test')
            if args.eval_valid:
                df_valid = eu.make_error_dataframe(full_model_names, pretty_names=pretty_names, data_type='valid')

                dfsum = df.groupby("model")[['f loglike', 'f mae', 'f mse']].mean()
                print("Validation set ...")
                print(dfsum)
                dfsum.to_csv(os.path.join(output_dir, "errordf-valid-summary.csv"))

            print("\nTest set ...")
            dfsum = df.groupby("model")[['f loglike', 'f mae', 'f mse']].mean()
            print(dfsum)
            dfsum.to_csv(os.path.join(output_dir, "errordf-test-summary.csv"))