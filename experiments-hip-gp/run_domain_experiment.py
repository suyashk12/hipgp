"""
Compares the results of two inference strategies with SVI
  - true analytic semi-integrated gradient
  - random grid version using MC samples
"""

import argparse, os
import json
import torch
import warnings

from ziggy.misc import experiment_util_domain as eud
from ziggy.misc import experiment_util as eu
from ziggy.misc.util import add_date_time
from ziggy.misc.util import NumpyEncoder

from exp_utils import domain_epoch_callback, plot_domain_true


parser = argparse.ArgumentParser(description='Synthetic Experiment')
parser.add_argument('--exp-name', default='null', type=str)

parser.add_argument("--fit-models", action="store_true")
parser.add_argument("--whitened-type", default='ziggy', type=str, help='type of whitened approach, '
                                                                        'choose from "ziggy" and "cholesky"')
parser.add_argument("--jitter-val", default=1e-3, type=float, help='jitter valued added to the diagonal of kernel matrix')
parser.add_argument("--full-model", action="store_true")
parser.add_argument("--block-model", action="store_true")
parser.add_argument("--mf-model", action="store_true")
parser.add_argument("--svgp", action="store_true")
parser.add_argument("--batch-solve", action='store_true')

# kernel learning
parser.add_argument("--learn-kernel", action="store_true", help='currently, only capatible with cholesky whitening')
parser.add_argument("--kernel-lr", default=1e-3, type=float)

# current not learning observation noise
#parser.add_argument("--learn-noise", action="store_true", help='whether or not to learn the observation noise')
#parser.add_argument("--noise-std-init", default=-1, type=float, help='initial value for noise std; if not provided, inferred from the data dict')

# natgrad parameters, used when not batch-solve
parser.add_argument("--batch-size", default=200, type=int, help="batch size in natural gradient descent")
parser.add_argument("--epochs", default=10, type=int, help="number of epochs to run natural gradient descent")
parser.add_argument("--lr", default=1e-2, type=float, help='learning rate for natural gradient descent')
parser.add_argument("--schedule-lr", action="store_true")
parser.add_argument("--step_decay", default=0.99, help="used when schedule-lr")
parser.add_argument("--print-debug-info", action="store_true")
parser.add_argument("--epoch-log-interval", default=1, type=int)
parser.add_argument("--batch-log-interval", default=1, type=int)  # print every X batches

# batch-solve-parameters, used whn batch-solve
parser.add_argument("--batch-solve-bsz", default=-1, type=int)


# cuda
parser.add_argument("--cuda-num", default=0, type=int)
parser.add_argument("--do-cuda", action="store_true")

# kernel
parser.add_argument("--kernel", default="Mat52", type=str,
    help="Mat12 | Mat32 | Mat52 | SqExp")
parser.add_argument("--sig2-init",    default=0.1, type=float)  # 1e-4
parser.add_argument("--ell-min", default=0.1, type=float)
parser.add_argument("--ell-max", default=0.1, type=float)
parser.add_argument("--ell-nsteps", default=1, type=int)

# data
parser.add_argument("--model_name", default='m12m_lsr0',
                    help="name of latte simulation", type=str)
parser.add_argument("--data-dir", default=None, type=str, help='path to domain data. Default to an exampler dataset in a 0.5x0.5x0.1kpc')
parser.add_argument("--xlo", default=-0.25,
                    help="minimum x and y extent in kpc", type=float)
parser.add_argument("--xhi", default=0.25,
                    help="maximum x and y extent in kpc", type=float)
parser.add_argument("--zlo", default=-0.05,
                    help="minimum z extent in kpc", type=float)
parser.add_argument("--zhi", default=0.05,
                    help="maximum z extent in kpc", type=float)
parser.add_argument("--nobs", default=10000,
                    help="number of observations", type=int)
parser.add_argument("--ntest", default=1000,
                    help="number of test samples", type=int)

# model structure
parser.add_argument("--num-inducing-x", default=15,
                    help="number of inducing points along the x and y direction", type=int)
parser.add_argument("--num-inducing-z", default=5,
                    help="number of inducing points along the z direction", type=int)
parser.add_argument("--xblock-size", default=2, help="block size along x dimension", type=int)
parser.add_argument("--zblock-size", default=2, help="block size along z dimension", type=int)

# misc
parser.add_argument('--maxiter-cg',   default=20, type=int)
parser.add_argument('--predict-maxiter-cg',   default=50, type=int)
parser.add_argument('--eval-train', action="store_true")
parser.add_argument("--only-eval-last-epoch", action="store_true")


args, _ = parser.parse_known_args()
print("Experiment script args: ", args)

model_param_name = 'kern={kern}-l={ell:.2f}-{ellmax:.2f}-M={indx}-{indz}-maxitercg={maxitercg}-nobs={nobs}-lr={lr}'.format(
    kern=args.kernel,
    ell=args.ell_min,
    ellmax=args.ell_max,
    indx=args.num_inducing_x,
    indz=args.num_inducing_z,
    maxitercg=args.maxiter_cg,
    nobs=args.nobs,
    lr=args.lr)

print("Model directory name: ", model_param_name)


import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
sns.set_context("paper")

#####################
# start script      #
#####################

import numpy as np
import numpy.random as npr; npr.seed(42)

if args.exp_name != 'null':
    output_dir = os.path.join('./output-domain-experiment', args.exp_name,
                              add_date_time(model_param_name))
else:
    output_dir = os.path.join("./output-domain-experiment", add_date_time(model_param_name))

if output_dir[:2] == "./":
    output_dir = os.path.join(os.getcwd(), output_dir[2:])


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

######################################################
# Make data, true function, etc                      #
######################################################

if args.data_dir is None:
    import git
    repo = git.Repo('.', search_parent_directories=True)
    repo_dir = repo.working_tree_dir  # hipgp
    args.data_dir = os.path.join(repo_dir, 'experiments-hip-gp', 'domain-data', 'domain_subsample.dat')

Data_kwargs = {
    'data_dir': args.data_dir,
    'Nobs'  : args.nobs,
    'Ntest' : args.ntest, #int(args.nobs/10.),
    'xlo'   : args.xlo,
    'xhi'   : args.xhi,
    'zlo'   : args.zlo,
    'zhi'   : args.zhi,
    'seed'  : 42,
    'noise_std': 0.005,
    'noise_mean': 0.005
    }


data_dict = eud.make_domain_data(**Data_kwargs)
print("Nobs = {}, Ntest = {}".format(data_dict['xobs'].shape[0], data_dict['xtest'].shape[0]))

xlo, xhi = data_dict['xlo'], data_dict['xhi']
zlo, zhi = data_dict['zlo'], data_dict['zhi']
xm_grid  = np.linspace(xlo, xhi, args.num_inducing_x)
zm_grid  = np.linspace(zlo, zhi, args.num_inducing_z)
xm1, xm2, xm3 = np.meshgrid(xm_grid, xm_grid, zm_grid)
xinduce  = np.column_stack([ xm1.flatten(), xm2.flatten(), xm3.flatten() ])

# report experiment details
print(" ---------- synthetic experiment --------- ")
for k, v in Data_kwargs.items():
    print("{0:10} : ".format(k), v)


plot_domain_true(data_dict, output_dir, alpha_value=1.0)


#################
# Fit Models    #
#################
if args.fit_models:
    # unpack data
    xobs, yobs, sobs = \
        data_dict['xobs'], data_dict['aobs'], data_dict['sobs']
    xtest, etest, ftest = \
        data_dict['xtest'], data_dict['etest'], data_dict['ftest']  # ftest is None
    xlo, xhi = data_dict['xlo'], data_dict['xhi']

    fit_kwargs = {'kernel': args.kernel,
                  'learn_kernel': args.learn_kernel,
                  'kernel_lr': args.kernel_lr,
                  'sig2_init_val': args.sig2_init,  # sig_init**2,
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

                  'do_cuda': args.do_cuda,
                  'cuda_num': args.cuda_num,

                  'batch_solve_bsz': args.batch_solve_bsz,

                  'integrated_obs': True,  # whether to integrate obs in training
                  'do_integrated_predictions': True,

                  'num_inducing_x': args.num_inducing_x,
                  'num_inducing_z': args.num_inducing_z,

                  'xblock_size': args.xblock_size,
                  'yblock_size': args.xblock_size,  # note we set xblock_size = yblock_size
                  'zblock_size': args.zblock_size,

                  'eval_train': args.eval_train,
                  'only_eval_last_epoch': args.only_eval_last_epoch,
                  'output_dir': output_dir,

                  'nobs': args.nobs,
                  'ntest': args.ntest}

    with open(os.path.join(output_dir, 'fit_params.json'), 'w') as file:
        json.dump(fit_kwargs, file, indent=4, cls=NumpyEncoder)

    # set up inducing point grids
    xinduce_grids = [torch.linspace(xlo, xhi, args.num_inducing_x),
                     torch.linspace(xlo, xhi, args.num_inducing_x),
                     torch.linspace(zlo, zhi, args.num_inducing_z)]
    num_inducing_in_each_dim = [len(s) for s in xinduce_grids]
    total_num_inducing = np.prod(num_inducing_in_each_dim)
    print("Total number of inducing points m1 x m2 x m3 = M: {} x {} x {} = {}".format(
        num_inducing_in_each_dim[0], num_inducing_in_each_dim[1], num_inducing_in_each_dim[2], total_num_inducing))

    if args.ell_nsteps == 1:
        ell_list = [args.ell_min]
    else:
        ell_list = np.linspace(args.ell_min, args.ell_max, args.ell_nsteps)


    # set up args for data
    data_kwargs = {'xobs': xobs[:, :],
                   'yobs': yobs[:, None],
                   'sobs': sobs[:, None],
                   'xinduce_grids': xinduce_grids,
                   'xtest': xtest,
                   'etest': etest,
                   'ftest': ftest,
                   'xgrid': data_dict['xgrid'],
                   'init_Svar': .1}

    xtest = data_dict['xtest']
    xgrid = data_dict['xgrid']
    xlo, xhi = data_dict['xlo'], data_dict['xhi']

    vi_mods = []
    if args.mf_model:
        vi_mods.append('mean-field')
    if args.block_model:
        vi_mods.append('block-diagonal-{x}{z}'.format(x=args.xblock_size, z=args.zblock_size))
    if args.full_model:
        vi_mods.append('full-rank')
    if args.svgp:
        vi_mods.append('SVGP')
    if total_num_inducing > 6000:  # 30*30*5=4500
        print(" removing full rank --- too many inducing points ")
        if 'full-rank' in vi_mods:
            vi_mods.remove("full-rank")

    full_model_names = []
    pretty_names = []
    for mclass in vi_mods:
        print("\n----------------------------------------------------")
        print("--- variational type: %s" % mclass)
        print("------------------------------------------------------")

        if mclass == "full-rank":
            fit_kwargs['fit_method'] = 'full-batch'
        else:
            fit_kwargs['fit_method'] = 'full-batch' if args.batch_solve else 'natgrad'

        for ell in ell_list:
            mod_name = "vi={}_ell={:.3f}".format(mclass, ell)
            fit_kwargs['ell_init'] = ell
            if mclass == 'full-rank':
                full_model_names.append(os.path.join(output_dir, mod_name))
                pretty_names.append(mod_name)
            else:
                full_model_names += [os.path.join(output_dir, mod_name, "epoch{}".format(epoch_idx))
                                     for epoch_idx in range(args.epochs)]
                pretty_names += ["{}-epoch{}".format(mod_name, epoch_idx) for epoch_idx in range(args.epochs)]
            mod = eu.svigp_fit_predict_and_save(
                name=mod_name,
                **data_kwargs,
                model_class=mclass,
                epoch_callback=domain_epoch_callback,
                **fit_kwargs)

    if len(vi_mods) != 0:
        # error + loglike
        print("\nerror and loglike comparison")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = eu.make_error_dataframe(full_model_names, pretty_names=pretty_names)
        dfsum = df.groupby("model")[['e loglike', 'e mae', 'e mse']].mean()
        print(dfsum)
        # df.to_csv(os.path.join(output_dir, "errordf.csv"))
        dfsum.to_csv(os.path.join(output_dir, "errordf-summary.csv"))






