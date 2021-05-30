"""
Sweep over all variables of synthetic data experiment

  - function-complexity : 'simple | medium | hard'
  - kernel_fun              : Mat12 | Mat32 | Mat52 | SqExp
  - num-obs             : N data
  - epochs              : 20
  - batch-size          : default = 200
  - maxiter-cg          : default = 2

... takes a long effing time to run.
"""

import argparse, os
import json
import warnings
from ziggy.misc.util import NumpyEncoder

parser = argparse.ArgumentParser(description='Synthetic 2d Experiment')
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
parser.add_argument("--learn-kernel", action="store_true", help='whether or not to learn the kernel hyper-parameters')
parser.add_argument("--kernel-lr", default=1e-3, type=float)

# observation noise learning
parser.add_argument("--learn-noise", action="store_true", help='whether or not to learn the observation noise')
parser.add_argument("--noise-std-init", default=-1, type=float, help='initial value for noise std; if not provided, inferred from the data dict')

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
parser.add_argument("--do-cuda", action="store_true")

# kernel
parser.add_argument("--kernel", default="Mat52", type=str,
    help="Mat12 | Mat32 | Mat52 | SqExp")
parser.add_argument("--sig2-init", default=None, type=float)
parser.add_argument("--ell-min", default=0.01, type=float)
parser.add_argument("--ell-max", default=1.0, type=float)
parser.add_argument("--ell-nsteps", default=3, type=int)

# data
parser.add_argument("--function-complexity", default="hard", type=str,
    help="simple|medium|hard (synthetic) or housing --- underlying func")
parser.add_argument("--num-obs",      default=20000, type=int)
parser.add_argument("--num-test",      default=2000, type=int)

# model structure
parser.add_argument("--num-inducing", default=125, type=int)
parser.add_argument("--xblock-size",  default=10, type=int,
    help="block size (along one dimension) for block-diagonal approx")


# misc
parser.add_argument('--maxiter-cg',   default=20, type=int)
parser.add_argument("--predict-maxiter-cg", default=50, type=int)
parser.add_argument("--output-dir", default="./output-synthetic", type=str)
parser.add_argument('--eval-train', action='store_true')
parser.add_argument("--only-eval-last-epoch", action="store_true")

args, _ = parser.parse_known_args()
print("Experiment script args: ", args)

# experiment directory params


if args.exp_name != 'null':
    experiment_name = "{exp}/fun={func}/kern={kern}-l={ell:.3f}-{ellmax:.3f}-M={numinduce}-maxitercg={maxitercg}-nobs={nobs}".format(
        exp       = args.exp_name,
        func      = args.function_complexity,
        kern      = args.kernel,
        ell       = args.ell_min,
        ellmax    = args.ell_max,
        numinduce = args.num_inducing,
        maxitercg = args.maxiter_cg,
        nobs      = args.num_obs)
else:
    experiment_name = "fun={func}/kern={kern}-l={ell:.3f}-{ellmax:.3f}-M={numinduce}-maxitercg={maxitercg}-nobs={nobs}".format(
        func=args.function_complexity,
        kern=args.kernel,
        ell=args.ell_min,
        ellmax=args.ell_max,
        numinduce=args.num_inducing,
        maxitercg=args.maxiter_cg,
        nobs=args.num_obs)

#####################
# start script      #
#####################
import torch; torch.manual_seed(42)
import numpy as np
import numpy.random as npr; npr.seed(42)
from ziggy.misc import experiment_util as eu
from ziggy import kernels as zkern
from ziggy.misc.util import add_date_time


output_dir = os.path.join(args.output_dir, add_date_time(experiment_name))
if output_dir[:2] == "./":
    output_dir = os.path.join(os.getcwd(), output_dir[2:])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



#################
# Make Data     #
#################
import synthetic_data as synth
data_kwargs = {'Nobs':  args.num_obs,
               'Ntest': args.num_test, #2000,
               'noise_std': .1,
               'xlo':   -1,
               'xhi':    1,
               'function_complexity': args.function_complexity,
               'gridnum': 50} # 256}
data_dict = synth.make_two_dim_data(**data_kwargs)

print("xobs average", data_dict['xobs'].mean())
print("yobs average", data_dict['yobs'].mean())
print("sobs average", data_dict['sobs'].mean())
print("init sig:", np.sqrt(data_dict['yobs'].var()-data_dict['noise_std']**2))

# plot and save function
import matplotlib.pyplot as plt;
fig, ax = synth.plot_synthetic_data(data_dict)
fig.savefig(os.path.join(output_dir,
    "true-fgrid-%s.pdf"%args.function_complexity), bbox_inches='tight')
plt.close("all")


"""
def synthetic_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                            cuda_num, predict_maxiter_cg,
                            do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                            elbo_trace, save_model=True, save_trace=True, elbo=None):
    _, eval_time_tuples = eu.standard_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                            cuda_num, predict_maxiter_cg,
                            do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                            elbo_trace, save_model, save_trace, elbo)
    return eval_time_tuples
"""
from exp_utils import synthetic_epoch_callback

#################
# Fit Models    #
#################
if args.fit_models:


    #data_dict["xobs"] = np.array([[0.0, 0.0]])
    #data_dict["yobs"] = np.array([[1.0]])

    # unpack data
    xobs, yobs, sobs = \
        data_dict['xobs'], data_dict['yobs'], data_dict['sobs']
    xtest, etest, ftest = \
        data_dict['xtest'], data_dict['etest'], data_dict['ftest']
    xlo, xhi = data_dict['xgrid'].min(), data_dict['xgrid'].max()
    print("xlo = ", xlo, " xhi = ", xhi)

    print("Nobs = {}, Ntest = {}".format(xobs.shape[0], xtest.shape[0]))

    if args.sig2_init is None:
        args.sig2_init = data_dict['yobs'].var()-data_dict['noise_std']**2
        print("Calculated from data, sig2_init_val = {}".format(args.sig2_init))
    else:
        print("sig2 init = {}".format(args.sig2_init))

    """
    if args.ell_init == -1:
        args.ell_init = {'simple': .25,
                         'medium': .08,
                         'hard': .05}[args.function_complexity]
    """

    # set up fit args
    #sig_init = np.sqrt(data_dict['yobs'].var()-data_dict['noise_std']**2)
    #sig_init = data_dict['noise_std']
    if args.noise_std_init == -1:
        args.noise_std_init = data_dict['noise_std']
    fit_kwargs = {'kernel'            : args.kernel,
                  'learn_kernel'      : args.learn_kernel,
                  'kernel_lr'         : args.kernel_lr,
                  'sig2_init_val': args.sig2_init,  # sig_init**2,
                  'ell_min': args.ell_min,
                  'ell_max': args.ell_max,
                  'ell_nsteps': args.ell_nsteps,

                  'learn_noise'       : args.learn_noise,
                  'noise_std_init'    : args.noise_std_init,

                  'whitened_type': args.whitened_type,
                  'jitter_val': args.jitter_val,

                  'batch_size': args.batch_size,
                  'epochs': args.epochs,
                  'lr' : args.lr,
                  'schedule_lr': args.schedule_lr,
                  'step_decay': args.step_decay,
                  'print_debug_info': args.print_debug_info,
                  'batch_log_interval': args.batch_log_interval,

                  'maxiter_cg'        : args.maxiter_cg,  # training maxiter
                  'predict_maxiter_cg': args.predict_maxiter_cg,

                  'do_cuda'           : args.do_cuda,
                  'cuda_num'          : args.cuda_num,

                  'batch_solve_bsz'   : args.batch_solve_bsz,

                  'integrated_obs'    : False,

                  'num_inducing'      : args.num_inducing,
                  'xblock_size'       : args.xblock_size,
                  'yblock_size'       : args.xblock_size, # NOTE: yblock_size = xblock_size for this experiment

                  'eval_train'        : args.eval_train,
                  'only_eval_last_epoch': args.only_eval_last_epoch,
                  'output_dir'        : output_dir,}

    with open(os.path.join(output_dir, 'fit_params.json'), 'w') as file:
        json.dump(fit_kwargs, file, indent=4, cls=NumpyEncoder)

    ell_list = np.linspace(args.ell_min, args.ell_max, args.ell_nsteps)

    # set up inducing point grids
    xinduce_grids = [torch.linspace(xlo, xhi, args.num_inducing),
                     torch.linspace(xlo, xhi, args.num_inducing)]
    num_inducing_in_each_dim = [len(s) for s in xinduce_grids]
    total_num_inducing = np.prod(num_inducing_in_each_dim)
    print("Total number of inducing points m1 x m2 = M: {} x {} = {}".format(
        num_inducing_in_each_dim[0], num_inducing_in_each_dim[1], total_num_inducing))

    #delta = (xhi-xlo) / args.num_inducing
    #fit_kwargs['ell_init'] = ell_init #np.max([2*delta, ell_init])

    # set up args for data
    data_kwargs = {'xobs': xobs[:,:],
                   'yobs': yobs[:,:],
                   'sobs': sobs[:,None],
                   'xinduce_grids' : xinduce_grids,
                   'xtest'    : xtest,
                   'etest'    : etest,
                   'ftest'    : ftest,
                   'xgrid'    : data_dict['xgrid'],
                   'fgrid'    : data_dict['fgrid'],
                   'init_Svar': .1}

    vi_mods = []
    if args.mf_model:
        vi_mods.append('mean-field')
    if args.block_model:
        vi_mods.append('block-diagonal-{}'.format(args.xblock_size))
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
                full_model_names += [os.path.join(output_dir, mod_name, "epoch{}".format(epoch_idx))
                                     for epoch_idx in range(args.epochs)]
            fit_kwargs['ell_init'] = ell
            mod = eu.svigp_fit_predict_and_save(
                name=mod_name,
                **data_kwargs,
                model_class=mclass,
                epoch_callback=synthetic_epoch_callback,
                **fit_kwargs)

    if len(vi_mods) != 0:
        # error + loglie
        print("\nerror and loglike comparison")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = eu.make_error_dataframe(full_model_names, pretty_names=pretty_names)
        dfsum = df.groupby("model")[['f loglike', 'f mae', 'f mse']].mean()
        print(dfsum)
        #df.to_csv(os.path.join(output_dir, "errordf.csv"))
        dfsum.to_csv(os.path.join(output_dir, "errordf-summary.csv"))


