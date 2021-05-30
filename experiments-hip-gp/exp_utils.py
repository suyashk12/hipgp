import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from ziggy import viz
from ziggy.misc.experiment_util import standard_epoch_callback, plot_posterior_grid


def turn_off_ticks(ax):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)


def plot_val_at_zslices(xval, val, zidx_slices, select_idx_list, vmin_list, vmax_list, name, odir='./'):
    fig, axes = plt.subplots(2, 5, figsize=(6 * 5, 6 * 2))
    for i, ax in enumerate(axes[0]):
        vmin, vmax = vmin_list[i], vmax_list[i]
        z_start, z_end = zidx_slices[i]
        viz.ax_scatter(fig, ax, xval[select_idx_list[i]][:, 0], xval[select_idx_list[i]][:, 1],
                       val[select_idx_list[i]],
                       'x', 'y', title='z in [{:.1f}, {:.1f}]'.format(z_start, z_end),
                       vmin=vmin, vmax=vmax)
    for i2, ax in enumerate(axes[1]):
        i = i2 + 5
        vmin, vmax = vmin_list[i], vmax_list[i]
        z_start, z_end = zidx_slices[i]
        viz.ax_scatter(fig, ax, xval[select_idx_list[i]][:, 0], xval[select_idx_list[i]][:, 1],
                   val[select_idx_list[i]],
                   'x', 'y', title='z in [{:.1f}, {:.1f}]'.format(z_start, z_end),
                   vmin=vmin, vmax=vmax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("{}".format(name), fontsize=20)
    plt.savefig(os.path.join(odir, "{}".format(name)))
    plt.close()


def domain_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                            cuda_num, predict_maxiter_cg,
                            do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                            elbo_trace, save_model=True, save_trace=True, elbo=None,
                            sig2_list=None, ell_list=None, noisesq_list=None,
                            xvalid=None, fvalid=None, evalid=None,
                            ):

    pdict, eval_time_tuples = standard_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                                                      cuda_num, predict_maxiter_cg,
                                                      do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                                                      elbo_trace, save_model, save_trace, elbo, sig2_list, ell_list, noisesq_list,
                                                      xvalid=xvalid, fvalid=fvalid, evalid=evalid,
                                                      return_pdict=True)

    plot_domain_rslt(epoch_odir, pdict)
    return eval_time_tuples


def plot_domain_rslt(odir, pdict):
    # load_dict
    xtest = pdict['xtest']
    ftest, fmu_test, fsig_test = pdict['ftest'], pdict['fmu_test'], pdict['fsig_test']
    etest, emu_test, esig_test = pdict['etest'], pdict['emu_test'], pdict['esig_test']
    fres_test = fmu_test - ftest
    eres_test = emu_test - etest

    ########################
    #   Prediction on f    #
    ########################

    fig = plt.figure(figsize=(8, 4))
    ax = Axes3D(fig)
    im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=fmu_test, s=20)
    turn_off_ticks(ax)
    fig.colorbar(im)
    plt.title("emu-test")
    fig.savefig(os.path.join(odir, "predict-fmu-test-3D.png"))
    plt.close()

    fig = plt.figure(figsize=(8, 4))
    ax = Axes3D(fig)
    im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=fsig_test, s=20)
    fig.colorbar(im)
    plt.title("esig-test")
    fig.savefig(os.path.join(odir, "predict-fsig-test-3D.png"))
    plt.close()

    ####################
    #  Prediction on e #
    ####################
    # 3d plot
    fig = plt.figure(figsize=(8, 4))
    ax = Axes3D(fig)
    im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=emu_test, s=20)
    turn_off_ticks(ax)
    fig.colorbar(im)
    plt.title("emu-test")
    fig.savefig(os.path.join(odir, "predict-emu-test-3D.png"))
    plt.close()

    fig = plt.figure(figsize=(8, 4))
    ax = Axes3D(fig)
    im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=esig_test, s=20)
    fig.colorbar(im)
    plt.title("esig-test")
    fig.savefig(os.path.join(odir, "predict-esig-test-3D.png"))
    plt.close()

def plot_domain_true(data_dict, output_dir, alpha_value=0.3):

    fig = plt.figure(figsize=(8, 4))
    ax = Axes3D(fig)
    turn_off_ticks(ax)
    im = ax.scatter(data_dict['xtest'][:, 0], data_dict['xtest'][:, 1], data_dict['xtest'][:, 2], c=data_dict['etest'],
                    s=20, alpha=alpha_value)
    fig.colorbar(im)
    plt.title("true etest")
    fig.savefig(os.path.join(output_dir, "true-etest-3D.png"))
    plt.close()

    # plot process data
    fig = plt.figure(figsize=(8, 4))
    ax = Axes3D(fig)
    turn_off_ticks(ax)
    im = ax.scatter(data_dict['xtest'][:, 0], data_dict['xtest'][:, 1], data_dict['xtest'][:, 2], c=data_dict['ftest'],
                    s=20, alpha=alpha_value)
    fig.colorbar(im)
    plt.title("true ftest")
    fig.savefig(os.path.join(output_dir, "true-ftest-3D.png"))
    plt.close()


def synthetic_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                            cuda_num, predict_maxiter_cg,
                            do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                            elbo_trace, save_model=True, save_trace=True, elbo=None, xvalid=None, fvalid=None, evalid=None, **kwargs):

    pdict, eval_time_tuples = standard_epoch_callback(epoch_odir, mod, eval_train, xobs, yobs, sobs, xtest, ftest, etest, xgrid, fgrid, egrid,
                                                      cuda_num, predict_maxiter_cg,
                                                      do_integrated_predictions, predict_ksemi_method, predict_ksemi_samples,
                                                      elbo_trace, save_model, save_trace, elbo, xvalid=xvalid, fvalid=fvalid, evalid=evalid,
                                                      return_pdict=True, **kwargs)

    plot_posterior_grid(pdict, epoch_odir, ticklabels=False)
    return eval_time_tuples


def load_uci_data(data_dir, dataset, nobs, nvalid, ntest, eval_valid, eval_grid, gridnum=256, noise_std=0.05, seed=42):
    rs = np.random.RandomState(seed=seed)

    data = np.loadtxt(os.path.join(data_dir, dataset), dtype=np.float64)
    total_num = len(data)

    xlo, xhi = data[:, 0].min() - 0.05, data[:, 0].max() + 0.05
    ylo, yhi = data[:, 1].min() - 0.05, data[:, 1].max() + 0.05
    # xlo, xhi = -2.52764094 - 0.05, 2.33919707 + 0.05
    # ylo, yhi = -1.73163792 - 0.05, 2.30328581 + 0.05

    if eval_valid:
        assert nobs + nvalid + ntest <= total_num, \
            "nobs = {}, nvalid = {}, ntest = {}, total_num = {}".format(nobs, nvalid, ntest, total_num)
    else:
        assert nobs + ntest <= total_num, "nobs = {}, ntest = {}, total_num = {}".format(nobs, ntest, total_num)

    idx = rs.permutation(total_num)
    idx_train = idx[:nobs]
    if eval_valid:
        idx_valid = idx[nobs:nobs+nvalid]
    idx_test = idx[-ntest:]

    xobs = data[idx_train, :2]
    yobs = data[idx_train, -1:]
    sobs = np.ones_like(yobs) * noise_std**2

    xvalid = None
    yvalid = None
    if eval_valid:
        xvalid = data[idx_valid, :2]
        yvalid = data[idx_valid, -1:]

    xtest = data[idx_test, :2]
    ytest = data[idx_test, -1:]

    xgrid = None
    if eval_grid:
        x1_grid = np.linspace(xlo, xhi, gridnum)
        x2_grid = np.linspace(ylo, yhi, gridnum)
        xx1, xx2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
        xgrid = np.column_stack([xx1.flatten(order='C'),
                                 xx2.flatten(order='C')])

    ddict = {
        'xobs': xobs, 'fobs': None, 'sobs': sobs, 'aobs': None, 'yobs': yobs,
        'xtest': xtest, 'ftest': None, 'ytest': ytest, 'stest': None,
        'xgrid': xgrid, 'fgrid': None,
        'xvalid': xvalid, 'yvalid': yvalid,
        'xlo': xlo, 'xhi': xhi, 'ylo': ylo, 'yhi': yhi
    }

    return ddict

