import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import os
from ziggy import viz

import os
import sys
script_dir = "../build/lib/ziggy/misc"
sys.path.append(os.path.abspath(script_dir))
import experiment_util as eu
from experiment_util import standard_epoch_callback, plot_posterior_grid

import seaborn as sns; sns.set_style("white")
sns.set_context("paper")

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

    sns.set(font_scale = 1, style='whitegrid')

    # load dictionary of predictions
    xtest = pdict['xtest']

    # determining a "good" slice (in the z, or x3 direction) for predictions, centered at mean_z and with thickness 2*diff_z
    mean_z = 0
    diff_z = 0.05

    # isolating indices within that fall within the slice
    inds = []

    for i in range(0, len(pdict['xtest'])):
        if(pdict['xtest'][i][2] >= mean_z - diff_z and pdict['xtest'][i][2] <= mean_z + diff_z):
            inds.append(i)

    # determining quantities from that slice
    xtest_slice = pdict['xtest'][inds]

    # if extinctions are predicted, plot their distribution and their statistics
    try:

        # get true extinctions from the testing set, the posterior mean and variance
        etest = pdict['etest']
        emu_test = pdict['emu_test']
        esig_test = pdict['esig_test']

        # calculate residual, relative error, and z-score for predictions viz-a-viz the true values
        eres_test = emu_test - etest
        erel_test = eres_test/etest
        ez_test = -eres_test/esig_test

        # 3-D plots

        # posterior mean
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca(projection='3d')
        im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=emu_test, s=20)
        cbar = fig.colorbar(im, location = "left")
        cbar.set_label(r'Posterior mean of $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_zlabel(r'$z$ (kpc)')
        ax.set_box_aspect([1,1,1])
        plt.savefig(os.path.join(odir, "predict-emu-test-3D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # posterior variance
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca(projection='3d')
        im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=esig_test, s=20)
        cbar = fig.colorbar(im, location = "left")
        cbar.set_label(r'Posterior error in $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_zlabel(r'$z$ (kpc)')
        ax.set_box_aspect([1,1,1])
        plt.savefig(os.path.join(odir, "predict-esig-test-3D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # residual
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca(projection='3d')
        im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=eres_test, s=20)
        cbar = fig.colorbar(im, location = "left")
        cbar.set_label(r'Residual of $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_zlabel(r'$z$ (kpc)')
        ax.set_box_aspect([1,1,1])
        plt.savefig(os.path.join(odir, "predict-eres-test-3D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # relative error
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca(projection='3d')
        im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=erel_test, s=20)
        cbar = fig.colorbar(im, location = "left")
        cbar.set_label(r'Relative error in $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_zlabel(r'$z$ (kpc)')
        ax.set_box_aspect([1,1,1])
        plt.savefig(os.path.join(odir, "predict-erel-test-3D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # z-score
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca(projection='3d')
        im = ax.scatter(xtest[:, 0], xtest[:, 1], xtest[:, 2], c=ez_test, s=20)
        cbar = fig.colorbar(im, location = "left")
        cbar.set_label(r'Z-score of $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_zlabel(r'$z$ (kpc)')
        ax.set_box_aspect([1,1,1])
        plt.savefig(os.path.join(odir, "predict-ez-test-3D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # get true extinctions, posterior mean and variance, and statistics within the slice

        etest_slice = pdict['etest'][inds]
        emu_test_slice = pdict['emu_test'][inds]
        esig_test_slice = pdict["esig_test"][inds]
        eres_test_slice = emu_test_slice - etest_slice
        erel_test_slice = eres_test_slice/etest_slice
        ez_test_slice = -eres_test_slice/esig_test_slice

        # 2-D plots

        # posterior mean
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca()
        im = ax.scatter(xtest_slice[:, 0], xtest_slice[:, 1], c = emu_test_slice)
        cbar = fig.colorbar(im)
        cbar.set_label(r'Posterior mean of $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(odir, "predict-emu-test-2D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # posterior variance
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca()
        im = ax.scatter(xtest_slice[:, 0], xtest_slice[:, 1], c = esig_test_slice)
        cbar = fig.colorbar(im)
        cbar.set_label(r'Posterior error in $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(odir, "predict-esig-test-2D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # residual
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca()
        im = ax.scatter(xtest_slice[:, 0], xtest_slice[:, 1], c = eres_test_slice)
        cbar = fig.colorbar(im)
        cbar.set_label(r'Residual of $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(odir, "predict-eres-test-2D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # relative error
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca()
        im = ax.scatter(xtest_slice[:, 0], xtest_slice[:, 1], c = erel_test_slice)
        cbar = fig.colorbar(im)
        cbar.set_label(r'Relative error in $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(odir, "predict-erel-test-2D.pdf"), dpi = 300, transparent = True)
        plt.close()

        # z-score
        fig = plt.figure(figsize = (6, 6))
        ax = plt.gca()
        im = ax.scatter(xtest_slice[:, 0], xtest_slice[:, 1], c = ez_test_slice)
        cbar = fig.colorbar(im)
        cbar.set_label(r'Z-score of $e$')
        ax.set_xlabel(r'$x$ (kpc)')
        ax.set_ylabel(r'$y$ (kpc)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(odir, "predict-ez-test-2D.pdf"), dpi = 300, transparent = True)
        plt.close()

    except:
        pass

def plot_domain_true(data_dict, output_dir):

    sns.set(font_scale = 1, style='whitegrid')

    # getting spatial extents for plots

    xlo = data_dict['xlo']
    xhi = data_dict['xhi']
    zlo = data_dict['zlo']
    zhi = data_dict['zhi']

    # 3-D plots

    # plot training set
    fig = plt.figure(figsize = (8, 6))
    ax = plt.gca(projection='3d')
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(xlo, xhi)
    ax.set_zlim(zlo, zhi)
    im = ax.scatter(data_dict['xobs'][:, 0], data_dict['xobs'][:, 1], data_dict['xobs'][:, 2], c=data_dict['eobs'], s=20)
    cbar = fig.colorbar(im, location='left')
    cbar.set_label(r'Train $e$')
    ax.set_xlabel(r'$x$ (kpc)')
    ax.set_ylabel(r'$y$ (kpc)')
    ax.set_zlabel(r'$z$ (kpc)')
    ax.set_box_aspect([2,2,1])
    plt.savefig(os.path.join(output_dir, "true-eobs-3D.pdf"), dpi = 300, transparent = True)
    plt.close()
    
    # plot testing set
    fig = plt.figure(figsize = (8, 6))
    ax = plt.gca(projection='3d')
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(xlo, xhi)
    ax.set_zlim(zlo, zhi)
    im = ax.scatter(data_dict['xtest'][:, 0], data_dict['xtest'][:, 1], data_dict['xtest'][:, 2], c=data_dict['etest'], s=20)
    cbar = fig.colorbar(im, location='left')
    cbar.set_label(r'Test $e$')
    ax.set_xlabel(r'$x$ (kpc)')
    ax.set_ylabel(r'$y$ (kpc)')
    ax.set_zlabel(r'$z$ (kpc)')
    ax.set_box_aspect([2,2,1])
    plt.savefig(os.path.join(output_dir, "true-etest-3D.pdf"), dpi = 300, transparent = True)
    plt.close()

    # 2D plots

    # training set

    # determining a "good" slice (in the z, or x3 direction) for training set, centered at mean_obs_z and with thickness 2*diff_obs_z
    mean_obs_z = 0
    diff_obs_z = 0.05

    # isolating indices that fall within the slice
    inds = []

    for i in range(0, len(data_dict['xtest'])):
        if(data_dict['xobs'][i][2] >= mean_obs_z - diff_obs_z and data_dict['xobs'][i][2] <= mean_obs_z + diff_obs_z):
            inds.append(i)

    # determining quantities from that slice
    xtest_slice = data_dict['xobs'][inds]
    etest_slice = data_dict['eobs'][inds]

    # plot training set
    fig = plt.figure(figsize = (6, 6))
    ax = plt.gca()
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(xlo, xhi)
    im = ax.scatter(xtest_slice[:, 0], xtest_slice[:, 1], c = etest_slice)
    cbar = fig.colorbar(im)
    cbar.set_label(r'Train $e$')
    ax.set_xlabel(r'$x$ (kpc)')
    ax.set_ylabel(r'$y$ (kpc)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "true-eobs-2D.pdf"), dpi = 300, transparent = True)
    plt.close()

    # determining a "good" slice (in the z, or x3 direction) for testing set, centered at mean_test_z and with thickness 2*diff_test_z
    mean_test_z = 0
    diff_test_z = 0.05

    # isolating indices that fall within the slice
    inds = []

    for i in range(0, len(data_dict['xtest'])):
        if(data_dict['xtest'][i][2] >= mean_test_z - diff_test_z and data_dict['xtest'][i][2] <= mean_test_z + diff_test_z):
            inds.append(i)

    # determining quantities from that slice
    xtest_slice = data_dict['xtest'][inds]
    etest_slice = data_dict['etest'][inds]

    # plot testing set
    fig = plt.figure(figsize = (6, 6))
    ax = plt.gca()
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(xlo, xhi)
    im = ax.scatter(xtest_slice[:, 0], xtest_slice[:, 1], c = etest_slice)
    cbar = fig.colorbar(im)
    cbar.set_label(r'Test $e$')
    ax.set_xlabel(r'$x$ (kpc)')
    ax.set_ylabel(r'$y$ (kpc)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "true-etest-2D.pdf"), dpi = 300, transparent = True)
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
