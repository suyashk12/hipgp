import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import torch

def colorbar(mappable, ax):
    """ sol'n from https://joseph-long.com/writing/colorbars/"""
    #ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def rotate_ticks(ax, rotation=40, fontsize=12):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        tick.label.set_rotation(rotation)
    return ax


def plot_smooth(ax, mu,
                xlim=(0,1), ylim=(0,1),
                vmin=None, vmax=None, ticklabels=True):
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if not isinstance(mu, np.ndarray):
        raise ValueError("mu must be nd array, but got {}".format(type(mu)))
    with sns.plotting_context(font_scale=1.2):
        xlo, xhi = xlim
        ylo, yhi = ylim
        """
        cm = ax.imshow(mu,
                       origin='lower',
                       extent=(xlo, xhi, ylo, yhi),
                       interpolation='bilinear',
                       vmin=vmin, vmax=vmax)
        """
        cm = ax.imshow(mu.T[::-1],
                       extent=(xlo, xhi, ylo, yhi),
                       interpolation='bilinear',
                       vmin=vmin, vmax=vmax)
        colorbar(cm, ax)
        ax.set_xlabel("$x_1$", fontsize=20)
        ax.set_ylabel("$x_2$", fontsize=20)
        if ticklabels==False:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        return cm


def plot_comparison(fx_grid, mu_grid, sig2_grid, xx1, xx2, xobs, smooth=True):

    xlo, xhi = xx1.min(), xx1.max()
    vmin, vmax = fx_grid.min(), fx_grid.max()
    if smooth:
        plotfun = lambda f, ax, vmin=vmin, vmax=vmax: ax.imshow(
            f.reshape(xx1.shape), origin='lower',
            extent=(xlo, xhi, xlo, xhi), interpolation='bilinear',
            vmin=vmin, vmax=vmax)
    else:
        plotfun = lambda f, ax, vmin=vmin, vmax=vmax: ax.contourf(
            xx1, xx2, f.reshape(xx1.shape), vmin=vmin, vmax=vmax)

    fig, axarr = plt.subplots(2,2, figsize=(11,10))
    ax = axarr[0,0]
    cs = plotfun(fx_grid, ax)
    colorbar(cs, ax)
    ax.scatter(xobs[:,0], xobs[:,1], c='black', s=2)
    ax.scatter(0, 0, c='red', s=100)
    ax.set_xlim(xlo,xhi)
    ax.set_ylim(xlo,xhi)
    ax.set_title("True rho(x)")

    ax = axarr[0, 1]
    cs = plotfun(mu_grid, ax)
    colorbar(cs, ax)
    ax.set_title("Inferred from %d observations"%(len((xobs))))

    ax = axarr[1, 0]
    cs = plotfun(np.sqrt(sig2_grid), ax, vmin=None, vmax=None)
    colorbar(cs, ax)
    ax.set_title("(marginal) posterior stdev")

    ax = axarr[1, 1]
    #cs = ax.contourf(xx1, xx2, (fx_grid - mu_grid).reshape(xx1.shape))#, vmin=vmin, vmax=vmax)
    res = fx_grid - mu_grid
    cs = plotfun(res, ax, vmin=res.min(), vmax=res.max())
    colorbar(cs, ax)
    ax.set_title("rho(x) - mu(x) (residual)")
    fig.tight_layout()
    return fig, axarr


def ax_scatter(fig, ax, x1, x2, c, x1label, x2label, title=None, s=20, alpha=1.0, vmin=None, vmax=None):
    im = ax.scatter(x1, x2, s=s, c=c, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_xlabel(x1label, fontsize=20)
    ax.set_ylabel(x2label, fontsize=20)
    if title is not None:
        ax.set_title(title, fontsize=20)
    fig.colorbar(im, ax=ax, orientation='vertical')

