from ziggy import svgp
import torch
from torch import nn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
from ziggy import viz
from ziggy import misc as zmisc


def make_two_dim_data(**kwargs):
    """ make two dimensional function with lots of direction changes """
    # unpack args
    rs = np.random.RandomState(42)
    Nobs, Ntest     = kwargs.get("Nobs"), kwargs.get("Ntest")
    noise_std       = kwargs.get("noise_std")
    func_complexity = kwargs.get("function_complexity", "medium")
    do_integrated   = kwargs.get("integrated_obs", False)

    # complexity is a function of weight standard deviation
    weight_std, hidden_dim = {
        'simple': (10, 10),
        'medium': (35, 10),
        'hard'  : (40, 25),
      }[func_complexity]

    # Create Test function and Grid Observations
    torch.manual_seed(42)
    ftrue_pt, ftrue = make_two_dim_synthetic_function(weight_std, hidden_dim)

    # grid observations --- make this MEAN ZERO
    xlo, xhi = kwargs.get("xlo", -1), kwargs.get("xhi", 1)
    gridnum = kwargs.get("gridnum", 256)
    x1_grid = np.linspace(xlo, xhi, gridnum)
    x2_grid = np.linspace(xlo, xhi, gridnum)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
    xgrid    = np.column_stack([xx1.flatten(order='C'),
                                xx2.flatten(order='C')])
    fgrid_orig = ftrue(xgrid)
    fgrid_orig_mean = np.mean(fgrid_orig)
    fgrid = fgrid_orig - fgrid_orig_mean
    fgrid = fgrid.reshape(gridnum, gridnum)

    # observation data
    xobs = rs.rand(Nobs, 2) * (xhi-xlo) + xlo
    sobs = noise_std*np.ones(xobs.shape[0])
    fobs = ftrue(xobs) - fgrid_orig_mean
    yobs = fobs + sobs[:,None]*rs.randn(fobs.shape[0],1)
    if do_integrated:
        eobs = zmisc.integrated_obs(xobs, ftrue)
        aobs = eobs + sobs*rs.randn(eobs.shape[0])
    else:
        eobs, aobs = None, None

    # test data
    xtest = (rs.rand(Ntest, 2)) * (xhi-xlo) + xlo
    ftest = ftrue(xtest) - fgrid_orig_mean
    if do_integrated:
        etest = zmisc.integrated_obs(xtest, ftrue)
    else:
        etest = None

    # snr statistics
    f_snr = np.std(fobs) / noise_std
    e_snr = None #np.std(eobs) / noise_std
    ddict = {
      'xobs' : xobs,  'fobs' : fobs,  'sobs' : sobs, 'aobs': aobs, 'yobs': yobs,
      'xtest': xtest, 'ftest': ftest, 'etest': etest,
      'f_snr': f_snr, 'e_snr': e_snr,
      'x1_grid': x1_grid, 'x2_grid': x2_grid, 'xx1': xx1, 'xx2': xx2,
      'xgrid': xgrid, 'fgrid': fgrid, 'vmin':0, 'vmax': fgrid.max()  # TODO: check vmin
    }
    # xobs shape: (N, 2) fobs shape: (N, 1) sobs shape: (N, 1) yobs shape: (N, 1)
    return {**kwargs, **ddict}


def make_two_dim_synthetic_function(weight_std=35, hidden_dim=10, seed=42):
    torch.manual_seed(seed)
    class TestFun(nn.Module):
        def __init__(self):
            super(TestFun, self).__init__()
            self.lin = nn.Linear(2, hidden_dim)
            self.out = nn.Linear(hidden_dim, 1)

            for p in self.lin.parameters():
                p.data.normal_(std=weight_std).double()
            for p in self.out.parameters():
                p.data.normal_(std=.2).double()

            self.sft_plus = nn.Softplus()

        def forward(self, x):
            h = torch.sin(self.lin(x))
            h = torch.tanh(h)
            out = self.out(h)
            return self.sft_plus(out)

    fun = TestFun()
    fun.double()

    def fun_npy(x):
        return fun(torch.DoubleTensor(x)).detach().numpy()

    return fun, fun_npy


def plot_synthetic_data(data_dict):
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns; sns.set_style("white")
    from ziggy import viz
    sns.set(font_scale = 1.2, style='white')
    xx1, xx2 = data_dict['xx1'], data_dict['xx2']
    vmin, vmax = data_dict['vmin'], data_dict['vmax']
    (xlo, xhi) = xx1.min(), xx1.max()
    (ylo, yhi) = xx2.min(), xx2.max()
    fig, ax = plt.figure(figsize=(6,6)), plt.gca()
    cm = viz.plot_smooth(ax, data_dict['fgrid'].reshape(xx1.shape),
                         xlim=(xlo, xhi), ylim=(xlo, xhi),
                         vmin=vmin, vmax=vmax,
                         ticklabels=False)
    viz.colorbar(cm, ax)
    return fig, ax
