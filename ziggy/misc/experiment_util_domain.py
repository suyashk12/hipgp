# try seg Agg                                                                                                                                                
from ziggy.misc import experiment_util as eu
import matplotlib
#matplotlib.use('agg')  AGG backend is used to writing to the file, not showing in the window
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
sns.set_context("paper")
import pandas as pd
import os
import git

#####################                                                                                                                                        
# start script      #                                                                                                                                        
#####################                                                                                                                                        
import torch
import numpy as np
import numpy.random as npr; npr.seed(42)


def load_integrated_data(data_dir):
    data = pd.read_table(data_dir, sep=' ')
    return {'xobs':data[['x', 'y', 'z']].values,
            'eobs':data['e'].values,
            'fobs': data['density'].values}


def make_domain_data(**kwargs):
    dd = load_integrated_data(kwargs.get('data_dir'))

    Nobs, Ntest = kwargs.get("Nobs"), kwargs.get("Ntest")
    total_nobs = dd['xobs'].shape[0]
    if Nobs == -1:
        Nobs = total_nobs - Ntest
        assert Nobs > 0, "total_nobs = {}, ntest = {}".format(total_nobs, Ntest)
    noise_std   = kwargs.get("noise_std")
    rs = np.random.RandomState(kwargs.get("seed"))

    #train_idx = slice(0, -Ntest, 1)

    # first shuffle data
    idx = np.arange(len(dd['xobs']))
    np.random.shuffle(idx)

    xall = dd['xobs'][idx]
    eall = dd['eobs'][idx]
    fall = dd['fobs'][idx]

    train_idx = slice(0, Nobs, 1)
    test_idx = slice(-Ntest, None, 1)

    xlo, ylo, zlo = kwargs['xlo'], kwargs['xlo'], kwargs['zlo']
    xhi, yhi, zhi =  kwargs['xhi'], kwargs['xhi'], kwargs['zhi']

    index = (xall[:,0] <= xhi) & (xall[:,0] >= xlo) & \
            (xall[:,1] <= yhi) & (xall[:,1] >= ylo) & \
            (xall[:,2] <= zhi) & (xall[:,2] >= zlo)

    xall = xall[index]
    eall = eall[index]
    # generate data with variable noise, [1/2 sig, 3/2 sig]                                                                                                                                    
    xobs = xall[train_idx,:]
    eobs = eall[train_idx]
    half_std = noise_std / 2
    sobs = rs.rand(len(eobs))*noise_std + half_std
    aobs = eobs + rs.randn(len(eobs))*sobs
    fobs = fall[train_idx]  # without noise

    # test data                                                                                                                                                                                
    xtest = xall[test_idx,:]
    etest = eall[test_idx]
    ftest = fall[test_idx]

    ddict = {
        'xobs' : xobs,  'fobs' : fobs,  'eobs' : eobs,
        'sobs' : sobs,  'aobs' : aobs,  'yobs' : None,
        'xtest': xtest, 'ftest': ftest, 'etest': etest,
        'xgrid': None,
        'vmin':0, 'vmax': None,
        'xlo': xlo, 'xhi': xhi,
        'ylo': ylo, 'yhi': yhi,
        'zlo': zlo, 'zhi': zhi
    }

    foodict = {'xobs':xobs, 'eobs':eobs, 'sobs':sobs, 'aobs':aobs}
    print('saving trainingdata.pkl')
    torch.save(foodict, "trainingdata.pkl")
    print('saved')

    return ddict


def make_error_plots(output_dir, model_names, pretty_names=None):
    """ visualizes different test sample errors                                                                                                                   
        - mean squared error (rho, e)                                                                                                                             
        - log like (rho, e)                                                                                                                                       
        - chisq (rho, e)                                                                                                                                          
    """
    df = eu.make_error_dataframe(model_names, pretty_names)

    # plots                                                                                                                                                       
    sns.set_context("paper")
    error_types = ["e mse", #"f mse",
                   "e mae", #"f mae",
                   "e loglike", #"f loglike",
                   "e chisq"] #, "f chisq"]

    # save plot of each error type                                                                                                                                
    for etype in error_types:
        #with sns.plotting_context(font_scale=10):                                                                                                                
        sns.set(font_scale = 1.2, style='whitegrid')

        fig, ax = plt.figure(figsize=(4,3)), plt.gca()
        ax = sns.pointplot(x="model", y=etype, data=df, join=False, ax=ax)

        if np.max(df[etype]) < 0.01:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-7,-7))

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
