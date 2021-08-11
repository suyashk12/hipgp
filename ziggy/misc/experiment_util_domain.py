# try seg Agg                                                                                                                                                
import experiment_util as eu                                                                                                                                
#from ziggy.misc import experiment_util as eu
from ziggy import svgp, kernels, viz
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
sns.set_context("paper")
import pandas as pd
import os

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
            'eobserr':data['e_err'].values,
            #'fobs': data['density'].values
            }

def make_domain_data(vmin=None, vmax=None, **kwargs):
    #dd = pickle.load(open("andysToySet0.6kpc0.6kpc0.5kpc.m12mlsr2.pkl", 'rb'))                                                                                                                
    #dd = load_ananke_integrated_data()                                                                                                                                                        
    dd = load_integrated_data(kwargs.get('data_dir'))

    Nobs, Ntest = kwargs.get("Nobs"), kwargs.get("Ntest")
    total_nobs = dd['xobs'].shape[0]

    if Nobs == -1:
        Nobs = total_nobs - Ntest
        assert Nobs > 0, "total_nobs = {}, ntest = {}".format(total_nobs, Ntest)

    xlo, xhi    = kwargs.get("xlo"), kwargs.get("xhi")
    noise_std   = kwargs.get("noise_std")
    rs = np.random.RandomState(kwargs.get("seed"))

    # first shuffle data
    idx = np.arange(len(dd['xobs']))
    np.random.shuffle(idx)

    xall = dd['xobs'][idx]
    eall = dd['eobs'][idx]
    # eerrall will be necessary for gaia data
    eerrall = dd['eobserr'][idx]

    try:
        fall = dd['fobs'][idx]
    except:
        fall = None

    # specify train and test slices
    train_idx = slice(0, Nobs, 1)
    test_idx = slice(-Ntest, None, 1)

    xlo, ylo, zlo = kwargs['xlo'], kwargs['xlo'], kwargs['zlo']
    xhi, yhi, zhi =  kwargs['xhi'], kwargs['xhi'], kwargs['zhi']

    # isolate points in the desired spatial extent
    index = (xall[:,0] <= xhi) & (xall[:,0] >= xlo) & \
            (xall[:,1] <= yhi) & (xall[:,1] >= ylo) & \
            (xall[:,2] <= zhi) & (xall[:,2] >= zlo)

    xall = xall[index]
    eall = eall[index]
    eerrall = eerrall[index]

    # nz=1 won't work directly unless you already have fgrid_30_30_1.npy already. check the readme to find out what to do if that is the case
    if kwargs['small_box']: nz = 1
    else: nz = 3

    # generate data with variable noise, [1/2 sig, 3/2 sig] (this is synthetic noise, only valid for simulation data)                                                                                                                                
    xobs = xall[train_idx,:]
    eobs = eall[train_idx]
    eerrobs = eerrall[train_idx]

    half_std = noise_std / 2

    sobs = rs.rand(len(eobs))*noise_std + half_std
    aobs = eobs + rs.randn(len(eobs))*sobs

    # for gaia data, we have actual uncertainties we can use
    if(kwargs['dataset'] == 'gaia'):
        sobs = eerrobs + 0.1 # 0.1 is from systematic uncertanties
        aobs = eobs # the gaia extinctions are already noise, no need to add noise to it, unlike simulation

    try:
        fobs = fall[train_idx]  # without noise
    except:
        fobs = None

    # test data                                                                                                                                                                                
    xtest = xall[test_idx,:]
    etest = eall[test_idx]

    try:
        ftest = fall[test_idx]
    except:
        ftest = None

    nx = ny = 30

    x1_grid = np.linspace(xlo, xhi, nx)
    x2_grid = np.linspace(ylo, yhi, ny)
    x3_grid = np.linspace(zlo, zhi, nz)

    if(nz==1):
        x3_grid = np.array([0])

    xx1, xx2, xx3 = np.meshgrid(x1_grid, x2_grid, x3_grid)

    xgrid = np.column_stack([xx1.flatten(), xx2.flatten(), xx3.flatten()])

    if(kwargs['dataset'] != 'gaia'):

        # position of predictions and their true process values                                                       
        fgridfile = 'fgrid_{0}_{1}_{2}.npy'.format(nx, ny, nz)
        #if kwargs.get('small_box'): fgridfile = 'fgrid_{0}_{1}_{2}_smallbox.npy'.format(nx, ny, nz)

        try: fgrid = np.load(fgridfile)
        except IOError:
            fgrid = genDustDensity(xgrid, nx, ny, nz)
            #fgrid, A_to_dust_density = genDustDensity(xgrid, nx, ny, nz)
            np.save(fgridfile, fgrid)
        #import pdb;pdb.set_trace(

        # conversion from conventional units to mag/kpc
        fgrid = np.swapaxes(fgrid, 0, 1)/0.20022

    else:
        fgrid = None

    ddict = {
        'xobs' : xobs,  'fobs' : fobs,  'eobs' : eobs,
        'sobs' : sobs,  'aobs' : aobs,  'yobs' : aobs,
        'xtest': xtest, 'ftest': ftest, 'etest': etest,
        #'f_snr': f_snr, 'e_snr': e_snr,                                                                                                                                                       
        'x1_grid': x1_grid, 'x2_grid': x2_grid, 'x3_grid': x3_grid,
        'xx1': xx1, 'xx2': xx2, 'xx3': xx3,
        'xgrid': xgrid, 'fgrid': fgrid, 'vmin':0, 'vmax': None,
        'xlo': xlo, 'xhi': xhi,
        'ylo': ylo, 'yhi': yhi,
        'zlo': zlo, 'zhi': zhi
    }

    foodict = {'xobs':xobs, 'eobs':eobs, 'sobs':sobs, 'aobs':aobs}

    print('saving trainingdata.pkl')
    torch.save(foodict, "trainingdata.pkl")
    print('saved')

    # create inducing points as well                                                                                                                                                           
    xinduce_grids = [torch.linspace(xlo, xhi, kwargs['num_inducing_x']),
                     torch.linspace(ylo, yhi, kwargs['num_inducing_x']),
                     torch.linspace(zlo, zhi, kwargs['num_inducing_z'])]

    return ddict, xinduce_grids

def make_data_plot(name, args, xobs, aobs, data_dict):
    # TODO create a integrated obs plot for domain data ... 
    Nobs, Dobs = xobs.shape
    if Dobs == 2:
        fig, ax = plt.figure(figsize=(6,6)), plt.gca()
        cs = ax.scatter(xobs[:,0], xobs[:,1], c=aobs, s=5, rasterized=True)
        viz.colorbar(cs, ax)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(xlo, xhi)
        #ax.set_title("$N = ${:,} observations".format(len(xobs)), fontsize=14)
        ax.set_xlabel("$x_1$", fontsize=16)
        ax.set_ylabel("$x_2$", fontsize=16)

        fig.savefig(os.path.join(name, "integrated-obs.pdf"), bbox_inches='tight', rasterized=True)
        plt.close("all")
    else:
        edges = np.linspace(np.min(xobs[:,2]), np.max(xobs[:,2]), 10)
        for elow, ehigh in zip(edges[:-1], edges[1:]):
            slice = (xobs[:,2] <= ehigh) & (xobs[:,2] > elow)
            fig, ax = plt.figure(figsize=(6,6)), plt.gca()
            cs = ax.scatter(xobs[:,0][slice], xobs[:,1][slice], c=aobs[slice], s=5, rasterized=True)
            viz.colorbar(cs, ax)
            xlo, xhi = np.min(xobs[:,0]), np.max(xobs[:,0])
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(xlo, xhi)
            #ax.set_title("$N = ${:,} observations".format(len(xobs)), fontsize=14)
            ax.set_xlabel("$x_1$", fontsize=16)
            ax.set_ylabel("$x_2$", fontsize=16)
            fig.savefig(os.path.join(name, "integrated-obs_zlow{0:.2f}_zhigh{1:.2f}.pdf".format(elow, ehigh)), bbox_inches='tight', rasterized=True)
            plt.close("all")

        #print("Data Viz not implemented for %d-dimensional input"%Dobs)                                                                                                                       

def make_model_plots(name, args, data_dict):
    """ make model-specific plots """
    model_name = name

    df = eu.make_error_dataframe([model_name])
    eu.make_zscore_histogram(model_name, target='e')
    #eu.make_zscore_histogram(model_name, target='f')

    xx1, xx2, xx3 = data_dict['xx1'], data_dict['xx2'], data_dict['xx3']
    plot_posterior_grid(model_name, xx1=xx1, xx2=xx2, xx3=xx3)

    # coverage data frame                                                                                                                                                                      
    covdf = eu.make_coverage_table([model_name], ['SqExp'], target='e')
    covdf.to_latex(os.path.join(model_name, "coverage-table.tex"),
                   escape=False, float_format="%2.3f")
    print(covdf.T)


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

def make_model_distance_plots(model_name, pretty_name=None):
    """ make plots that show error as a function of distance """
    df = eu.make_error_dataframe([model_name], None)

    # plot posterior mean on xgrid                                                                                                                                
    error_types = ["e mse", #"f mse",
                   "e mae", #"f mae",
                   "e loglike", #"f loglike",
                   "e chisq", #"f chisq",
                   "esig_test", #"fsig_test",
                   "etest"] #, "ftest"]
    for error_type in error_types:
        fig, ax = plt.figure(figsize=(6,4)), plt.gca()
        ax.scatter(df['xtest_dist'], df[error_type], s=10)
        ax.set_xlabel("distance", fontsize=12)
        ax.set_ylabel(error_type, fontsize=12)
        if error_type == 'esig_test': ax.set_ylim(0, 0.001)
        if error_type == 'etest': ax.set_ylim(0,)
        fig.savefig(os.path.join(model_name, "dist-by-%s.pdf"%error_type), bbox_inches='tight')
        plt.close("all")

    # for zscore, show 1/2 sd, 1 sd, and 2 sd                                                                                                                     
    error_types = ["e zscore"] #, "f zscore"]
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

def genDustDensity(xgrid, nx, ny, nz, unit_kwargs=None):
    import yt
    from yt.units import dimensions, kpc, kiloparsec, Msun

    def _metal_weighted_density(field, data):
        solarMetallicity = 0.2
        massfraction_he = data['massfraction_he']
        massfraction_all  = data['massfraction_all']
        fractionHydrogen = 1. - massfraction_all - massfraction_he
        neutralHydrogenMass = data['density']*fractionHydrogen*data['hydrogen_neutral_fraction']
        metalWeightedNeutralHydrogenMass = neutralHydrogenMass*10.**data['metallicity']
        return metalWeightedNeutralHydrogenMass

    yt.add_field(("io","dustDensity"), function=_metal_weighted_density, units='Msun/pc**3', #units="auto",                                                                                   
                 dimensions=dimensions.density, sampling_type='particle', force_override=True)

    xscale = np.max(xgrid[:,0])
    yscale = np.max(xgrid[:,1])
    zscale = np.max(xgrid[:,2])
    latte = np.load('latte10kpc_m12f_lsr2_corrected.npz')

    unit_kwargs = {
        'Q'            : 2.5e22, #H/cm2/mag                                                                                                                                                   
        'EBV'          : 1/3.1,
        'm_p'          : 1.6726219e-24, #g                                                                                                                                                    
        'length_unit'  : 3.08567758149137e21, #kpc to cm                                                                                                                                      
        'mass_unit'    : 1.98847e33, #solar mass to g                                                                                                                                         
        'velocity_unit': 1e5, #km to cm                                                                                                                                                       
        'unit_system'  : "galactic", #'cgs'                                                                                                                                                   
        'time_unit'    : 3.15576e16, #Gyr to s,                                                                                                                                               
        'sim_time'     : latte['snapshottime']*3.15576e16,
        'field'        : 'io',
        'boxlength'    : latte['boxlength'],
        'normalizing_factor': 1.0}

    data = {(unit_kwargs['field'], 'particle_position_x'): np.array(latte['x'], dtype='float64'),
            (unit_kwargs['field'], 'particle_position_y'): np.array(latte['y'], dtype='float64'),
            (unit_kwargs['field'], 'particle_position_z'): np.array(latte['z'], dtype='float64'),
            (unit_kwargs['field'], 'particle_velocity_x'): np.array(latte['velocity'][:,0], dtype='float64'),
            (unit_kwargs['field'], 'particle_velocity_y'): np.array(latte['velocity'][:,1], dtype='float64'),
            (unit_kwargs['field'], 'particle_velocity_z'): np.array(latte['velocity'][:,2], dtype='float64'),
            (unit_kwargs['field'], 'density'):             np.array(latte['density'], dtype='float64'),
            (unit_kwargs['field'], 'hydrogen_neutral_fraction'): np.array(latte['hydrogenneutralfraction'], dtype='float64'),
            (unit_kwargs['field'], 'massfraction_he'):     np.array(latte['massfraction'][:,1], dtype='float64'),
            (unit_kwargs['field'], 'massfraction_all'):    np.array(latte['massfraction'][:,0], dtype='float64'),
            (unit_kwargs['field'], 'metallicity'):         np.array(latte['metallicitytotal'], dtype='float64'),
            (unit_kwargs['field'], 'smoothing_length'):    np.array(latte['smoothlength'], dtype='float64'),
            (unit_kwargs['field'], 'particle_mass'):       np.array(latte['mass'], dtype='float64')
        }

    half_box = unit_kwargs['boxlength']/2.*unit_kwargs['length_unit']
    bbox = np.array([[-half_box, half_box], [-half_box, half_box], [-half_box, half_box]])

    ds = yt.load_particles(data, length_unit=unit_kwargs['length_unit'],
                        mass_unit  =unit_kwargs['mass_unit'],   velocity_unit=unit_kwargs['velocity_unit'],
                        time_unit  =unit_kwargs['time_unit'],   bbox         =bbox,
                        unit_system=unit_kwargs['unit_system'], sim_time     =unit_kwargs['sim_time'])
    
    left_edge  = [-xscale, -yscale, -zscale]*kpc
    right_edge = [ xscale,  yscale,  zscale]*kpc

    ag = ds.arbitrary_grid(left_edge, right_edge, dims=[nx, ny, nz])

    print(ag[('io', 'dustDensity')])

    dustDensity = ag[('io', 'dustDensity')]
    return dustDensity #, A_to_dust_density
