"""
UK Housing Dataset and plotting functions.  This is a hodgepodge of shape
files from different sources

  - UK land registry monthly price data:
    https://ckan.publishing.service.gov.uk/dataset/land-registry-monthly-price-paid-data

    This file includes POSTCODE, which are pretty granular in the UK.
    Crucially, it does not include lat/long data, which I got from another
    source

  - UK Post Code lat/long data: https://www.freemaptools.com/download-uk-postcode-lat-lng.htm

  - UK shapefiles: https://gadm.org/download_country_v3.html


ACM
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np

class UKHousingData:
    """ Class contains housing data, attributes, shapefile for mapping, etc

    should also handle input warping, and other stuff...

    Note: 
        - x axis = 'longitude'
        - y axis = 'lattitude' ....
        - property_type = "F" flats
    Subtract the mean out of ys to make it easier

    """
    def __init__(self, data_dir="./uk-price-paid-data",
                       roi_xlim=(-5.7, 1.8),
                       roi_ylim=(50, 55.5),
                       property_type="F"):
        self.data_dir = data_dir

        # price data
        self.pricedf = load_uk_pricing_data(data_dir=data_dir, year="2018")

        # subselect to region of interest
        self.roi_xlim = roi_xlim
        self.roi_ylim = roi_ylim
        xs = self.pricedf[['latitude', 'longitude']]
        roi_idx = (xs['longitude'] > roi_xlim[0]) & \
                  (xs['longitude'] < roi_xlim[1]) & \
                  (xs['latitude']  > roi_ylim[0]) & \
                  (xs['latitude']  < roi_ylim[1])

        # property type df
        fidx = self.pricedf['property-type']=="F"
        self.pricedf = self.pricedf[ roi_idx & fidx]

        # load shape files
        self.shapedf = load_uk_shape(data_dir=data_dir)

        # shortcut to input and output data
        self.ys_orig = self.pricedf['log-price'].values
        self.ys = self.ys_orig - np.mean(self.ys_orig)
        self.xs = self.pricedf[['longitude','latitude']].values

    def plot_uk(self, ax):
        self.shapedf.plot(ax=ax)
        return ax


def load_uk_pricing_data(data_dir="./", year="2018"):

    # load pricing dataframe --- has POST CODE
    year = "2018"
    colnames = ['id', 'price', 'date', 'postcode', 'property-type', 'new',
                'duration', 'primary-addressable', 'secondary-addressable',
                'street', 'locality', 'city', 'district', 'county', 
                'ppd-category', 'record-status']
    fname = os.path.join(data_dir, "pp-%s.csv"%year)
    pricedf = pd.read_csv(fname, skiprows=0, header=None, names=colnames)

    # date to datetime object, log price
    pricedf['date'] = pd.to_datetime(pricedf['date'])
    pricedf['log-price'] = np.log(pricedf['price'])

    # load postcode => lat/long dataframe
    pname = os.path.join(data_dir, "ukpostcodes.csv")
    postdf = pd.read_csv(pname, skiprows=0)

    # merge postdf and pricedf
    mdf = pd.merge(pricedf, postdf, on='postcode', how='left')

    # remove outliers and missing data
    miss_idx = pd.isnull(mdf['longitude']) | (mdf['price'] < 1000) | \
               (mdf['latitude'] > 65)
    mdf = mdf[~miss_idx]
    return mdf


def load_uk_shape(data_dir="./"):
    fname = os.path.join(data_dir, "gadm36_GBR_shp/gadm36_GBR_3.shp")
    sdf = gpd.read_file(fname)
    return sdf


def idx_inside(pts, xlim, ylim):
    return (pts[:,0] > xlim[0]) & \
           (pts[:,0] < xlim[1]) & \
           (pts[:,1] > ylim[0]) & \
           (pts[:,1] < ylim[1])


def local_linear_noise_var_approx(hdata):
    """ grabs small regions, fits a local linear approx, estimates noise
    from residual, and averages over many patches."""
    # subselect a handful of small regions
    rs = np.random.RandomState(42)
    xd = hdata.roi_xlim[1] - hdata.roi_xlim[0]
    yd = hdata.roi_ylim[1] - hdata.roi_ylim[0]
    dx = xd / 1000
    dy = yd / 1000

    # random box
    num_found = []
    var_found = []
    for ni in range(500):
        x0 = rs.rand()*xd + hdata.roi_xlim[0]
        y0 = rs.rand()*yd + hdata.roi_ylim[0]
        idx = idx_inside(hdata.xs, xlim=(x0, x0+dx), ylim=(y0, y0+dy))
        nfound = np.sum(idx)
        if nfound < 5:
            continue
        xsi, ysi = hdata.xs[idx], hdata.ys[idx]
        _, residual_sum, _, _= np.linalg.lstsq(a=xsi, b=ysi, rcond=None)
        if len(residual_sum) == 0:
            continue
        var_hat = residual_sum / (len(ysi)-1)
        num_found.append(np.sum(idx))
        var_found.append(var_hat[0])

    df = pd.DataFrame({'n': num_found, 'var': var_found})
    print(" ... empirical noise var estimate: ", df['var'].mean())
    return df['var'].mean()/10.


def make_data_dict(Ntrain=-1, Ntest=20000, gridnum=256):
    rs = np.random.RandomState(0)
    hdata = UKHousingData()
    #pdf, sdf = hdata.pricedf, hdata.shapedf
    #xs, ys = pdf[['latitude', 'longitude']], pdf['log-price']

    # noise variance estimate
    noise_var = local_linear_noise_var_approx(hdata)
    total_var = hdata.ys.var()
    sig2_est  = (total_var - noise_var)
    sall = np.sqrt(noise_var)*np.ones(len(hdata.xs))

    # split train/test
    Ntest = Ntest
    total_num = len(hdata.xs)  #  180947 data points in total
    if Ntrain == -1:
        Ntrain = total_num - Ntest
    idx = rs.permutation(total_num)
    idx_train = idx[:Ntrain]
    idx_test  = idx[-Ntest:]
    xobs = hdata.xs[idx_train,:]
    yobs = hdata.ys[idx_train][:,None]
    sobs = sall[idx_train][:,None]

    xtest = hdata.xs[idx_test,:]
    ytest = hdata.ys[idx_test][:,None]
    stest  = sall[idx_test][:,None]

    # x1 and x2 grids
    x1_grid  = np.linspace(*hdata.roi_xlim, gridnum)
    x2_grid  = np.linspace(*hdata.roi_ylim, gridnum)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
    xgrid    = np.column_stack([xx1.flatten(order='C'),
                                xx2.flatten(order='C')])
    vmin, vmax = hdata.ys.min()+np.sqrt(noise_var), \
                 hdata.ys.max()-np.sqrt(noise_var)
    ddict = {
      'xobs' : xobs,  'fobs' : None,  'sobs' : sobs, 'aobs': None, 'yobs': yobs,
      'xtest': xtest, 'ftest': None,  'ytest': ytest, 'stest': stest,
      'x1_grid': x1_grid, 'x2_grid': x2_grid, 'xx1': xx1, 'xx2': xx2,
      'xgrid': xgrid, 'fgrid': None, 'vmin':vmin, 'vmax': vmax,
      'total_var': total_var,
      'sig2_est' : sig2_est,
      'noise_std': np.sqrt(noise_var),
      'hdata': hdata
    }
    return ddict


if __name__=="__main__":

    from uk_housing_data import *
    hdata = UKHousingData()
    pdf, sdf = hdata.pricedf, hdata.shapedf
    xs, ys = pdf[['latitude', 'longitude']], pdf['log-price']

    idx = np.random.permutation(len(xs))[:10000]
    import matplotlib.pyplot as plt; plt.ion()
    from ziggy import viz
    fig, ax = plt.figure(figsize=(6,8)), plt.gca()
    sdf.plot(ax=ax, facecolor='white', edgecolor='black',alpha=.5)
    cm = ax.scatter(xs['longitude'][idx], xs['latitude'][idx], c=ys[idx].values, s=10)
    viz.colorbar(cm, ax)

    ax.set_ylim(50, 55.5)
    ax.set_xlim(-5.7, 1.8)
    #hdata.roi_ylim)
    #ax.set_xlim(hdata.roi_xlim)
    ax.set_aspect("auto")




    #sdf = sdf.to_crs(epsg=3857)
    #ax = sdf.plot()
    #ax.set_aspect("equal")
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    #pdf = pd.read_csv("ukpostcodes.csv", skiprows=0)
    #pricedf = load_uk_pricing_data()
    ##pricedf.join(pdf, on='postcode', how='left')

    #idx = np.random.permutation(len(pdf))[:100000]
    #x = pdf['longitude'].values[idx]
    #y = pdf['latitude'].values[idx]
    #ax.scatter(x, y, c='red')

    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)



