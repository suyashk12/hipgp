"""
Old, numpy implementation of integrated obs idea.  File should be deleted.
"""
import numpy as np
import numpy.random as npr
from ziggy import lowrank_mvn as lmvn


#########################################################################
# Main use function --- tune kernel_fun parameters and predict on new data  #
#########################################################################
def tune_kernel_params(xobs, aobs, xinduce, sig2_noise, kernel_obj,
                       objfun="fitc", xstar=None, xgrid=None,
                       sig2_init=None, ell2_init=None):
    if objfun=="fitc":
        def obj(ln_theta):
            return integrated_fitc_objective(np.exp(ln_theta), kernel_obj,
                                                 xobs, aobs, xinduce, sig2_noise, do_vfe=False)        
    elif objfun=="vfe":
        def obj(ln_theta):
            val = integrated_fitc_objective(np.exp(ln_theta), kernel_obj,
                                                xobs, aobs, xinduce, sig2_noise, do_vfe=True)
            return val
    elif objfun=="validation":
        Nval = int(.2*len(xobs))
        def obj(ln_theta):
            val = -1*validation_fitc_objective(np.exp(ln_theta), kernel_obj,
                                            xobs=xobs[:Nval], aobs=aobs[:Nval], 
                                            xtest=xobs[Nval:], atest=aobs[Nval:],
                                            xinduce=xinduce, sig2_noise=sig2_noise).mean()
            return val
    elif objfun=="untuned":
        obj = None
    else:
        raise NotImplementedError("%s unavailable, fitc|vfe|validation")

    # constrain sig2 and ell2 to be reasonable values --- wrap
    def objc(ln_theta):
        sig2, ell2 = np.exp(ln_theta)
        if ell2 > 10 or sig2 > 1e5:
            print("theta: ", np.exp(ln_theta), "returning inf")
            return obj(ln_theta)
        return obj(ln_theta)

    if obj is not None:
        from scipy.optimize import minimize, fmin_bfgs
        theta0 = np.log([10., .5])
        ll0    = objc(theta0)
        print("\nstarting obj: ", ll0)
        res = minimize(objc, theta0, method='Nelder-Mead', options={'maxiter':110, 'disp':True})
        sig2_hat, ell2_hat = np.exp(res.x)
    else:
        sig2_hat, ell2_hat = sig2_init, ell2_init

    # params, store
    params_hat = (sig2_hat, ell2_hat)
    resdict = {'sig2_hat' : sig2_hat,
               'ell2_hat' : ell2_hat}

    # predictions for test stars
    if xstar is not None:
        resdict['emu_star'], resdict['esig2_star'] = \
            gp_sparse_integrated_predict(xobs, aobs, sig2_noise, kernel_obj,
                kernel_params=params_hat,
                xstar=xstar, xinduce=xinduce, xstar_is_integrated=True)
        resdict['fmu_star'], resdict['fsig2_star'] = \
            gp_sparse_integrated_predict(xobs, aobs, sig2_noise, kernel_obj,
                kernel_params=params_hat,
                xstar=xstar, xinduce=xinduce, xstar_is_integrated=False)

    # also save on GRID
    if xgrid is not None:
        resdict['emu_grid'], resdict['esig2_grid'] = \
            gp_sparse_integrated_predict(xobs, aobs, sig2_noise, kernel_obj,
                kernel_params=params_hat,
                xstar=xstar, xinduce=xinduce, xstar_is_integrated=True)
        resdict['fmu_grid'], resdict['fsig2_grid'] = \
            gp_sparse_integrated_predict(xobs, aobs, sig2_noise, kernel_obj,
                kernel_params=params_hat,
                xstar=xgrid, xinduce=xinduce, xstar_is_integrated=False)

    return resdict


#################################################
# Algorithms for inferring predictive values    #
#################################################

def gp_predict(xobs, yobs, sig2, Sinv, sig2_noise, xstar):
    """ Predict function value with noisy realizations of a GP
    with a squared exponential kernel_fun """
    Kxx = K_squared_exp(xobs, xobs, sig2, Sinv)
    Kss = K_squared_exp(xstar, xstar, sig2, Sinv)
    Ksx = K_squared_exp(xstar, xobs, sig2, Sinv)
    mu_grid, S_grid = condition_on(yobs, Kxx, Ksx, Kss, sig2_noise)
    return mu_grid, S_grid


def gp_sparse_predict(xobs, yobs, sig2_noise,
                      kernel_obj, kernel_params,
                      xstar, xinduce, do_slow=False):
    """ Inducing point predictions:  
    Args:
        - xobs
        - yobs
        - sig2_noise
        - kernel_obj
        - kernel_params
        - xstar
        - xinduce

    Returns:
        - mu_star
        - sig2_star
    """
    M, N, S = xinduce.shape[0], xobs.shape[0], xstar.shape[0]
    Kmm = kernel_obj.kfun(xinduce, xinduce, params=kernel_params)
    Knm = kernel_obj.kfun(xobs, xinduce, params=kernel_params)
    Ksm = kernel_obj.kfun(xstar, xinduce, params=kernel_params)
    Knn_diag = kernel_obj.kfun_diag(xobs, params=kernel_params).squeeze()
    Kss_diag = kernel_obj.kfun_diag(xstar, params=kernel_params).squeeze()

    def Q(Knm, Kms):
        """ returns N x S matrix """
        return np.dot(Knm, np.linalg.solve(Kmm, Kms))

    def Ktilde(Knm, Kn_diag):
        Qn = Q(Knm, Knm.T)
        return Qn + np.diag(Kn_diag - np.diag(Qn))

    # slow for checking
    if do_slow:
        Qsn = Q(Ksm, Knm.T)
        Kn_tilde = Ktilde(Knm, Knn_diag)
        mu_s  = np.dot(Qsn,
            np.linalg.solve(Kn_tilde+np.eye(N)*sig2_noise, yobs))
        Sig_s = Ktilde(Ksm, Kss_diag) - \
            np.dot(Qsn, np.linalg.solve(Kn_tilde+np.eye(N)*sig2_noise, Qsn.T))
        return mu_s, Sig_s

    # more efficient implementation -- should never instantiate an NxN matrix
    # Compute the low rank factor
    Qsn = Q(Ksm, Knm.T)
    cKmm = np.linalg.cholesky(Kmm + np.eye(xinduce.shape[0])*1e-5)
    cQ = np.linalg.solve(cKmm, Knm.T)
    Qnn_diag = np.sum(cQ.T*cQ.T, axis=1).squeeze()
    Ki_y = lmvn.woodbury_solve(cQ.T, np.log(Knn_diag-Qnn_diag+sig2_noise), yobs.squeeze())
    mu_s = np.dot(Qsn, Ki_y)

    Ki_Q = lmvn.woodbury_solve(cQ.T, np.log(Knn_diag-Qnn_diag+sig2_noise), Qsn.T)
    Sig_s = Kss_diag.squeeze() - np.sum(Qsn * Ki_Q.T, axis=1)
    return mu_s, Sig_s


def gp_integrated_predict(xobs, aobs, sig2, Sinv, sig2_noise, xstar):
    # integrated covariance for Cov(y_i, y_j)
    Kxx = integrated_sqe(xobs, xobs, sig2, Sinv)
    # standard GP covariance for cov(rho_i, rho_j)
    Kss = K_squared_exp(xstar, xstar, sig2, Sinv)
    # semi-integrated covariance for cov(y_i, rho_j)
    Ksx = semi_integrated_sqe(xobs, xstar, sig2, Sinv).T
    mu_grid, S_grid = condition_on(aobs, Kxx, Ksx, Kss, sig2_noise)
    return mu_grid, S_grid


def gp_sparse_integrated_predict(xobs, aobs, sig2_noise,
                                 kernel_obj, kernel_params,
                                 xstar, xinduce, xstar_is_integrated=False):
    """ Inducing point predictions:  
    Args:
        - xobs : observation location for integrated measurements
        - aobs : noisy integrated observations
        - sig2_noise: observation noise for each integrated measurement (fixed, known)
        - kernel_obj: kernel_fun object fron kernels.py
        - kernel_parmas: unpackable params (e.g. sig2, ell2 for sq expo)
        - xstar : prediction star locations (for both integrated and pointwise)
        - xinduce: location of inducing points
        - xstar_is_integrated: flag for predictions --- are pointwise or integrated

    Returns:
        - mu_star --- expected value of estar (integrated) or fstar (pointwise)
        - sig2_star --- var of estar (integrated) or fstar (pointwise)
    """
    M, N, S = xinduce.shape[0], xobs.shape[0], xstar.shape[0]

    # Inducing point Gram Matrix
    Kmm = kernel_obj.kfun(xinduce, xinduce, params=kernel_params)

    # Inducing-point to Observation Gram Matrix
    # This (and the diagonal calculation below) is the only difference
    # between the integrated obs and the noisy-evaluation obs above
    #Knm = semi_integrated_sqe(xobs, xinduce, sig2, Sinv)
    Knm = kernel_obj.k_semi(xinduce, xobs, params=kernel_params).T

    # Inducing point to prediction Gram Matrix
    if xstar_is_integrated:
        #print("Predicting ~integrated~ values!")
        Ksm = kernel_obj.k_semi(xinduce, xstar, params=kernel_params).T
        Ks_diag = kernel_obj.k_doubly_diag(xstar, params=kernel_params).squeeze()
    else:
        Ksm = kernel_obj.kfun(xstar, xinduce, params=kernel_params)
        Ks_diag = kernel_obj.kfun_diag(xstar, params=kernel_params)

    # following equation (11) in Snelson + Ghahramani 2007
    # we need to compute (Kn_tilde + sig2*I)^{-1} using the woodbury 
    # identity.  We can use woodbury solve for memory efficiency
    def Q(Knm, Kms):
        """ returns N x S matrix """
        return np.dot(Knm, np.linalg.solve(Kmm, Kms))

    # Compute the low rank factor 
    Qsn = Q(Ksm, Knm.T)
    cKmm = np.linalg.cholesky(Kmm + np.eye(xinduce.shape[0])*1e-5)
    cQ = np.linalg.solve(cKmm, Knm.T)
    Qnn_diag = np.sum(cQ.T*cQ.T, axis=1).squeeze()

    # diagonal of covariance is much smaller
    Knn_diag = kernel_obj.k_doubly_diag(xobs, params=kernel_params).squeeze()
    K_minus_Q = np.clip(Knn_diag - Qnn_diag, a_min=0., a_max=np.inf)
    lnv = np.log(K_minus_Q + sig2_noise)
    #print(Knn_diag)
    #print(Qnn_diag)
    Ki_y = lmvn.woodbury_solve(cQ.T, lnv, aobs.squeeze())
    mu_s = np.dot(Qsn, Ki_y)

    # compute predictive variance
    Ki_Q = lmvn.woodbury_solve(cQ.T, lnv, Qsn.T)
    Sig_s = Ks_diag.squeeze() - np.sum(Qsn * Ki_Q.T, axis=1)
    return mu_s, Sig_s


#####################################################
# Objective functions to tune kernel_fun params         #
#####################################################

def integrated_fitc_objective(kernel_params, kernel_obj,
                              xobs, aobs, xinduce, sig2_noise,
                              do_vfe=False):
    """ surrogate objective for the covariance function params theta
    Args:
        - kernel_params : covariance function parameters
        - xobs  : location of x observations
        - aobs  : noisy integrated observations
        - xinduce : location of inducing points
        - sig2_noise: noise level for observations
        - do_vfe : use variational free energy objective
    """
    # unpack kernel_fun parameters
    M, D = xinduce.shape
    N, _ = xobs.shape
    #sig2, ell2 = theta
    #Sinv = np.eye(D) / ell2

    # compute approximate log likelihood value
    # This (and the diagonal calculation below) is the only difference
    # between the integrated obs and the noisy-evaluation obs above
    Kmm = kernel_obj.kfun(xinduce, xinduce, params=kernel_params)
    Knm = kernel_obj.k_semi(xinduce, xobs, params=kernel_params).T

    # diagonal of covariance is much smaller
    Knn_diag = kernel_obj.k_doubly_diag(xobs, params=kernel_params).squeeze()
    #Knn_diag = K_integrated_squared_exp_diag_interp(xobs, sig2, Sinv).squeeze()

    # following equation (11) in Snelson + Ghahramani 2007
    # we need to compute (Kn_tilde + sig2*I)^{-1} using the woodbury 
    # identity.  We can use woodbury solve for memory efficiency
    def Q(Knm, Kms):
        """ returns N x S matrix """
        return np.dot(Knm, np.linalg.solve(Kmm, Kms))

    # Compute the low rank factor 
    cKmm = np.linalg.cholesky(Kmm + np.eye(xinduce.shape[0])*1e-5)
    cQ = np.linalg.solve(cKmm, Knm.T)
    Qnn_diag = np.sum(cQ.T*cQ.T, axis=1).squeeze()

    # following Equations 5,6,7 in Bauer, van der Wilk, Rasmussen 2016
    coef = (N/2) * np.log(2*np.pi)
    if do_vfe:
        G = np.ones(N) * sig2_noise
        T = Knn_diag - Qnn_diag
    else:
        G = Knn_diag - Qnn_diag + sig2_noise
        T = np.zeros(N)

    # compute complexity, data fit, and trace term
    complexity_penalty = .5*lmvn.woodbury_lndet(cQ.T, np.log(G))
    Ki_y = lmvn.woodbury_solve(cQ.T, np.log(G), aobs.squeeze())
    data_fit = np.sum(aobs.squeeze() * Ki_y)
    trace_term = np.sum((.5/sig2_noise) * T)
    #print(complexity_penalty, data_fit)
    return coef + complexity_penalty + data_fit + trace_term


def validation_fitc_objective(kernel_params, kernel_obj,
        xobs, aobs, xtest, atest, xinduce, sig2_noise):
    """ objective function that tunes on out of sample predictive values """
    M, D = xinduce.shape
    mu_s, Sig_s = gp_sparse_integrated_predict(xobs, aobs, sig2_noise,
        kernel_obj, kernel_params,
        xstar=xtest, xinduce=xinduce, xstar_is_integrated=True)
    lls = norm.logpdf(atest, loc=mu_s, scale=np.sqrt(Sig_s + sig2_noise))
    return lls


#####################
# Util Functions    #
#####################

#def K_squared_exp(x1, x2, sig2, Sinv):
#    """ K(x1,x2) = sig2*exp(-1/2 (x1-x2) Sinv (x1-x2)) """
#    dist = x1[:,None] - x2[None,:]
#    qterm = np.sum(np.matmul(dist, Sinv)*dist, axis=-1)
#    return sig2*np.exp(-.5*qterm)
#
#
#def K_squared_exp_diag(x1, sig2, Sinv):
#    return np.ones((x1.shape[0], 1)) * sig2
#
#
#def K_integrated_squared_exp_diag(x1, sig2, Sinv, origin=0.):
#    knn = []
#
#    # verbose if long
#    gen = pyprind.prog_bar(x1) if len(x1) > 1000 else x1
#    for x in gen:
#        xdir = x - origin
#        xdist = np.sqrt(np.sum(xdir**2))
#        def rayfun(alpha):
#            return semi_integrated_sqe_single(x,
#                (1-alpha)*origin + alpha*xdir, sig2, Sinv)*xdist
#        res = integrate.quad(rayfun, 0, 1)
#        knn.append(res[0])
#    return np.array(knn)
#
#
#def K_integrated_squared_exp_diag_interp(x1, sig2, Sinv, origin=0., num_grid_points=100):
#    """ computes the diagonal term, K(xi, xi), for the doubly integrated 
#    squared exponential kernel_fun for a set of Xis """
#    # compute all input distances to the origin
#    N, D = x1.shape
#    dist_to_origin = np.sqrt(np.sum(x1*x1, axis=-1))
#
#    # grid up distances, compute doubly integrated kernel_fun on this distance
#    dist_grid = np.linspace(1e-4, 1.15*np.max(dist_to_origin), num_grid_points)
#    xgrid = np.column_stack([dist_grid] +
#                            [np.zeros(len(dist_grid)) for d in range(D-1)])
#    Knn_grid = K_integrated_squared_exp_diag(xgrid, sig2, Sinv)
#
#    # interpolate the grid values for each observed distance value
#    Knn_interp = np.interp(dist_to_origin, dist_grid, Knn_grid)
#    return Knn_interp


def check_cached_differences():
    x1 = np.random.rand(200, 2) * 10
    sig2 = 1.2
    Sinv = np.eye(2)*.1
    knn_diag = K_integrated_squared_exp_diag(x1, sig2, Sinv)
    num_grid_points = 100
    knn_fast = K_integrated_squared_exp_diag_interp(x1, sig2, Sinv)
    print(np.max(np.abs(knn_diag - knn_fast)))
    print(np.mean(np.abs(knn_diag-knn_fast)/knn_diag))


def sample_gp(xpts, Kc, eps=None):
    eps = npr.randn(Kc.shape[0]) if eps is None else eps
    return np.dot(eps, Kc.T)


def condition_on(y, Kxx, Ksx, Kss, sig2_noise):
    """ MVN conditional distribution, computes p(s | y)
    for model where
            rho        ~ GP
            y_i | x_i  ~ rho(x_i) + N(0, sig2_noise)
    Args:
        Kxx = cov(rho(x_i), rho(x_j))
        Ksx = cov(rho(x_i), rho(x_star))  new points, with no observations
        Kss = cov(rho(x_star), rho(x_star))
    """
    nobs = Kxx.shape[0]
    Kxx_noise_inv = np.linalg.inv(Kxx + sig2_noise*np.eye(nobs))
    mu  = np.dot(Ksx, np.dot(Kxx_noise_inv, y))
    Sig = Kss - np.dot(Ksx, Kxx_noise_inv).dot(Ksx.T)
    return mu, Sig


###########################
# Integrated Observations #
###########################
from scipy.special import erf
from scipy.stats.distributions import norm
#def semi_integrated_sqe_single(xintegrated, x, sig2, Sinv):
#    """ assumes the origin is (0,0,0), computes int_0,1 K(xintegrated, x)
#       Returns matrix of size len(xintegrated) x len(x)
#    """
#    xdist = np.sqrt(np.sum(xintegrated**2))
#    a = np.dot(xintegrated, Sinv).dot(xintegrated)
#    b = np.dot(np.dot(x, Sinv), xintegrated)
#    c = np.dot(x, Sinv).dot(x)
#
#    coef = sig2*np.exp(b**2/(2*a) - c/2) * np.sqrt(2*np.pi/a)
#    ca = norm.cdf(1, loc=b/a, scale=np.sqrt(1/a))
#    cb = norm.cdf(0, loc=b/a, scale=np.sqrt(1/a))
#    return coef*(ca-cb)*xdist
#
#
#def semi_integrated_sqe_slow(xintegrated, x, sig2, Sinv):
#    n1, n2 = xintegrated.shape[0], x.shape[0]
#    Kix = np.zeros((n1, n2))
#    for i in pyprind.prog_bar(range(n1)):
#        for j in range(n2):
#            Kix[i,j] = semi_integrated_sqe_single(xintegrated[i], x[j], sig2, Sinv)
#    return Kix
#
#
#def semi_integrated_sqe(xintegrated, x, sig2, Sinv):
#    """ integrates over FIRST argument """
#    xdists = np.sqrt(np.sum(xintegrated*xintegrated, axis=-1))
#    a = np.sum(np.matmul(xintegrated, Sinv)*xintegrated, axis=-1)
#    xint_Si = np.matmul(xintegrated, Sinv)
#    b = np.matmul( xint_Si[:,None,None,:], x[None,:,:,None]).squeeze() # num_xi x num_x
#    c = np.sum(np.matmul(x, Sinv)*x, axis=-1)
#
#    # compute loc and scale for CDFs
#    scale = np.sqrt(1/a[:,None])
#    loc   = b / a[:,None]
#    coef = sig2*np.exp(b**2/(2*a[:,None]) - c/2) * np.sqrt(2*np.pi)*scale
#    ca = norm.cdf(1, loc=loc, scale=scale)
#    cb = norm.cdf(0, loc=loc, scale=scale)
#    return coef*(ca-cb) * xdists[:,None]
#
#
#def test_semi_integrated_vectorized():
#    xint = np.random.randn(10, 2)
#    x = np.random.randn(5, 2)
#    sig2 = .5
#    Sinv = np.eye(2)
#    Kslow = semi_integrated_sqe_slow(xint, x, sig2, Sinv)
#    Kfast = semi_integrated_sqe(xint, x, sig2, Sinv)
#    assert np.allclose(Kslow, Kfast)
#
#
#def integrated_sqe(x1, x2, sig2, Sinv):
#    n1, n2 = x1.shape[0], x2.shape[0]
#    Kxx = np.zeros((n1, n2))
#    for i in pyprind.prog_bar(range(n1)):
#        for j in range(n2):
#            # function to integrate 
#            ifun = lambda alpha : semi_integrated_sqe_single(x1[i], alpha*x2[j], sig2, Sinv)
#            res  = integrate.quad(ifun, 0, 1)
#            Kxx[i,j] = res[0]
#
#    # for each pair of X's, we integrate
#
#    # return Kxx_integrated --- use with condition_on as in a normal GP
#    return Kxx


################################################
# Sample (numerically) Integrated observations  #
#################################################

def make_ray_and_deltas(xpt, origin=0., num_cells=25):
    ray = np.row_stack([origin + alpha*(xpt-origin)
                    for alpha in np.linspace(0, 1, num_cells) ])
    delta = np.sqrt(np.sum((ray[1] - ray[0])**2))
    return ray, np.ones(len(ray))*delta

def sample_integrated_observations(xobs, xgrid, xfull, Kc, ray_info):
    ray_pts, ray_deltas, obs_idx = ray_info

    # sample everything jointly
    eps = npr.randn(len(xfull))
    fx_full = np.dot(eps, Kc.T)

    # separate out the field evaluations at the Rays, the star locations, and the full grid for vis
    fx_rays = fx_full[:len(ray_pts)]
    fx_obs  = fx_full[len(ray_pts):-len(xgrid)]
    fx_grid = fx_full[-len(xgrid):]

    # numerically integrate rays
    fint_obs = np.zeros(len(ray_deltas))
    for n in range(len(ray_deltas)):
        fray = fx_rays[obs_idx==n]
        fint_obs[n] = np.sum(fray*ray_deltas[n])

    return fint_obs, (fx_rays, fx_obs, fx_grid)


def generate_integrated_covariance(xobs, xgrid, sig2, Sinv):
    """ create covariance matrix over three sets of points
          - xobs  : observation points
          - xrays : discretized rays from Origin to each xobs point
          - xgrid : fixed grid for visualizing
    """
    # for each observation, make fixed grid along the rays
    ray_pts, ray_deltas, obs_idx = [], [], []
    for i, x in enumerate(xobs):
        rpts, rdelts = make_ray_and_deltas(x, num_cells=25)
        ray_pts.append(rpts)
        ray_deltas.append(rdelts)
        obs_idx.append([i]*len(rpts))

    ray_pts = np.row_stack(ray_pts)
    ray_deltas = np.row_stack(ray_deltas)
    obs_idx = np.concatenate(obs_idx)

    # construct covariance
    xfull = np.row_stack([ray_pts, xobs, xgrid])    
    Kfull = K_squared_exp(xfull, xfull, sig2, Sinv) + np.eye(len(xfull))*1e-5
    print("Inverting a %d x %d matrix ... "%(len(xfull), len(xfull)))
    Kc = np.linalg.cholesky(Kfull)
    return Kc, xfull, (ray_pts, ray_deltas, obs_idx)

