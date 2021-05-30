import torch
import numpy as np

def diag_kl_to_standard(m, S):
    """ computes KL div btw diagonal gaussian m, S to standard normal N(0, I)
    """
    val = torch.sum(S) + torch.sum(m*m) - torch.sum(torch.log(S)) - len(m)
    return .5*val

def kl_to_standard(m, S):
    _, lndet = torch.slogdet(S)  # TODO: why slogdet since det(S) should be positive?
    val = torch.trace(S) + torch.sum(m*m) - lndet - len(m)
    return .5*val

def block_kl_to_standard(blk_m, blk_S):
    # cholesky blk_S
    I  = torch.eye(blk_S.shape[1], dtype=blk_S.dtype, device=blk_S.device)  # (blk_size, blk_size)
    Schol = torch.cholesky(blk_S + 1e-4 * I)
    blk_lndets = torch.sum(
        torch.log(torch.diagonal(Schol, dim1=-2, dim2=-1)), dim=-1
    )
    lndet = 2.0 * torch.sum(blk_lndets)

    # trace
    n_blk, blk_size, _ = blk_S.shape
    D = n_blk*blk_size
    Strace = torch.sum(torch.diagonal(blk_S, dim1=-2, dim2=-1))
    val = Strace + torch.sum(blk_m*blk_m) - lndet - D
    return .5*val


#########################################################
# more normal model functions (used by SVGP model)      #
#########################################################
lntwopi = np.log(2*np.pi)
sqrt2pi = np.sqrt(2*np.pi)

def NormalLogPdf(y, loc, scale):
    return -.5*lntwopi - torch.log(scale) -(.5/(scale*scale))*(y-loc)**2

def kl_mvn(m0, S0, m1, S1):
    k = S0.shape[-1]
    S1_i_S0 = torch.solve(S0, S1)[0]
    trace_term = torch.trace(S1_i_S0)
    diff = m1-m0
    S1_i_diff = torch.solve(diff, S1)[0]
    quad_term = torch.sum(diff*S1_i_diff)
    det_term = torch.logdet(S1)-torch.logdet(S0)
    return .5*(trace_term + quad_term - k + det_term)

def kl_mvn_chol(m0, cS0, m1, cS1):
    """
    computes KL( (m0, S0)  || (m1, S1) )
        m0  : k x 1 tensor
        cS0 : k x k lower triangular scale matrix
        m1  : k x 1 mean tensor
        cS1 : k x k lower triangular scale matrix
    """
    k = cS0.shape[-1]
    lndetS0 = 2*cS0.diagonal().log().sum()
    lndetS1 = 2*cS1.diagonal().log().sum()
    det_term = lndetS1 - lndetS0

    diff = m1-m0
    sqrt_mahal, _ = torch.triangular_solve(diff, cS1, upper=False)
    quad_term = torch.sum(sqrt_mahal**2)

    tr_term, _ = torch.triangular_solve(cS0, cS1, upper=False)
    trace_term = torch.sum(tr_term*tr_term)

    return .5*(det_term + quad_term + trace_term - k)


def normal_cdf(x, loc, scale):
    sqrt2 = np.sqrt(2)
    return .5 * (1. + torch.erf( (x - loc) / (scale*sqrt2) ))


#################
# Gamma stats   #
#################

def lngamma_pdf(x, alpha, beta):
    """ 
    Args: 
        x     : gamma obs (> 0)
        alpha : shape parameter
        beta  : inverse scale parameter
    """
    return (alpha+1)*torch.log(x) - beta*x

def lngamma_pdf_lnx(lnx, alpha, beta):
    """ 
    log gamma pdf given LOG x observations
    Args:
        x     : gamma obs (> 0)
        alpha : shape parameter
        beta  : inverse scale parameter
    """
    return (alpha+1)*lnx - beta*lnx.exp()

def gamma_moments(alpha, beta):
    """ Converts gamma parameters into mean, variance
        alpha: shape; beta: inverse scale
    """
    return alpha/beta, alpha/(beta**2)

def gamma_params(mean, var):
    """ alpha: shape; beta: inverse scale."""
    beta = mean / var
    alpha = mean*beta
    return alpha, beta


if False:

    import scipy.stats.distributions as dists
    from ziggy.misc.stats import gamma_params
    mean = .1
    scales = np.linspace(.01, .1, 7)

    fig, ax = plt.figure(figsize=(8,6)), plt.gca()
    xgrid = np.linspace(.01, .5, 100)
    for s in scales:
        a, b = gamma_params(mean, s**2)
        g = dists.gamma(a=a, scale=1/b)
        lab = "%2.3f +- [%2.3f, %2.3f] (s=%2.3f)" % \
            (g.mean(), g.ppf(.025), g.ppf(.975), s)
        ps = g.pdf(xgrid)
        ax.plot(xgrid, ps, label=lab)

    ax.legend()


