import numpy as np
import torch


def conj_grad(A_mul, b, precond=None, maxiter=20, tol=1e-10, callback=None):
    """ computes by computing A^{-1} b by solving system Ax = b
    using conjugate gradient updates 
    Args:
        - A_mul   : Function handle that left multiplies a matrix A by
            vector (or column stack of vectors) b.  A is MxM and b is MxL
        - b       : M x L matrix
        - precond : Preconditioner function handle that left multiplies a
            vector (or stack of vectors) v that is MxL.  Note that the best
            preconditioner will look a lot like A^{-1}.
    """
    if precond is None:
        precond = lambda x: x

    x  = torch.zeros_like(b)
    r  = b - A_mul(x)
    z  = precond(r)
    p  = z

    for n in range(maxiter):
        rs = torch.sum(r*z, dim=0)
        Ap = A_mul(p)
        alpha = rs / torch.sum(p*Ap, dim=0)
        x = x + alpha[None,:] * p
        r = r - alpha[None,:] * Ap
        rnew = torch.sum(r*r, dim=0)
        if torch.all(torch.sqrt(rnew) < tol):
            break

        z = precond(r)
        beta = torch.sum(z*r, dim=0) / rs
        p = z + beta[None,:]*p

        if callback is not None:
            callback(n, x)

    return x


def conj_grad2(A_mul, b, precond=None, maxiter=20, tol=1e-10, callback=None):
    """ computes by computing A^{-1} b by solving system Ax = b
    using conjugate gradient updates
    Args:
        - A_mul   : Function handle that left multiplies a matrix A by
            vector (or column stack of vectors) b.  A is MxM and b is (bsz, M)
        - b       : (bsz, M) matrix
        - precond : Preconditioner function handle that left multiplies a
            vector (or stack of vectors) v that is MxL.  Note that the best
            preconditioner will look a lot like A^{-1}.
    """
    if precond is None:
        precond = lambda x: x

    x  = torch.zeros_like(b)
    r  = b - A_mul(x)
    z  = precond(r)
    p  = z

    for n in range(maxiter):
        rs = torch.sum(r*z, dim=1)  # (bsz, )
        Ap = A_mul(p)
        alpha = rs / torch.sum(p*Ap, dim=1)  # (bsz, )
        x = x + alpha.unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1) * Ap
        rnew = torch.sum(r*r, dim=1)  # (bsz,)
        if torch.all(torch.sqrt(rnew) < tol):
            break

        z = precond(r)
        beta = torch.sum(z*r, dim=1) / rs  # (bsz, )
        p = z + beta.unsqueeze(-1)*p

        if callback is not None:
            callback(n, x)

    return x


