import torch


def tridiagonal_solve(d, c, b):
    """
    Solve symmetric tridiagonal system: Ax = b
    where the diagonal elements of A is d, and the upper diagonal elements of A is c
    :param d: (N, bsz)
    :param c: (N, bsz)
    :param b: (N, bsz)
    :return: x: (N, bsz). the solution to Ax = b
    """
    tol = 1e-16

    N, bsz = d.shape

    p = torch.zeros(N, bsz, dtype=d.dtype, device=d.device)
    q = torch.zeros(N-1, bsz, dtype=d.dtype, device=d.device)
    if torch.any(torch.abs(d[0]) < tol):
        raise ValueError("Method failed, d[0] < {}".format(tol))

    p[0].copy_(d[0])
    q[0].copy_(c[0]/d[0])

    for k in range(1, N-1):
        p[k].copy_(d[k] - c[k-1] * q[k-1])
        if torch.any(torch.abs(p[k]) < tol):
            raise ValueError("Method failed, p[{}] < {}".format(k, tol))
        q[k].copy_(c[k]/p[k])
    p[-1] = d[-1] - c[-1] * q[-1]
    if torch.any(p[-1].abs() < tol):
        raise ValueError("Method failed, p[-1] < {}".format(tol))

    y = torch.zeros(N, bsz, dtype=d.dtype, device=d.device)
    x = torch.zeros(N, bsz, dtype=d.dtype, device=d.device)
    y[0].copy_(b[0] / p[0])
    for k in range(1, N):
        y[k].copy_((b[k] - c[k-1] * y[k-1])/ p[k])
    x[-1].copy_(y[-1])
    for k in range(N-2, -1, -1):
        x[k].copy_(y[k] - q[k] * x[k+1])
    return x

