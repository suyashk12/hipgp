import torch
from ziggy.misc.tridiagonal_solve import tridiagonal_solve


def golub_kahan_bidiag(A_matmul, Astar_matmul, matrix_shape, max_iter, dtype, device, b, tol=1e-5, run_all=False):
    """
     Perform Golub-Kahan bidiagonalization procedure
    Compute column-orthogonal matrix UJ, VJ such that UJ^* A VJ  = BJ is in bidiagonal form
    ref: http://www.netlib.org/utk/people/JackDongarra/etemplates/node198.html + full re-orthognalization
    for re-orthogalization, see similar example in
    https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/lanczos.py

    A: (M, N), U: (M, J), V: (M, J), where M, N = matrix_shape, and J = number of iterations
    :param A_matmul: a functional callable object that multiplies A by a vector of shape (N, bsz)
    and returns a tensor of shape (M, bsz)
    :param Astar_matmul: a functional callable object that multiplies A* by a vector of shape (M, bsz)
    and returns a tensor of shape (N, bsz)
    :param max_iter:
    :param dtype:
    :param device:
    :param b: a tensor of shape (N, bsz), used for the first column of V (normalized)
    :param tol:
    :param run_all: run the full process wihtout early stopping
    :return: VJ, alphasJ, betasJ if not run_all
    else, UJ, VJ, alphasJ, betasJ
    """

    M, N = matrix_shape
    Nb, bsz = b.shape
    assert Nb == N, "vector b shape = {}, does not match matrix shape = {}".format(b.shape, matrix_shape)

    v0 = b / torch.norm(b, 2, dim=0, keepdim=True)
    beta0 = 0
    vk = v0
    beta_km1 = beta0
    ukm1 = 0

    U = torch.zeros(max_iter, M, bsz, dtype=dtype, device=device)
    V = torch.zeros(max_iter, N, bsz, dtype=dtype, device=device)
    alphas = torch.zeros(max_iter, bsz, dtype=dtype, device=device)
    betas = torch.zeros(max_iter, bsz, dtype=dtype, device=device)

    V[0].copy_(v0)
    for k in range(0, max_iter):
        uk = A_matmul(vk) - beta_km1 * ukm1  # (2N, bsz)
        if k == 0:
            alpha_k = torch.norm(uk, 2, dim=0, keepdim=False)  # (bsz, )
            uk.div_(alpha_k.unsqueeze(0))
            U[k].copy_(uk)
            alphas[k].copy_(alpha_k)
            could_reorthogonalize_U = True
        else:
            # r <- r - U(U^T r)
            correction = uk.unsqueeze(0).mul(U[:k]).sum(-2, keepdim=True)  # (k, 1, bsz)
            correction = U[:k].mul(correction).sum(0)  # (2N, bsz)
            uk.sub_(correction)
            alpha_k = torch.norm(uk, 2, dim=0, keepdim=False)  # (bsz, )
            uk.div_(alpha_k.unsqueeze(0))

            # run more reorthoganlization if necessary
            inner_products = U[:k].mul(uk.unsqueeze(0)).sum(-2)
            could_reorthogonalize_U = False
            for _ in range(10):
                if torch.sum(inner_products) < tol:
                    could_reorthogonalize_U = True
                    break
                correction = uk.unsqueeze(0).mul(U[:k]).sum(-2, keepdim=True)  # (k, 1, bsz)
                correction = U[:k].mul(correction).sum(0)  # (2N, bsz)
                uk.sub_(correction)
                uk_norm = torch.norm(uk, 2, dim=0, keepdim=True)  # (bsz, )
                uk.div_(uk_norm)
                inner_products = U[:k].mul(uk.unsqueeze(0)).sum(-2)
            U[k].copy_(uk)
            alphas[k].copy_(alpha_k)

        vkp1 = Astar_matmul(uk) - alpha_k * vk

        correction = vkp1.unsqueeze(0).mul(V[:k + 1]).sum(-2, keepdim=True)  # (k, 1, bsz)
        correction = V[:k + 1].mul(correction).sum(0)  # (N, bsz)
        vkp1.sub_(correction)
        beta_k = torch.norm(vkp1, 2, dim=0, keepdim=False)
        betas[k].copy_(beta_k)

        if k == max_iter - 1:
            # no need to compute Vj and re-orthoganlize
            break
        # full re-orthogonalize
        vkp1.div_(beta_k.unsqueeze(0))

        # run more reorthoganlization if necessary
        inner_products = V[:k + 1].mul(vkp1.unsqueeze(0)).sum(-2)
        could_reorthogonalize_V = False
        for _ in range(10):
            if torch.sum(inner_products) < tol:
                could_reorthogonalize_V = True
                break
            #print("vk", k, " ", _)
            correction = vkp1.unsqueeze(0).mul(V[:k + 1]).sum(-2, keepdim=True)  # (k, 1, bsz)
            correction = V[:k + 1].mul(correction).sum(0)  # (N, bsz)
            vkp1.sub_(correction)
            vkp1_norm = torch.norm(vkp1, 2, dim=0, keepdim=True)  # (bsz, )
            vkp1.div_(vkp1_norm)
            inner_products = V[:k + 1].mul(vkp1.unsqueeze(0)).sum(-2)

        if not run_all:
            if torch.sum(beta_k.abs() > 1e-6) == 0 or not could_reorthogonalize_V or not could_reorthogonalize_U:
                break

        V[k + 1].copy_(vkp1)
        ukm1 = uk
        vk = vkp1
        beta_km1 = beta_k

    J = k + 1
    # permute
    U = U[:J].permute(1, 0, -1).contiguous()  # (M, J, bsz)
    V = V[:J].permute(1, 0, -1).contiguous()  # (N, J, bsz)
    alphas = alphas[:J]
    betas = betas[:J]

    if not run_all:
        return V, alphas, betas
    return U, V, alphas, betas


def bidiag_solve(A_matmul, Astar_matmul, matrix_shape, max_iter, dtype, device, b, tol=1e-5):
    """
    Solve c = K^{-1/2} b = (A^T UN VN^T)^{-1} b for batch b
    we have access to non-square root to K: A^TA = K, where A is a 2N x N matrix
    b is a tensor of shape (N, bsz)
    return: (N, bsz)
    """

    N, bsz = b.shape

    VJ, alphasJ, betasJ = golub_kahan_bidiag(A_matmul, Astar_matmul, matrix_shape,
                                             max_iter, dtype, device, b, tol) # VJ: (N, J, bsz)
    J = VJ.shape[1]

    diagonal_elements = alphasJ ** 2 + betasJ ** 2  # (J, bsz)
    upper_diagonal_elements = alphasJ[1:] * betasJ[:-1]  # (J-1, bsz)

    alpha1_b_e1 = torch.zeros(J, bsz, dtype=dtype, device=device)
    alpha1_b_e1[0].copy_(alphasJ[0] * torch.norm(b, 2, dim=0, keepdim=False))
    d = tridiagonal_solve(diagonal_elements, upper_diagonal_elements, alpha1_b_e1)  # (J, bsz)
    c = torch.sum(VJ.mul(d.unsqueeze(0)), dim=1)  # (N, bsz)

    return c


def bidiag_solve_with_callback(A_matmul, Astar_matmul, matrix_shape, max_iter, dtype, device, b, tol=1e-5, callback=None):
    """
    Solve c = K^{-1/2} b = (A^T UN VN^T)^{-1} b for batch b
    we have access to non-square root to K: A^TA = K, where A is a 2N x N matrix
    b is a tensor of shape (N, bsz)
    return: (N, bsz)
    """

    N, bsz = b.shape

    VJ, alphasJ, betasJ = golub_kahan_bidiag(A_matmul, Astar_matmul, matrix_shape,
                                             max_iter, dtype, device, b, tol) # VJ: (N, J, bsz)

    J = VJ.shape[1]
    print("J = {}, max_iter={}".format(J, max_iter))
    for j in range(2, J):

        J = VJ.shape[1]

        diagonal_elements = alphasJ[:j] ** 2 + betasJ[:j] ** 2  # (j, bsz)
        upper_diagonal_elements = alphasJ[:j][1:] * betasJ[:j][:-1]  # (j-1, bsz)

        alpha1_b_e1 = torch.zeros(j, bsz, dtype=dtype, device=device)
        alpha1_b_e1[0].copy_(alphasJ[0] * torch.norm(b, 2, dim=0, keepdim=False))
        d = tridiagonal_solve(diagonal_elements, upper_diagonal_elements, alpha1_b_e1)  # (j, bsz)
        c = torch.sum(VJ[:, :j].mul(d.unsqueeze(0)), dim=1)  # (N, bsz)
        if callback is not None:
            callback(j, c)

    return c