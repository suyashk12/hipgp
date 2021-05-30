import torch
import numpy as np
import matplotlib.pyplot as plt

from ziggy.exact_gp_1d_derivatives import *

torch.manual_seed(42)


import torch.nn as nn

def make_two_dim_synthetic_function(weight_std=35, hidden_dim=10, seed=42):
    torch.manual_seed(seed)
    class TestFun(nn.Module):
        def __init__(self):
            super(TestFun, self).__init__()
            self.lin = nn.Linear(1, hidden_dim)
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
    #fun.double()

    def fun_npy(x):
        return fun(torch.FloatTensor(x)).detach().numpy()

    return fun, fun_npy


func_complexity = 'hard'
weight_std, hidden_dim = {
    'simple': (9, 9), #(8, 8), #(10, 10),
    'medium': (35, 10),
    'hard'  : (40, 25),
  }[func_complexity]

torch.manual_seed(42)
ftrue_pt, ftrue = make_two_dim_synthetic_function(weight_std, hidden_dim)


def get_observations(x, fun):
    x = torch.tensor(x.clone().detach().numpy(), requires_grad=True)
    f = fun(x.unsqueeze(-1))
    for i in range(len(x)):
        f[i].backward(retain_graph=True)
    fprime = x.grad
    return f.detach(), fprime


kwargs = dict(xlo=-1, xhi=1)
xlo, xhi = kwargs.get("xlo", -1), kwargs.get("xhi", 1)
gridnum = kwargs.get("gridnum", 256)
xgrid = torch.linspace(xlo, xhi, gridnum)

fgrid, fprime_grid = get_observations(xgrid, ftrue_pt)

# half observations, half derivative observations
nlatent = 1000
nprime = 10
xlatent, _ = torch.sort(torch.rand(nlatent) * 2-1)
xprime, _ = torch.sort(torch.rand(nprime) * 2 -1)

f = ftrue_pt(xlatent.unsqueeze(1))
f = f.detach()

f_at_xprime, fprime = get_observations(xprime, ftrue_pt)

# add observation noise
obs_std = torch.tensor(0.05, dtype=torch.float)
derivative_std = torch.tensor(0.01, dtype=torch.float)
ylatent = f.squeeze() + torch.randn(nlatent) * obs_std
yprime = fprime + torch.randn(nprime) * derivative_std

M = 100
u = torch.linspace(-1, 1, M)

sig2_init = 0.2
ell_init = 0.05
log_sig2 = torch.tensor(np.log(sig2_init), dtype=torch.float32, requires_grad=True)
log_ell = torch.tensor(np.log(ell_init), dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam(params=[log_ell, log_sig2], lr=1e-3)

niters = 10

elbo_list = []
whitened_type = 'cholesky'
#whitened_type = 'ziggy'
for i in range(niters):
    optimizer.zero_grad()

    with torch.no_grad():
        sig2, ell = torch.exp(log_sig2), torch.exp(log_ell)
        m, S = svgp_batch_solve(u, xprime, yprime, xlatent, ylatent, sig2, ell,
                                derivative_obs_noise_std=derivative_std, obs_noise_std=obs_std,
                                whitened_type=whitened_type, maxiter=20)
    sig2, ell = torch.exp(log_sig2), torch.exp(log_ell)
    elbo = compute_elbo(u, m, S, xprime, yprime, xlatent, ylatent, sig2, ell, derivative_std, obs_std, batch_size=-1,
                         whitened_type=whitened_type, maxiter=20, precond=True, tol=1e-8, print_debug_info=False)
    print("iter i, elbo = {}".format(elbo.detach().numpy()))
    print("iter i, sig2 = {:.8f}, ell = {:.8f}".format(sig2, ell))
    loss = - elbo / 10000

    loss.backward()
    optimizer.step()
    elbo_list.append(elbo.detach().cpu().numpy())

