"""compare congergence of pcg and bidiag against cholesky solve
"""

import torch
import pandas as pd
import os
import argparse

import matplotlib.pyplot as plt;

plt.ion()
import seaborn as sns;

sns.set_style("white")
sns.set_context("paper")
sns.set(font_scale=1.15)

sns.set_style("whitegrid", {'axes.grid': False})


from ziggy.misc.toeplitz_expanded import ToeplitzMatmul, gram_solve
from ziggy.kernels import SqExp, Matern

parser = argparse.ArgumentParser(description='PCG vs Cholesky')
parser.add_argument('--wall-clock-time', action="store_true")

args, _ = parser.parse_known_args()
#print("Experiment script args: ", args)

wall_clock_time = args.wall_clock_time
use_cuda = torch.cuda.is_available()
device_name = "cuda" if use_cuda else "cpu"
dtype =torch.float32
device = torch.device("cuda") if use_cuda else torch.device('cpu')
print("Using device: ", device)


# cpu timing
import time


output_dir = "output-pcg-vs-cholesky"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

torch.manual_seed(42)

max_iter_cg = 2000

dim = 2
xlo, xhi = 0, 2
nobs = 200

# define xobs
xobs = torch.rand(nobs, dtype=dtype, device=device) * (xhi-xlo) + xlo
xobs = xobs.unsqueeze(-1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

kerns = [SqExp(), Matern(nu=2.5), Matern(nu=1.5), Matern(nu=0.5)]
names = ['SqExp', "Mat12", "Mat32", "Mat52"]

ninduce_list = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
#ninduce_list = [1e2, 1e3]

kernel_time_df_dict = {}

# cholesky does not work above 5e4, including 5e4
#ninduce_list = [100, 1000]
for kernel, name in zip(kerns, names):
    print("########################################")
    print("########## kern = {} ############".format(name))
    print("#######################################\n")
    svgp_time_list = {}
    hipgp_time_list = {}
    for ninduce in ninduce_list:

        ninduce = int(ninduce)
        print("###################################")
        print("M = {}".format(ninduce))
        M = ninduce
        Mprime = 2*(ninduce-1)
        # define xinduce
        xgrids = [torch.linspace(xlo, xhi, ninduce, dtype=dtype, device=device)]
        xxs = torch.meshgrid(*xgrids)
        xinduce = torch.stack([x.reshape(-1) for x in xxs], dim=-1)

        sig2, ell = 0.1, (xhi-xlo)/ninduce
        print("sig2={}, ell={}".format(sig2, ell))

        kernel_fun = lambda x, y: kernel(x, y, params=(sig2, ell))

        # compute Kun
        Kun = kernel_fun(xinduce, xobs) # (ninudce, nobs)

        # use hip-gp to solve
        print("Start hip-gp")
        if use_cuda and not wall_clock_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            # define topelitz matmul
            K_matmul = ToeplitzMatmul(xgrids, kernel_fun, batch_shape=Kun.shape[-1:])

            A_matmul = lambda x: K_matmul(x.t(), multiply_type="RTv").t()
            Astar_matmul = lambda x: K_matmul(x.t(), multiply_type="Rv").t()

            # kn = Kuu^{-1/2} Kun, a tensor of shape (ninduce, nobs)
            kn_2 = gram_solve(xgrids, kernel_fun, Kun.t(), maxiter=max_iter_cg, tol=1e-10, K_matmul=None,
                              mult_RT=True)
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            hipgp_time = start.elapsed_time(end)
        else:
            start = time.time()
            # define topelitz matmul
            K_matmul = ToeplitzMatmul(xgrids, kernel_fun, batch_shape=Kun.shape[-1:])

            A_matmul = lambda x: K_matmul(x.t(), multiply_type="RTv").t()
            Astar_matmul = lambda x: K_matmul(x.t(), multiply_type="Rv").t()

            # kn = Kuu^{-1/2} Kun, a tensor of shape (ninduce, nobs)
            kn_2 = gram_solve(xgrids, kernel_fun, Kun.t(), maxiter=max_iter_cg, K_matmul=None, mult_RT=True)
            hipgp_time = time.time() - start


        print("took {} time {}\n".format(device_name, hipgp_time))
        hipgp_time_list["M{}".format(ninduce)] = hipgp_time

        # use Cholesky to compute true
        if ninduce < 5e4:

            print("Start cholesky")
            if use_cuda and not wall_clock_time:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                Kuu = kernel_fun(xinduce, xinduce)
                I = torch.eye(Kuu.shape[0], device=device, dtype=dtype)
                cKuu = torch.cholesky(Kuu + I * 1e-4, upper=False)
                true_kn_cholesky = torch.triangular_solve(Kun, cKuu, upper=False)[0]  # (ninduce, bsz)
                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()

                svgp_time = start.elapsed_time(end)

            else:
                start = time.time()
                Kuu = kernel_fun(xinduce, xinduce)
                I = torch.eye(Kuu.shape[0], device=device, dtype=dtype)
                cKuu = torch.cholesky(Kuu + I * 1e-4, upper=False)
                true_kn_cholesky = torch.triangular_solve(Kun, cKuu, upper=False)[0]  # (ninduce, bsz)
                svgp_time = time.time() - start

            print("took {} time {}\n".format(device_name, svgp_time))
            svgp_time_list["M{}".format(ninduce)] = svgp_time

    df = pd.DataFrame(dict(svgp=svgp_time_list, hipgp=hipgp_time_list))
    kernel_time_df_dict[name] = df


print("####################################################")
print("Finish running....")
if use_cuda and args.wall_clock_time:
    print("Reporting wall clock time on {}".format(device_name))
else:
    print("Reporting cuda time")
for kernel_name, df in kernel_time_df_dict.items():
    print("Kenrel = {}".format(kernel_name))
    print(df)
    print("\n")
    if wall_clock_time:
        df.to_csv("wall_clock_time_summary_pcg_vs_cholesky_{}.csv".format(kernel_name))
    else:
        df.to_csv("cuda_time_summary_pcg_vs_cholesky_{}.csv".format(kernel_name))


