import torch
from ziggy.kernels import Matern
from ziggy.misc import toeplitz_expanded
import os
import numpy as np

output_dir = "./output-solve-kn-experiment"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##################################################
# Setup --- kernel_fun, domain, vector for solve     #
##################################################


def check_cg_convergence(to_plot=True, compute_type="KinvV"):
    """
    Compute K^{-1} v for v a random vector
    Compute K^{-1/2} v = R^T K^{-1} v for v a random vector
    :param to_plot:
    :return:
    """
    available_compute_types = ['KinvV', 'RtKinvV']
    assert compute_type in available_compute_types, \
        "got compute type = {}, please choose among {}".format(compute_type, available_compute_types)

    torch.manual_seed(42)

    # construct kernel_fun
    kern = Matern(nu=2.5, length_scale=.5)

    # iterate over M x M grids
    num_vecs = 1
    # num_vecs = 25
    # Ms = [25, 50, 100]
    Ms = [(25, 25), (50,50), (100,100)]
    res_dict = {}

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: ", device)

    for Mx in Ms:
        print("Running {} x {} = {}".format(Mx[0], Mx[1], Mx[0] * Mx[1]))
        x1 = torch.linspace(0, 4, Mx[0], device=device)
        x2 = torch.linspace(-2, 2, Mx[1], device=device)

        # evaluate problem --- test 3d
        xgrids = [x1, x2]
        # xinduce_grids = [x1]
        M = np.prod([len(x) for x in xgrids])
        vec = torch.randn(num_vecs, M, device=device)

        kernel = lambda x, y: kern.forward(x, y, params=(1, .1))
        print("Solving a system of %d variables" % M)

        def make_callback():
            xs = []

            def callback(n, x):
                if (n > 100) and (n % 50 == 0):
                    print("  ... iter %d " % n)
                xs.append(x.detach().cpu())

            return callback, xs

        cb_cg, xs_cg = make_callback()
        mult_RT = True if compute_type == "RtKinvV" else False
        res = toeplitz_expanded.gram_solve(xgrids, kernel, vec,
                                           do_precond=False, tol=1e-10, maxiter=2000, callback=cb_cg, mult_RT=mult_RT)

        cb_pcg, xs_pcg = make_callback()
        res_pre = toeplitz_expanded.gram_solve(xgrids, kernel, vec,
                                               do_precond=True, tol=1e-10, maxiter=2000, callback=cb_pcg, mult_RT=mult_RT)

        res_dict[Mx] = {"cg": xs_cg, "pcg": xs_pcg}

    if to_plot:
        def sequence_error(xs_list):
            xs = torch.stack(xs_list)
            # mse + mae
            diff = xs - xs[-1:, :, :]
            mse = torch.sqrt(torch.mean(diff ** 2, dim=1)).numpy()
            mae = torch.mean(torch.abs(diff), dim=1).numpy()
            # get percentiles on these lines as well
            mse_cis = np.percentile(mse, [2.5, 50, 97.5], axis=1)
            mae_cis = np.percentile(mae, [2.5, 50, 97.5], axis=1)
            return mse_cis, mae_cis

        import matplotlib.pyplot as plt;

        plt.ion()
        import seaborn as sns;

        sns.set_style("white")
        sns.set_context("paper")
        sns.set(font_scale=1.15)

        def make_cg_pcg_comparison_plot(error_type="rmse"):
            fig, ax = plt.figure(figsize=(6, 4)), plt.gca()
            cg_col = "blue"
            pcg_col = "purple"
            marks = {Mx: x for Mx, x in zip(res_dict.keys(), ['o', 'v', 'X', 'd'])}
            lines = {Mx: x for Mx, x in zip(res_dict.keys(), ['-', '--', ':', '-.'])}

            for Mx in res_dict.keys():
                xs_mse_cg, xs_mae_cg = sequence_error(res_dict[Mx]['cg'])
                xs_mse_pcg, xs_mae_pcg = sequence_error(res_dict[Mx]['pcg'])

                # mae/mse
                if error_type == "rmse":
                    err_cg, err_pcg = xs_mse_cg, xs_mse_pcg
                elif error_type == "mae":
                    err_cg, err_pcg = xs_mae_cg, xs_mae_pcg

                # plot as a function of maximum iterations
                max_iters = np.max([err_cg.shape[1], err_pcg.shape[1]])
                cg_frac = err_cg.shape[1] / max_iters
                pcg_frac = err_pcg.shape[1] / max_iters

                # plot CG (0, 1)
                iters = np.linspace(0, cg_frac, err_cg.shape[1])
                ax.fill_between(iters, y1=err_cg[0], y2=err_cg[2],
                                alpha=.35, color=cg_col)
                ax.plot(iters, err_cg[1], lines[Mx],
                        label='cg (M={:,})'.format(Mx[0]*Mx[1]), c=cg_col)
                        #label="cg (M=%dx%d)" % (Mx[0], Mx[1]), c=cg_col)
                # plot PCG (0, frac_pcg)
                iters = np.linspace(0, pcg_frac, err_pcg.shape[1])
                ax.fill_between(iters, y1=err_pcg[0], y2=err_pcg[2],
                                alpha=.35, color=pcg_col)
                ax.plot(iters, err_pcg[1], lines[Mx],
                        label='pcg (M={:,})'.format(Mx[0]*Mx[1]), c=pcg_col)
                        #label="pcg (M=%dx%d)" % (Mx[0], Mx[1]), c=pcg_col)

            ax.set_yscale("log")
            handles, labels = ax.get_legend_handles_labels()
            #idx = np.arange(len(handles))
            idx = [0, 2, 4, 1, 3, 5]
            handles = [handles[i] for i in idx]
            labels = [labels[i] for i in idx]
            # sort both labels and handles by labels
            # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels, frameon=True, loc='upper right', ncol=2, framealpha=.78)
            ax.set_xlabel("frac. of CG iters")
            ax.set_ylabel(error_type.upper())
            return fig, ax

        fig, ax = make_cg_pcg_comparison_plot(error_type="rmse")
        fig.savefig(os.path.join(output_dir, "cg-pcg-comparison-mse.pdf"), bbox_inches='tight')

        fig, ax = make_cg_pcg_comparison_plot(error_type="mae")
        fig.savefig(os.path.join(output_dir, "cg-pcg-comparison-mae.pdf"), bbox_inches='tight')


if __name__=="__main__":
    check_cg_convergence()
