import torch
from torch.autograd import Function

#from gpytorch.utils.toeplitz import sym_toeplitz_derivative_quadratic_form  # this requires gpytorch version=1.1.1
from ziggy.misc.gpt_toeplitz import sym_toeplitz_derivative_quadratic_form


class InvMatmul(Function):
    @staticmethod
    def forward(ctx, toeplitz_tensor, column, right_tensor, do_precond, maxiter, tol):

        # compute d = K^{-1} y
        assert right_tensor.ndimension() == 2, right_tensor.ndimension()
        vec = right_tensor

        ctx.toeplitz_tensor = toeplitz_tensor

        with torch.no_grad():
            solves = toeplitz_tensor._solve(vec, do_precond=do_precond, maxiter=maxiter, tol=tol, callback=None)

        maxiter = torch.tensor(maxiter, dtype=torch.int)
        tol = torch.tensor(tol, dtype=torch.float)
        ctx.save_for_backward(solves, maxiter, tol)

        return solves

    @staticmethod
    def backward(ctx, grad_output):

        right_solves, maxiter, tol = ctx.saved_tensors
        maxiter = int(maxiter)
        tol = float(tol)

        if any(ctx.needs_input_grad):
            # compute left solves
            left_solves = InvMatmul.apply(ctx.toeplitz_tensor, ctx.toeplitz_tensor.column, grad_output, True, maxiter, tol)

        column_grad = None
        if ctx.needs_input_grad[1]:
            #print("left solves shape", left_solves.shape)
            #print("right solves shape", right_solves.shape)
            left_vecs = torch.cat([left_solves, right_solves], 0).t()
            right_vecs = torch.cat([right_solves, left_solves], 0).t().mul(-0.5)

            #print("left vecs shape", left_vecs.shape)
            #print("right vecs shape", right_vecs.shape)

            if left_vecs.ndimension() == 1:
                left_vecs = left_vecs.unsqueeze(1)
                right_vecs = right_vecs.unsqueeze(1)

            column_grad = sym_toeplitz_derivative_quadratic_form(left_vecs, right_vecs)

            if column_grad.dim() > ctx.toeplitz_tensor.column.dim():
                column_grad = column_grad.view(-1, *ctx.toeplitz_tensor.column.shape).sum(0)
            #print(column_grad.shape)

        right_grad = None
        if ctx.needs_input_grad[2]:
            right_grad = left_solves

        #print("column_grad", column_grad)

        return None, column_grad, right_grad, None, None, None
