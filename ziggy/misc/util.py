"""
Utility functions for running synthetic/simulated experiments; saving
models + predictions, etc
"""

from scipy import integrate
import numpy as np
import torch
import datetime
import json


def print_vec(name, vec, np=False):
    if np:
        print("{} max = {}, min = {}, mean = {}".format(name, np.max(np.abs(vec)),
                                                        np.min(np.abs(vec)),
                                                        np.mean(np.abs(vec))))
    else:
        print("{} max = {}, min = {}, mean = {}".format(name, torch.max(torch.abs(vec)),
                                                            torch.min(torch.abs(vec)),
                                                        torch.mean(torch.abs(vec))))


def add_date_time(s=""):
    """
    @author: danielhernandez
    Adds the current date and downsampled_t at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16] + date[17:19]
    return s + '_D' + date


class NumpyEncoder(json.JSONEncoder):
    # Special json encoder for numpy types
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                      np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                        np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def integrated_obs(xobs, ftrue, origin=0.):
    """ numerically integrate observations (for synthetic data)

    Args:
        - xobs : N x D (e.g. 3-dimensional) list of points; integration
            backstop from 0 to xobs[n]
        - ftrue: scalar-valued function, takes in D-dimensional point

    Returns:
        - es : N x 1 array of numerically integrated values (from 0 to each x)
    """
    import pyprind
    es = []
    origin = np.array([[0., 0.]])
    for x in pyprind.prog_bar(xobs):
        xdir = x[None,:] - origin
        xdist = np.sqrt(np.sum(xdir**2))
        def rayfun(alpha):
            return ftrue( (1-alpha)*origin + alpha*xdir )[0]
        res = integrate.quad(rayfun, a=0., b=1., limit=100)
        es.append(res[0]*xdist)
    return np.array(es)


#########################################
# blocking up inducing points ---       #
#########################################

def define_block_chunks(xgrids, chunk_sizes):
    """ creates neighboring chunks of blocks of size = product of chunk_size in each dimension
    """
    # dimension checking
    ndim = len(xgrids)
    ndim_2 = len(chunk_sizes)
    assert ndim == ndim_2, "xgrids ndim = {}, chunk_sizes ndim = {}".format(ndim, ndim_2)
    assert (len(xgrids)==2) or (len(xgrids)==3), "only 2d or 3d inputs"
    for d, (x, chunk_size) in enumerate(zip(xgrids, chunk_sizes)):
        assert (len(x)%chunk_size) == 0, "xgrid-{}={} not divis by chunk_size={}".format(d, len(x), chunk_size)

    # chunk indices of each dimension by chunk size, save each list
    xyz_chunks = [torch.split(torch.arange(len(x)), chunk_size) for x, chunk_size in zip(xgrids, chunk_sizes)]

    blk_idx = []
    if len(xgrids) == 2:
        # create all pairs
        xchunks, ychunks = xyz_chunks
        for bx in xchunks:
            for by in ychunks:
                xxi, yyi = torch.meshgrid(bx, by)
                gidx = xxi*len(xgrids[1]) + yyi
                blk_idx.append(gidx.flatten())

    elif len(xgrids) == 3:
        # create all triples
        xchunks, ychunks, zchunks = xyz_chunks
        for bx in xchunks:
            for by in ychunks:
                for bz in zchunks:
                    xxi, yyi, zzi = torch.meshgrid(bx, by, bz)
                    gidx = xxi*(len(xgrids[1])*len(xgrids[2])) + yyi*len(xgrids[2]) + zzi
                    blk_idx.append(gidx.flatten())

    # stack into tensor of size (num_blocks x block_size)
    blk_idx = torch.stack(blk_idx, dim=0)

    # create easy transformation between BLOCK ordering and TOEPLITZ ordering
    # and back
    def to_blocks(m):
        # batched
        return m[..., blk_idx]

    flat_blk_idx = blk_idx.flatten()
    flat_blk_idx_reverse = torch.argsort(flat_blk_idx)
    def from_blocks(block_m):
        """first dimension is batch dimension"""
        return block_m.flatten(start_dim=1)[..., flat_blk_idx_reverse]

    return blk_idx, to_blocks, from_blocks

    # debug plot in 2d
    if False:

        # plot all inducing pts
        import matplotlib.pyplot as plt; plt.ion()
        fig, ax = plt.figure(figsize=(8,6)), plt.gca()
        ax.scatter(xx.flatten(), yy.flatten(), s=5, c='grey')

        # now we can index into inducing pts like this and acess
        # them in block order!
        pts = inducing_pts[blk_idx,:]

        # plot each block
        for bi in range(blk_idx.shape[0]):
            #bidx = blk_idx[bi,:]
            #pts_b = inducing_pts[bidx,:]
            pts_b = pts[bi,:]
            ax.scatter(pts_b[:,0], pts_b[:,1], label='bi = %d'%bi)

        ax.legend()
        plt.close("all")


def define_interleaved_blocks(xgrids, block_size):
    """ takes a list of inducing points (represented by grids along
    each dimension) and a target block_size, and creates a list
    of distinct blocks.  We note that there are two important orders
    of the M inducing points

        - "global ordering": meshgrid ordering of all inducing points,
          defined by `xx, yy, zz = meshgrid(xgrids)`.  This is like
          the C order in pytorch by default.
          This is the "global" ordering
        - "block ordering": ordering of points within each block --- 
          this is represented by a list of global order indices into a 
          two-dim array, e.g.

            blk_order = [ [0, 4,  8, 12],
                          [1, 5,  9, 13],
                          [2, 6, 10, 14],
                          [3, 7, 11, 15] ]

    This ordering corresponds to how we store the block-diagonal 
    S matrix (each row here corresponds to a 4x4 covariance matrix).
    """
    # create blocks such that each side is divisible

    # enumerate all inducing pts
    num_blocks = 4
    xx, yy = torch.meshgrid(xgrids)
    inducing_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # enumerate list ordering
    blk_list, blk_idx = [], []
    for bx in range(num_blocks):
        for by in range(num_blocks):
            xidx = torch.arange(bx, len(xgrids[0]), num_blocks)
            yidx = torch.arange(by, len(xgrids[1]), num_blocks)
            xxi, yyi = torch.meshgrid(xidx, yidx)
            #blk_list.append(torch.stack([xxi.flatten(), yyi.flatten()], 1))
            gidx = xxi*len(xgrids[0]) + yyi
            blk_idx.append(gidx.flatten())

    # stack block indices into a (num_blocks x blk_size) matrix
    blk_idx = torch.stack(blk_idx, dim=0)

    return blk_idx

    # debug plot in 2d
    if False:

        # plot all inducing pts
        import matplotlib.pyplot as plt; plt.ion()
        fig, ax = plt.figure(figsize=(8,6)), plt.gca()
        ax.scatter(xx.flatten(), yy.flatten(), s=5, c='grey')

        # now we can index into inducing pts like this and acess
        # them in block order!
        pts = inducing_pts[blk_idx,:]

        # plot each block
        for bi in range(blk_idx.shape[0]):
            #bidx = blk_idx[bi,:]
            #pts_b = inducing_pts[bidx,:]
            pts_b = pts[bi,:]
            ax.scatter(pts_b[:,0], pts_b[:,1], label='bi = %d'%bi)

        ax.legend()
        plt.close("all")


def batch_indices(it, num_batches, batch_size, total_size):
    idx = it % num_batches
    return slice(idx*batch_size, min((idx+1)*batch_size, total_size))



