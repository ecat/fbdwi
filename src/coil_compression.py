import sigpy as sp
import numpy as np    
import matplotlib.pyplot as plt
import cupy as cp
from utils import match_device

def get_gcc_matrices_2d(ksp_xyzc, ncc):
    # 2d kspace in xy, and slices are z
    nx, _, nz, nc = ksp_xyzc.shape
    xp = sp.get_array_module(ksp_xyzc)
    cc_matrices_aligned_zxcc = xp.zeros((nz, nx, nc, ncc), dtype=np.complex64)

    for zz in range(nz):
        cc_matrices_aligned_zxcc[zz, ...] = get_cc_matrix(ksp_xyzc[:, :, zz, np.newaxis, :], ncc)

    return cc_matrices_aligned_zxcc

def apply_gcc_matrices_2d(ksp_in_xyzc, gcc_matrices_zxcc):
    nx, ny, nz, nc = ksp_in_xyzc.shape
    ncc = gcc_matrices_zxcc.shape[-1]
    assert nz == gcc_matrices_zxcc.shape[0], "Did not provide correct dimensions for gcc matrices"
    
    xp = sp.get_array_module(ksp_in_xyzc)
    hybrid_data_out_xyzc = xp.zeros((nx, ny, nz, ncc), dtype=np.complex64)
    readout_axis = 0
    hybrid_data_in_xyzc = sp.ifft(ksp_in_xyzc, axes=(readout_axis,))

    for zz in range(nz):
        hybrid_data_out_xyzc[:, :, zz, :] = xp.squeeze(apply_gcc_matrices(hybrid_data_in_xyzc[:, :, zz, np.newaxis, :], gcc_matrices_zxcc[zz, :, :], readout_axis))

    return sp.fft(hybrid_data_out_xyzc, axes=(readout_axis,))

def get_gcc_matrices_3d(ksp_xyzc, ncc):
    # get cc matrices for every readout slice
    nx, _, _, nc = ksp_xyzc.shape
    xp = sp.get_array_module(ksp_xyzc)
    im_x_ksp_yzc = sp.ifft(ksp_xyzc, axes=(0,))
    cc_matrices_init_xcc = xp.zeros((nx, nc, ncc), dtype=np.complex64)    
    cc_matrices_aligned_xcc = xp.zeros_like(cc_matrices_init_xcc)
    
    for xx in range(nx):
        cc_matrices_init_xcc[xx, :, :] = get_cc_matrix(im_x_ksp_yzc[xx, ...], ncc)

    # align matrices along readout dimension https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3396763/    
    cc_matrices_aligned_xcc[0, :, :] = cc_matrices_init_xcc[0, :, :]

    for xx in range(1, nx):
        A_x = cc_matrices_init_xcc[xx, :, :]
        A_xm1 = cc_matrices_aligned_xcc[xx - 1, :, :]
        C_x = xp.matmul(xp.conj(A_x.T), A_xm1)
        U_x, S_x, Vh_x = xp.linalg.svd(C_x)
        P_x = xp.matmul(U_x, Vh_x)
        cc_matrices_aligned_xcc[xx, :, :] = xp.matmul(A_x, P_x)

    return cc_matrices_aligned_xcc


def get_cc_matrix(data_in_c_last_axis, ncc, do_plot=False):
    xp = sp.get_array_module(data_in_c_last_axis)
    nc = data_in_c_last_axis.shape[-1]
    u, cc_singular_values, vh = xp.linalg.svd(np.reshape(data_in_c_last_axis, (-1, nc)), full_matrices=False)
    v = xp.conj(vh.T)
    cc_matrix = v[:, 0:ncc]

    if do_plot:
        cc_singular_values = cp.asnumpy(cc_singular_values)
        plt.figure()
        plt.plot(cc_singular_values / np.max(cc_singular_values))
        plt.plot(np.cumsum(cc_singular_values)/np.sum(cc_singular_values))
        plt.plot([ncc, ncc], [0, 1.])
        plt.title('Coil compression singular values')
        plt.show()

    return cc_matrix    


def apply_cc_matrix(data_in_c_last_axis, cc_matrix):
    CC = sp.linop.RightMatMul(data_in_c_last_axis.shape, cc_matrix)
    return CC * data_in_c_last_axis


def apply_gcc_matrices(hybrid_data_in_c_last_axis, gcc_matrices_xcc, readout_axis):
    # data should must already be in image domain along readout axis

    xp = sp.get_array_module(hybrid_data_in_c_last_axis)
    nx, nc, ncc = gcc_matrices_xcc.shape

    gcc_matrices_xcc = match_device(gcc_matrices_xcc, hybrid_data_in_c_last_axis)
    with sp.get_device(hybrid_data_in_c_last_axis):
        data_out = xp.zeros(hybrid_data_in_c_last_axis.shape[0:-1] + (ncc,), dtype=np.complex64)
        for xx in range(nx):
            slicer = tuple((slice(xx, xx+1) if dim == readout_axis else slice(None)) for dim in range(hybrid_data_in_c_last_axis.ndim))                
            data_out[slicer] = apply_cc_matrix(hybrid_data_in_c_last_axis[slicer], gcc_matrices_xcc[xx, ...])

        return data_out    