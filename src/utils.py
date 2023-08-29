import os
import sys

import numpy as np
import sigpy as sp
import cupy as cp
import warnings
import matplotlib.pyplot as plt

def filter_files(dirpath, filter):
    all_files = [os.path.join(basedir, filename) for basedir, dirs, files in os.walk(dirpath) for filename in files if (filter in filename) ]
    return all_files

def uncenter_crop(x, full_matrix_size):
    assert(len(full_matrix_size) == x.ndim)    
    assert np.all(np.logical_or(np.array(full_matrix_size) % 2 == 0, np.array(full_matrix_size) == -1)), "Does not support odd matrix sizes, use -1 for full"
    inner_matrix_size = x.shape
    output_matrix_size = [x.shape[dim] if full_matrix_size[dim] == -1 else full_matrix_size[dim] for dim in range(x.ndim)]
    xp = sp.get_array_module(x)
    with sp.get_device(x):        
        y_full = xp.zeros(output_matrix_size, dtype=x.dtype)
        slicer = (slice(full_matrix_size[dim]//2 - inner_matrix_size[dim]//2, full_matrix_size[dim]//2 + inner_matrix_size[dim]//2) if full_matrix_size[dim] > 0 else slice(None) for dim in range(0, x.ndim))
        y_full[tuple(slicer)] = x
    return y_full

def get_center_mask(full_matrix_size, inner_matrix_size, dtype=np.complex64):
    # returns a mask with center is ones
    assert(len(full_matrix_size) == len(inner_matrix_size))
    return uncenter_crop(np.ones(inner_matrix_size, dtype=dtype), full_matrix_size)

def center_crop(x, crop_size):
    # center crops the array x to a dimension specified by crop size, if crop_size has a -1 it will take the whole axis    
    assert(x.ndim == len(crop_size))
    assert(all(tuple(crop_size[dim] <= x.shape[dim] for dim in range(0, x.ndim))))    
    slicer = tuple(slice((x.shape[dim] - crop_size[dim])//2 if crop_size[dim] > 0 else None, (x.shape[dim] + crop_size[dim])//2 if crop_size[dim] > 0 else None) for dim in range(0, x.ndim))
    return x[slicer]

def montage(im_xyz, grid_cols=4, normalize=False, title=None, do_show=False):
    assert(im_xyz.ndim == 3), "input must have three dimensions"
    im_xyz = sp.to_device(im_xyz)
    nx, ny, nz = im_xyz.shape  
    if (nz % grid_cols != 0):
        warnings.warn("Number of requested grid columns does not evenly divide nz, some slices will be absent")

    slicer = tuple((slice(None), slice(None), slice(row * grid_cols, min((row+1) * grid_cols, nz))) for row in range(0, nz//grid_cols))    
    im_to_show = np.vstack(tuple(np.reshape(np.transpose(im_xyz[s], (0, 2, 1)), (nx, -1)) for s in slicer))
    scale = np.max(np.abs(np.ravel(im_to_show))) if normalize else 1.
    
    plt.figure()
    plt.imshow(im_to_show / scale, cmap='gray', aspect=1)

    if title is not None:
        plt.title(title)

    if do_show:
        plt.show()

    return im_to_show

def match_device(a, b):
    # copies a to the same device as b
    return sp.to_device(a, sp.backend.get_device(b))

def print_and_clear_cupy_memory(do_clear=True):

    mempool = cp.get_default_memory_pool()
    print("Before clear " + str(mempool.used_bytes()))

    if do_clear:
        mempool.free_all_blocks()
        print("After clear " + str(mempool.used_bytes()))

def print_and_clear_cupy_fft_cache(do_print=True, do_clear=False):

    cache = cp.fft.config.get_plan_cache()
    if do_print:
        cache.show_info()

    if do_clear:    
        cache.clear()
        print("Cleared FFT Cache")

def whiten(x, cov):
    """
    Applies a Cholesky whitening transform to the data.
    
    Parameters
    ----------
    x : numpy/cupy array
        Input array whose first dimension corresponds to the number of coils
    cov : numpy/cupy array
        Covariance matrix of the noise present in x
        
    Returns
    -------
    y : numpy/cupy array
        A whitened version of x with the same dimensions
    """
    
    # Check that the whitening transform can be applied
    if x.shape[-2] != cov.shape[0]:
        raise ValueError("The first dimension of x and the provided covariance matrix do not match.")
    
    device = sp.get_device(x)
    xp = device.xp
    cov = sp.to_device(cov, device)
        
    # Get the whitening transform operator
    L = xp.linalg.cholesky(xp.linalg.inv(cov))
    LH = L.conj().T
    
    # Apply the whitening transform and return the result
    y = LH @ x
    return y

def sos(matrix_in, axis_in, keepdims=False):
    device = sp.get_device(matrix_in)
    xp = device.xp
    with device:
        matrix_out = xp.sqrt(xp.sum(np.abs(matrix_in) ** 2, axis=axis_in, keepdims=keepdims))
    return matrix_out

def calculate_adc_map(im_xyz_b0, im_xyz_dw, diff_bval):
    
    with np.errstate(divide='ignore', invalid='ignore'):
        im_divide_xyz = np.true_divide(np.abs(im_xyz_dw), np.abs(im_xyz_b0))
        im_log_xyz = np.log(im_divide_xyz)
        im_log_xyz[im_log_xyz == np.inf] = 0
        im_log_xyz = np.nan_to_num(im_log_xyz)
        
    adc_map = np.abs(-1/diff_bval * im_log_xyz)
    adc_map[adc_map > 5e-3] = 0
    
        
    return np.abs(adc_map)    

def get_us_linear_transform(x):
    """
    Gets the slope and intercept of the linear transform to convert the data contained in the numpy array
    to unsigned short (uint16) values.
    """
    ii16 = np.iinfo(np.int16)
    y = np.array([x.min(), x.max()])
    A = np.array([[ii16.min + 1, 1], [ii16.max - 1, 1]]) # don't go exactly to min/max to avoid rounding errors
    params = np.linalg.solve(A, y)
    slope = params[0]
    intercept = params[1]
    
    return slope, intercept

def convert_to_int16_full_range(unscaled_im):
    assert(unscaled_im.dtype == np.float32)
    slope, intercept = get_us_linear_transform(unscaled_im)
    return ((unscaled_im - intercept) / slope), slope, intercept

def get_git_hash():
    git_exit_code = os.system("git rev-parse --verify HEAD >> /dev/null")
    git_rev = None
    if git_exit_code == 0:
        git_rev = os.popen("git rev-parse --verify HEAD").read().strip()

    return git_rev, git_exit_code

def add_git_hash_to_dicom_header(ds):
    # ds is object generated from dcmread
    tag = get_git_hash()
    if type(tag) is str:
        ds.add_new([0x0018, 0x9315], "CS", tag[0:16].upper()) # only append 16 since that is the number that is supported

    return ds

