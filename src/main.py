# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import os, sys

is_jupyter = hasattr(sys, 'ps1') # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode

from utils import sos, print_and_clear_cupy_memory, whiten, match_device, montage, center_crop, uncenter_crop, print_and_clear_cupy_fft_cache, calculate_adc_map
from mpflash_recon_functions import gauss_window, triangle_window, pe_to_grid, test_pe_to_grid, reconstruct_phase_navigators, get_diffusion_forward_model
from mpflash_recon_functions import PhaseNavWeightingType, EHEScalingType, EHbScalingType
from shot_rejection import walsh_method, PhaseNavNormalizationType, PhaseNavPerVoxelNormalizationType, normalize_phasenav_weights
from coil_compression import get_gcc_matrices_3d, apply_gcc_matrices, get_cc_matrix, apply_cc_matrix
from respiratory_sorting import sort_respiratory_navigators
from cfl import writecfl, readcfl

# +
import numpy as np
import cupy as cp
import sigpy as sp
import sigpy.plot as pl
import sigpy.mri as mr
import time
import argparse
import scipy
import matplotlib.pyplot as plt
import warnings
import random
import pickle

# %matplotlib widget

# +
data_folder = '../data/data_for_figure_10/'
scan_str = 'scan2_' # see README for descriptions of scans
base_str = data_folder + scan_str

ksp_vsxpec = readcfl(base_str + 'ksp_vsxpec')
ksp_respnav_vsxyc = readcfl(base_str + 'ksp_respnav_vsxyc')
shot_mask_vsyz = readcfl(base_str + 'shot_mask_vsyz')
shot_mask_vspecoord = readcfl(base_str + 'shot_mask_vspecoord').astype(np.int32)
ksp_cal_vsxpec = readcfl(base_str + 'ksp_cal_vsxpec')
ksp_cal_respnav_vsxyc = readcfl(base_str + 'ksp_cal_respnav_vsxyc')
shot_mask_cal_vsyz = readcfl(base_str + 'shot_mask_cal_vsyz')
shot_mask_cal_vspecoord = readcfl(base_str + 'shot_mask_cal_vspecoord').astype(np.int32)
covariance_matrix = readcfl(base_str + 'covariance_matrix')

with open(base_str + 'sequence_opts.pkl', 'rb') as handle:
    sequence_opts = pickle.load(handle)
# -

nv, ns, nx, npe, nc = ksp_vsxpec.shape
_, _, ny, nz = shot_mask_vsyz.shape
ny_respnav = ksp_respnav_vsxyc.shape[-2] if sequence_opts["do_respnav"] > 0 else 0
n_cal_scans = ksp_cal_vsxpec.shape[1]
ncc = 8
nr = 1 if ny_respnav == 0 else 4 # number of respiratory phases
ns_per_r = ns // nr
ns_recon = ns_per_r * nr
resp_phase_slicer = tuple(slice(rr * ns_per_r, (rr + 1) * ns_per_r if rr < nr - 1 else ns) for rr in range(nr))
n_lowres_lines = sequence_opts["phase_nav_lines_start"]
phase_nav_w_ky, phase_nav_w_kz = sequence_opts["phase_nav_w"]
print('nv', nv, 'ns', ns, 'nx', nx, 'npe', npe, 'nc', nc, 'ncc', ncc, 'ny', ny, 'nz', nz, \
    'ny_respnav', ny_respnav, 'ncal', n_cal_scans, 'nr', nr, 'n_lowres_lines', n_lowres_lines, 'phase_nav_w_ky', phase_nav_w_ky, 'phase_nav_w_kz', phase_nav_w_kz,
    'ns_per_r', ns_per_r, 'ns_recon', ns_recon, 'etl_skips_start', sequence_opts["etl_skips_start"])

# +
# the main recon loop uses about 14GB of GPU memory but this could be reduced by loading one volume at a time to gpu
# this is okay since the volumes reconstruct independently
# volume 0 is b=0 data, volume 1 is b=500 data, this entire recon could feasibly fit in a 8-12 GB GPU
gpu_device = sp.Device(0) # can change this to cpu if no gpu available, will be 4-5x slower
xp = gpu_device.xp
print('Run reconstruction with', gpu_device)

ksp_vsxpec = sp.to_device(ksp_vsxpec, gpu_device)
ksp_respnav_vsxyc = sp.to_device(ksp_respnav_vsxyc, sp.cpu_device)
shot_mask_vsyz = sp.to_device(shot_mask_vsyz, gpu_device)
shot_mask_vspecoord = sp.to_device(shot_mask_vspecoord, sp.cpu_device)

ksp_cal_vsxpec = sp.to_device(ksp_cal_vsxpec, sp.cpu_device)
ksp_cal_respnav_vsxyc = sp.to_device(ksp_cal_respnav_vsxyc, sp.cpu_device)
shot_mask_cal_vsyz = sp.to_device(shot_mask_cal_vsyz, sp.cpu_device)
shot_mask_cal_vspecoord = sp.to_device(shot_mask_cal_vspecoord, sp.cpu_device)

shot_occurence_v_yz = xp.sum(shot_mask_vsyz, axis=(1,), keepdims=True)
shot_occurence_v_yz[shot_occurence_v_yz < 1] = 1
dcf_v_yz = 1 / shot_occurence_v_yz

# +
do_whiten = True

if do_whiten:    
    order_R = (4, 0, 1, 2, 3)
    order_RH = np.argsort(order_R)
    R = lambda x: xp.reshape(xp.transpose(x, order_R), (nc, -1))
    RH = lambda x, dims: xp.transpose(xp.reshape(x, tuple(dims[i] for i in order_R)), order_RH)
    whiten_helper = lambda x: RH(whiten(R(x), match_device(covariance_matrix, x)), x.shape)

    ksp_vsxpec = whiten_helper(ksp_vsxpec)
    ksp_cal_vsxpec = whiten_helper(ksp_cal_vsxpec)

# +
# set the FFT cache to have maximum 2 plans since it uses a lot of memory
fft_plan_cache = cp.fft.config.get_plan_cache()
fft_plan_cache.set_size(2)

# separate along readout
im_vsx_ksp_pec = sp.ifft(ksp_vsxpec, axes=(2,))

if ny_respnav > 0:
    # reconstruct respiratory navigators
    im_respnav_vsxyc = sp.ifft(ksp_respnav_vsxyc, axes=(2, 3))
    im_cal_respnav_vsxyc = sp.ifft(ksp_cal_respnav_vsxyc, axes=(2, 3))

# +
do_plot_respiratory_navigators = True and ny_respnav > 0

if is_jupyter and do_plot_respiratory_navigators:
    plt.figure()
    plt.subplot(221)
    plt.imshow(np.squeeze(np.abs(ksp_cal_respnav_vsxyc[0, 0, :, :, 0])), cmap="gray")
    plt.title("Respnav cal k-space")
    plt.subplot(222)
    plt.imshow(np.squeeze(np.abs(im_cal_respnav_vsxyc[0, 0, :, :, 0])), cmap="gray")
    plt.title("Respnav cal image")    

    coils_to_plot = np.arange(0, nc, 1)
    shot_to_plot = 4
    im_respnav_to_show = np.reshape(np.transpose(np.squeeze(im_respnav_vsxyc[0, shot_to_plot, :, :, coils_to_plot]), (1, 0, 2)), (nx, -1))
    plt.subplot(212)
    plt.imshow(np.abs(im_respnav_to_show), cmap="gray", aspect=2)
    plt.title("Respnav images for all coils")
    plt.show()

# +
if ny_respnav > 0:
    
    sorted_shot_indices_v = [[] for vv in range(nv)]

    for vv in range(nv):
        sorted_shot_indices, _ = sort_respiratory_navigators(sp.to_device(im_respnav_vsxyc[vv, :, :, :], gpu_device), True, (vv == (nv - 1)))
        sorted_shot_indices_v[vv] = sorted_shot_indices

        # sort everything including respnavs again in place so that if we run this cell again it stays sorted
        im_vsx_ksp_pec[vv, ...] = im_vsx_ksp_pec[vv, sorted_shot_indices, ...]
        ksp_respnav_vsxyc[vv, ...] = ksp_respnav_vsxyc[vv, sorted_shot_indices, ...]
        shot_mask_vsyz[vv, ...] = shot_mask_vsyz[vv, sorted_shot_indices, ...]
        shot_mask_vspecoord[vv, ...] = shot_mask_vspecoord[vv, sorted_shot_indices, ...]
        im_respnav_vsxyc[vv, ...] = im_respnav_vsxyc[vv, sorted_shot_indices, ...]



# +
# do linear recon, assumes axes are always rsxyzc
P_phasenavless = sp.linop.Sum((ns, ny, nz, nc), axes=(0,)).H
DCF = sp.linop.Multiply((ns, ny, nz, nc), xp.reshape(dcf_v_yz[0, ...], (1, ny, nz, 1)))
D = sp.linop.Multiply((ns, ny, nz, nc), xp.reshape(shot_mask_vsyz[0, ...], (ns, ny, nz, 1)))
F = sp.linop.FFT((ny, nz, nc), axes=(0, 1))

im_naive_vxyzc = np.zeros((nv, nx, ny, nz, nc), dtype=np.complex64)

for vv in range(1):        
    for xx in range(nx):
        data_for_x = im_vsx_ksp_pec[vv, np.newaxis, :, xx, np.newaxis, :, :]
        ksp_syzc = xp.squeeze(pe_to_grid(data_for_x, shot_mask_vspecoord[vv, np.newaxis, :, :, :], (ny, nz)))
        im_yzc = F.H * P_phasenavless.H * DCF * ksp_syzc
        im_naive_vxyzc[vv, xx, :, :, :] = cp.asnumpy(im_yzc)

im_naive_vxyz = sos(im_naive_vxyzc, 4)


do_clear = True

if do_clear:
    del data_for_x, ksp_syzc, im_yzc, P_phasenavless, DCF, D, F
# -

normalizing_factor = np.mean(sos(sos(im_naive_vxyz, 3), 2), axis=1)[0]
im_vsx_ksp_pec = im_vsx_ksp_pec / normalizing_factor

# +
# do coil compression
im_naive_xyzc = sp.to_device(im_naive_vxyzc[0, :, :, :, :], gpu_device)
ksp_naive_xyzc = sp.fft(im_naive_xyzc, axes=(0, 1, 2))

do_gcc = True

ncalibx, ncaliby, ncalibz = nx if do_gcc else 24, 24, min(16, nz)
ksp_center_xyzc = center_crop(ksp_naive_xyzc, (ncalibx, ncaliby, ncalibz, -1))

if do_gcc:
    cc_matrices_xcc = get_gcc_matrices_3d(ksp_center_xyzc, ncc)

    im_cc_vsx_ksp_pec = apply_gcc_matrices(im_vsx_ksp_pec, cc_matrices_xcc, 2)
    im_naive_cc_vxyzc = apply_gcc_matrices(sp.to_device(im_naive_vxyzc, gpu_device), cc_matrices_xcc, 1)

else:
    cc_matrix = get_cc_matrix(ksp_center_xyzc, ncc, do_plot=True)

    im_cc_vsx_ksp_pec = apply_cc_matrix(im_vsx_ksp_pec, cc_matrix)
    im_naive_cc_vxyzc = apply_cc_matrix(im_naive_vxyzc, cc_matrix)
    
im_naive_cc_vxyz = cp.asnumpy(sos(im_naive_cc_vxyzc, 4))

do_clear = False
if do_clear:
    del im_vsx_ksp_pec, im_naive_vxyzc, im_naive_xyzc, ksp_naive_xyzc, ksp_center_xyzc

# +
do_cc_plot = True

if do_cc_plot:
    im_to_show = np.flip(np.concatenate((im_naive_vxyz, im_naive_cc_vxyz), axis=2), axis=1) # flip along axis 1 to flip readout
    plt.figure()
    plt.imshow(cp.asnumpy(im_to_show[0, :, :, 12]), cmap='gray')
    plt.show()
    plt.figure()
    plt.imshow(np.angle(cp.asnumpy(im_naive_cc_vxyzc[0, :, :, 10, 0])), cmap='gray')
    plt.show()

# +
# estimate coil sensitivities
ksp_for_sens_estimation_cxyz = sp.fft(sp.to_device(np.transpose(im_naive_cc_vxyzc[0, :, :, :, :], (3, 0, 1, 2)), gpu_device), axes=(1, 2, 3))
sens_lowres_window = sp.to_device(triangle_window((24, 24, 16), (nx, ny, nz)).reshape(1, nx, ny, nz), gpu_device)

sens_xyzc = xp.transpose(mr.app.EspiritCalib(ksp_for_sens_estimation_cxyz * sens_lowres_window, calib_width=16, device=gpu_device, kernel_width=5, crop=0.97, show_pbar=False).run(), (1, 2, 3, 0))
sens_lowres_xyzc = xp.transpose(sp.ifft(ksp_for_sens_estimation_cxyz * sens_lowres_window, axes=(1, 2, 3)), (1, 2, 3, 0))
del ksp_for_sens_estimation_cxyz
# -

_ = montage(xp.abs(sens_xyzc[:, :, :, 0]), grid_cols=6)
_ = montage(xp.abs(sens_lowres_xyzc[:, :, :, 0]), grid_cols=6)


# +
phase_nav_recon_size = (nx//2, ny//2, nz)

window_triangle_xyz = triangle_window(phase_nav_recon_size, (nx, ny, nz))
window_gauss_xyz = gauss_window((nx, ny, nz), (.5, .25, .33))

phasenav_window_type = 'triangle'

if phasenav_window_type == 'gauss':
    window_xyz = sp.to_device(window_gauss_xyz, gpu_device)
elif phasenav_window_type == 'triangle':
    window_xyz = sp.to_device(window_triangle_xyz, gpu_device)
else:
    assert False, "Invalid window type"


# +
# reconstruct phase navigators
im_phasenavs_vxyzs = xp.zeros((nv, nx, ny, nz, ns), dtype=np.complex64)
im_phasenavs_lowres_vxyzs = xp.zeros((nv,) + phase_nav_recon_size + (ns,), dtype=np.complex64)
im_phasenavs_contrast_nostic_vxyzs = xp.zeros_like(im_phasenavs_vxyzs)
im_phasenavs_contrast_nostic_lowres_vxyzs = xp.zeros_like(im_phasenavs_lowres_vxyzs)

start = time.time()

for vv in range(nv):        
    im_phasenavs_xyzs, im_phasenavs_lowres_xyzs = reconstruct_phase_navigators(im_cc_vsx_ksp_pec[vv, ...], shot_mask_vspecoord[vv, ...], sens_xyzc, window_xyz, \
        phase_nav_recon_size, n_lowres_lines)
    
    im_phasenavs_vxyzs[vv, ...] = im_phasenavs_xyzs
    im_phasenavs_lowres_vxyzs[vv, ...] = im_phasenavs_lowres_xyzs

    im_phasenavs_contrast_nostic_xyzs, im_phasenavs_contrast_nostic_lowres_xyzs = reconstruct_phase_navigators(im_cc_vsx_ksp_pec[vv, ...], shot_mask_vspecoord[vv, ...], sens_lowres_xyzc, window_xyz, \
        phase_nav_recon_size, n_lowres_lines)

    im_phasenavs_contrast_nostic_vxyzs[vv, ...] = im_phasenavs_contrast_nostic_xyzs
    im_phasenavs_contrast_nostic_lowres_vxyzs[vv, ...] = im_phasenavs_contrast_nostic_lowres_xyzs
    
del im_phasenavs_xyzs, im_phasenavs_lowres_xyzs, im_phasenavs_contrast_nostic_xyzs, im_phasenavs_contrast_nostic_lowres_xyzs
print("Reconstruct phase navigator took %.2f seconds" % (time.time() - start))
# -

if is_jupyter:
    vol_to_plot = min(nv - 1, 1)
    shots_to_plot = [0, 1, 2, 3]
    slices_to_plot = slice(4, 20)

    for shot_to_plot in shots_to_plot:
        for im_phasenavs_to_plot in [im_phasenavs_vxyzs]:
            _ = montage(xp.abs(im_phasenavs_to_plot[vol_to_plot, :, :, slices_to_plot, shot_to_plot]), grid_cols=6, normalize=True)
            plt.clim([0, 0.8])

# +
# compute spatial shot-to-shot weighting
# use second gpu since memory for this step is largeish and too lazy to clear memory on first gpu
walsh_device = sp.Device(1) if True else sp.cpu_device # option to execute walsh on CPU or a second GPU, CPU a bit slower but doesn't have memory isuses
window_shape = (4, 4, 4)
window_stride = tuple(w//2 for w in window_shape)
gauss_window_stds = (1 / window_shape[0], 1 / window_shape[1], 1.)
phasenav_weighting_gauss_window_xyz = sp.to_device(gauss_window((nx, ny, nz), gauss_window_stds)[..., np.newaxis], walsh_device)

im_phasenav_weightings_vxyzs = xp.zeros((nv, nx, ny, nz, ns), dtype=np.complex64)
phasenav_weighting_alg = 'eig'

# this true is necessary for the deep breathing since
# different parts of the object may move into a window, for shallow
# breathing this can probably be False and weights can be computed jointly across all resp phases
do_weights_per_respiratory_phase = True 

# if True, uses espirit coil sensitivities which keeps the t2 weighting in the phase navigators
# using False will use low res T2-weighted images (not normalized to sos=1) so that factors
# out some of the underlying contrast. However it's not too good for exag breathing
do_contrast_agnostic_phasenav_weighting = True 
walsh_shot_slicer = resp_phase_slicer if do_weights_per_respiratory_phase else (slice(0, ns), )

start = time.time()
for vv in range(nv):

    if do_contrast_agnostic_phasenav_weighting:
        im_phasenav_for_weighting_calc_vxyzs = im_phasenavs_lowres_vxyzs
    else:
        im_phasenav_for_weighting_calc_vxyzs = im_phasenavs_contrast_nostic_lowres_vxyzs

    for slicer in walsh_shot_slicer:
        im_phasenav_lowres_weightings_xyzs = sp.to_device(xp.abs(im_phasenav_for_weighting_calc_vxyzs[vv, :, :, :, slicer]), walsh_device)

        if phasenav_weighting_alg == 'lowres':
            im_phasenav_lowres_weightings_xyzs = im_phasenav_lowres_weightings_xyzs / xp.max(im_phasenav_lowres_weightings_xyzs)
        else:
            im_phasenav_lowres_weightings_xyzs = walsh_method(im_phasenav_lowres_weightings_xyzs, window_shape, window_stride, phasenav_weighting_alg)

        # perform the upsample fft on walsh_device because it is rather large, then copy to gpu with other recon variables (gpu0)
        with walsh_device:
            im_phasenav_upsampled_lowres_weightings_xyzs = xp.abs(sp.ifft(phasenav_weighting_gauss_window_xyz * \
                uncenter_crop(sp.fft(sp.to_device(im_phasenav_lowres_weightings_xyzs, walsh_device), axes=(0, 1, 2)), (nx, ny, nz, -1)), axes=(0, 1, 2)))

        im_phasenav_weightings_vxyzs[vv, :, :, :, slicer] = sp.to_device(im_phasenav_upsampled_lowres_weightings_xyzs, gpu_device)

print("Phase nav weights per respiratory phase %d took %.2f seconds" % (do_weights_per_respiratory_phase, time.time() - start))

# +
if do_weights_per_respiratory_phase:
    phasenav_weighting_per_voxel_normalization = PhaseNavPerVoxelNormalizationType.PERCENTILE_PER_RESPIRATORY_PHASE
else:
    phasenav_weighting_per_voxel_normalization = PhaseNavPerVoxelNormalizationType.MEAN_ACROSS_MAX_RESP_PHASE

phasenav_weighting_normalization = PhaseNavNormalizationType.NOOP

im_phasenav_weightings_vxyzs = normalize_phasenav_weights(im_phasenav_weightings_vxyzs, phasenav_weighting_normalization, phasenav_weighting_per_voxel_normalization,
                                                          nv, nx, ny, nz, nr, ns_per_r, resp_phase_slicer)
# -

if is_jupyter:
    slices_to_plot = slice(4, 20)
    vol_to_plot = min(nv - 1, 1)
    shots_to_plot = [0, -2, -1]
    for shot_to_plot in shots_to_plot:
        _ = montage(np.abs(im_phasenav_weightings_vxyzs[vol_to_plot, :, :, slices_to_plot, shot_to_plot]), grid_cols=6, normalize=True)
        plt.clim([0, 1])

# +
# set what kind of recon to do 
# for clinic we only care about vol two recons which has recon time ~8-10 minutes
# technically the recon loop which is the largest part only uses
# 1 gpu, so could parallelize volumes across multiple gpus
# to reduce recon time
is_clinic = False

if is_clinic:
        vol_out_to_vol_in_table = [0, 1]
        phase_nav_weighting_types_v = [PhaseNavWeightingType.NONE, PhaseNavWeightingType.SENSE_LIKE]
else:
    do_wlsq_comparison = False # use this if you want to see contrast changes, effect is most obvious in the figure 8 data at posterior

    if do_wlsq_comparison:
        vol_out_to_vol_in_table = [0, 1, 1, 1, 1] # repeating 1 will do the dw reconstruction with different forward models
        phase_nav_weighting_types_v = [PhaseNavWeightingType.NONE, PhaseNavWeightingType.NONE, 
                                        PhaseNavWeightingType.SENSE_LIKE, PhaseNavWeightingType.WLSQ, PhaseNavWeightingType.WLSQ_DC]
    else:
        vol_out_to_vol_in_table = [0, 1, 1]
        phase_nav_weighting_types_v = [PhaseNavWeightingType.NONE, PhaseNavWeightingType.NONE, 
                                    PhaseNavWeightingType.SENSE_LIKE]    


do_phasenav_for_b0 = True
do_phasenav_v = [v == 1 or do_phasenav_for_b0 for v in vol_out_to_vol_in_table]   

nv_recon = len(vol_out_to_vol_in_table)
assert len(phase_nav_weighting_types_v) == nv_recon

# +
# do the recon
start = time.time()
EHE_scaling_type = EHEScalingType.TRACE_EHE
EHb_scaling_type = EHbScalingType.BY_FIRST_VOLUME 

print(str(EHE_scaling_type) + " " + str(EHb_scaling_type))


im_adjoint_vrxyz = xp.zeros((nv_recon, nr, nx, ny, nz), dtype=np.complex64)
max_eigenvalues_EHE_vx = xp.zeros((nv_recon, nx), dtype=np.float32)
max_eigenvalues_EHE_normalized_vx = xp.zeros((nv_recon, nx), dtype=np.float32)
oneH_EHE_one_vx = xp.zeros((nv_recon, nx), dtype=np.float32)
oneH_EEH_one_vx = xp.zeros((nv_recon, nx), dtype=np.float32)
trace_EHE_vx = xp.zeros((nv_recon, nx), dtype=np.float32)

im_cg_vrxyz = xp.zeros_like(im_adjoint_vrxyz)
cg_iters = 0 # cg always looks bad unless nr = 1 

im_fista_vrxyz = xp.zeros_like(im_adjoint_vrxyz)
fista_iters = 100
do_save_fista_obj = is_jupyter and False
fista_obj = np.zeros((nv_recon, nx, fista_iters), dtype=np.float32)
lambda_spatial_wav = 1e-3 if nr > 1 else 2e-3
lambda_resp_wav = 1e-3 * ((nr > 1) and (nr % 2 == 0))
do_3d_wavelet = True
do_use_contrast_agnostic_phasenavs = False

num_regularizers = xp.array((lambda_spatial_wav > 0) + (lambda_resp_wav > 0), dtype=np.complex64)

linear_operators_vx = [[0 for x in range(nx)] for v in range(nv_recon)]
linear_operators_normalized_vx = [[0 for x in range(nx)] for v in range(nv_recon)]

# build the linear operators
for vv in range(nv_recon):
    # some minor loop modifications to make comparing methods easy
    vol_in = vol_out_to_vol_in_table[vv]
    do_phasenav_weighting = phase_nav_weighting_types_v[vv]
    do_phasenav = do_phasenav_v[vv]

    im_phasenavs__xyzs = im_phasenavs_vxyzs[vol_in, np.newaxis, :, :, :, 0:ns_recon] if do_use_contrast_agnostic_phasenavs else im_phasenavs_contrast_nostic_vxyzs[vol_in, np.newaxis, :, :, :, 0:ns_recon]
    im_phasenav_weightings__xyzs = im_phasenav_weightings_vxyzs[vol_in, np.newaxis, :, :, :, 0:ns_recon]
    mask_rsyz_ = xp.reshape(shot_mask_vsyz[vol_in, 0:ns_recon, ...], (nr, ns_per_r, ny, nz, 1))
    
    print('vol out ' + str(vv) + ' vol in ' + str(vol_in) + ' phase nav fwd model ' + str(do_phasenav_weighting))

    linear_operators_x, trace_EHE_x, max_eigenvalues_EHE_x = \
        get_diffusion_forward_model(mask_rsyz_, sens_xyzc, im_phasenavs__xyzs, im_phasenav_weightings__xyzs, do_phasenav, do_phasenav_weighting, power_iters=10)

    trace_EHE_vx[vv, :] = trace_EHE_x
    max_eigenvalues_EHE_vx[vv, :] = max_eigenvalues_EHE_x
    linear_operators_vx[vv] = linear_operators_x


max_eigenvalues_EHE_smoothed_vx = sp.to_device(scipy.signal.medfilt(cp.asnumpy(max_eigenvalues_EHE_vx), (1, 5)), gpu_device)
trace_EHE_smoothed_vx = sp.to_device(scipy.signal.medfilt(cp.asnumpy(trace_EHE_vx), (1, 5)), gpu_device)

if EHE_scaling_type == EHEScalingType.NONE:
    EHE_scaling_vx = xp.ones_like(max_eigenvalues_EHE_smoothed_vx)
elif EHE_scaling_type == EHEScalingType.MAX_EIG_EHE:
    EHE_scaling_vx = max_eigenvalues_EHE_smoothed_vx    
elif EHE_scaling_type == EHEScalingType.TRACE_EHE:
    EHE_scaling_vx = trace_EHE_smoothed_vx

if do_3d_wavelet:
    EHE_scaling_vx = xp.repeat(xp.max(EHE_scaling_vx, axis=(1,), keepdims=True), nx, axis=1)

max_eigenvalues_EHE_normalized_vx = max_eigenvalues_EHE_smoothed_vx / EHE_scaling_vx

# since the EHE differs between recons, need to renormalize EHE to get same relative regularization strength and have same bias
for vv in range(nv_recon):
    vol_in = vol_out_to_vol_in_table[vv]
    print('vol out ' + str(vv) + ' vol in ' + str(vol_in))
    
    for xx in range(nx):
        E, EHE, W_sqrt_shot = linear_operators_vx[vv][xx]

        EHE_scaling = EHE_scaling_vx[vv, xx]

        E_normalized = sp.linop.Compose((sp.linop.Multiply(E.oshape, 1.0 / xp.sqrt(EHE_scaling)), E))
        EHE_normalized = sp.linop.Compose((sp.linop.Multiply(EHE.oshape, 1.0 / EHE_scaling), EHE))

        coords_for_slice = shot_mask_vspecoord[vol_in, np.newaxis, 0:ns_recon, ...]
        data_for_slice__s_pec = im_cc_vsx_ksp_pec[vol_in, np.newaxis, 0:ns_recon, xx, np.newaxis, :, :]
        ksp_for_slice_rsyzc = xp.reshape(pe_to_grid(data_for_slice__s_pec, coords_for_slice, (ny, nz)), (nr, ns_per_r, ny, nz, ncc))
        im_for_slice_ryz = E_normalized.H * W_sqrt_shot * ksp_for_slice_rsyzc

        oneH_EHE_one_vx[vv][xx] = xp.sum(xp.abs(E_normalized * xp.ones_like(im_for_slice_ryz))**2) /xp.sqrt(ny * nz * nr)
        #oneH_EEH_one_vx[vv][xx] = xp.sum(xp.abs(E_normalized.H * xp.ones_like(ksp_for_slice_rsyzc))**2) /xp.sqrt(ny * nz * nr)
        
        im_adjoint_vrxyz[vv, :, xx, :, :] = xp.reshape(im_for_slice_ryz, (nr, ny, nz))        

        linear_operators_normalized_vx[vv][xx] = (E_normalized, EHE_normalized)
        

if EHb_scaling_type == EHbScalingType.BY_FIRST_VOLUME:
    # same normalizing factor for all volumes calculated from b=0
    EHb_renormalizing_factor_v_x = xp.mean(sos(im_adjoint_vrxyz[0, np.newaxis, ...], (-2, -1), keepdims=True), axis=(1, 2), keepdims=True) 
    print("EHb renormalizing factor " + str(EHb_renormalizing_factor_v_x))
elif EHb_scaling_type == EHbScalingType.PER_VOLUME:
    EHb_renormalizing_factor_v_x = xp.mean(sos(im_adjoint_vrxyz, (-2, -1), keepdims=True), axis=(1, 2), keepdims=True)
    print("EHb renormalizing factor " + str(EHb_renormalizing_factor_v_x))
elif EHb_scaling_type == EHbScalingType.PER_READOUT_SLICE:
    EHb_renormalizing_factor_v_x = xp.mean(sos(im_adjoint_vrxyz, (-2, -1), keepdims=True), axis=(1), keepdims=True)

EHb_renormalizing_factor_v_x[EHb_renormalizing_factor_v_x < 1e-6] = 1.0
im_adjoint_vrxyz = im_adjoint_vrxyz / EHb_renormalizing_factor_v_x

# do the recon
for vv in range(nv_recon):

    im_adjoint_rxyz_ = xp.reshape(im_adjoint_vrxyz[vv, ...], (nr, nx, ny, nz, 1))  # add dummy dimension for coils
    
    if EHE_scaling_type == EHEScalingType.TRACE_EHE:
        fista_step_size = 1 / xp.max(xp.ravel(max_eigenvalues_EHE_normalized_vx))
    else:
        fista_step_size = 0.8 / xp.max(max_eigenvalues_EHE_normalized_vx[vv, :])
    print("vol " + str(vv) + " step size " + str(fista_step_size))

    EHb_shape_2d = (nr, 1, ny, nz, 1) # dummy at end for coils
    EHb_shape_3d = (nr, nx, ny, nz, 1)
    EHb_shape = EHb_shape_3d if do_3d_wavelet else EHb_shape_2d

    haar_filter = xp.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex64) / xp.sqrt(2.0)
    
    if lambda_resp_wav > 0:
        resp_to_last = sp.linop.Transpose(EHb_shape, (4, 1, 2, 3, 0))
        to_block = sp.linop.ArrayToBlocks(resp_to_last.oshape, (1, 1, 2), (1, 1, 2))
        haar_matmul = sp.linop.RightMatMul(to_block.oshape, haar_filter)
        W_respphase = sp.linop.Compose((haar_matmul, to_block, resp_to_last))            
        prox_resp = sp.prox.UnitaryTransform(sp.prox.L1Reg(W_respphase.oshape, lambda_resp_wav), W_respphase)
    else:
        prox_resp = lambda alpha, x: 0

    wavelet_axes = (1, 2, 3) if do_3d_wavelet else (2, 3)
    W_spatial = sp.linop.Wavelet(EHb_shape, wave_name="db4", axes=wavelet_axes, level=1)
    prox_spatial = sp.prox.UnitaryTransform(sp.prox.L1Reg(W_spatial.oshape, lambda_spatial_wav), W_spatial) if lambda_spatial_wav > 0 else lambda alpha, x: 0

    def proxg_pfista(alpha, x):
        r_shift = random.randint(-nr//2, nr//2 + 1) if nr > 1 else 0
        x_shift = random.randint(-2, 3) if do_3d_wavelet else 0
        y_shift = random.randint(-4, 5)
        z_shift = random.randint(-2, 3)
        C = sp.linop.Circshift(EHb_shape, (r_shift, x_shift, y_shift, z_shift, 0))
        Cx = C * x                
        return C.H * (prox_spatial(alpha, Cx) + prox_resp(alpha, Cx)) / num_regularizers    


    if cg_iters > 0:
        for xx in range(nx):
            E, EHE = linear_operators_normalized_vx[vv][xx]

            im_for_slice_ryz = xp.reshape(im_adjoint_rxyz_[:, xx, :, :], EHb_shape) 
    
        cg_alg = sp.alg.ConjugateGradient(EHE, im_for_slice_ryz, xp.zeros_like(im_for_slice_ryz), max_iter=cg_iters)
        while not cg_alg.done():
            cg_alg.update()
        im_cg_vrxyz[vv, :, xx, :, :] = xp.reshape(cg_alg.x, (nr, ny, nz))

    
    if fista_iters > 0:

        def g(x):
            g_x = 0
            if lambda_spatial_wav > 0:
                g_x = g_x + lambda_spatial_wav * xp.sum(xp.ravel(xp.abs(W_spatial * x)))
            if lambda_resp_wav > 0:
                g_x = g_x + lambda_resp_wav * xp.sum(xp.ravel(xp.abs(W_respphase * x)))
            return g_x

        if do_3d_wavelet:  

            def gradf(x):
                linear_operators_normalized_x = linear_operators_normalized_vx[vv]        
                EHE_list_for_all_x = [linear_operators_normalized_x[xx][1] for xx in range(0, nx)] 
                EHE_stacked = sp.linop.Diag(EHE_list_for_all_x, oaxis=1, iaxis=1) # this creates a linop that loops over readout for us, sigpy has everything and is amazing...    
                EHEx_rxyz_ = EHE_stacked * x

                return EHEx_rxyz_ - im_adjoint_rxyz_

            fista_alg = sp.alg.GradientMethod(gradf, xp.zeros_like(im_adjoint_rxyz_), alpha=fista_step_size, proxg=proxg_pfista, max_iter=fista_iters, accelerate=True)
            while not fista_alg.done():
                fista_alg.update()

                if do_save_fista_obj:
                    obj = 0
                    # compute error in this roundabout way to account for scaling of E, otherwise need some more divides on data
                    for xx in range(nx):
                        E, EHE = linear_operators_normalized_vx[vv][xx]
                        x_r_yz_ = fista_alg.x[:, xx, np.newaxis, ...]
                        ra = xp.dot(xp.ravel(xp.conj(x_r_yz_)), xp.ravel(EHE(x_r_yz_)))
                        rb = -2 * xp.dot(xp.ravel(xp.conj(x_r_yz_)), xp.ravel(im_adjoint_rxyz_[:, xx, ...]))
                        
                        obj = obj + ra + rb # missing a constant inner product of y for now, but ignore it since not function of x
                        
                    obj = obj + g(fista_alg.x)
                    fista_obj[vv, xx, fista_alg.iter - 1] = cp.asnumpy(xp.abs(obj)).astype(np.float32)


            im_fista_vrxyz[vv, ...] = xp.reshape(fista_alg.x, (nr, nx, ny, nz))                       
        else:
            for xx in range(nx):
                E, EHE = linear_operators_normalized_vx[vv][xx]

                im_for_slice_ryz = xp.reshape(im_adjoint_rxyz_[:, xx, :, :], EHb_shape) 
                gradf = lambda x: EHE * x - im_for_slice_ryz

                fista_alg = sp.alg.GradientMethod(gradf, xp.zeros_like(im_for_slice_ryz), alpha=fista_step_size, proxg=proxg_pfista, max_iter=fista_iters, accelerate=True)
                while not fista_alg.done():
                    fista_alg.update()
                    
                im_fista_vrxyz[vv, :, xx, :, :] = xp.reshape(fista_alg.x, (nr, ny, nz)) 

# for adjoint don't need to account for EHE scaling
im_adjoint_vrxyz = sp.to_device(im_adjoint_vrxyz * EHb_renormalizing_factor_v_x)

# need to also undo scaling on EHE
# say we solve ||y - Ex||^2, and A = cE where c is the scaling on E to normalize EHE then
# then the answer x is divided by c, so to recover original x we need to multiply by c
im_cg_vrxyz = sp.to_device(im_cg_vrxyz * EHb_renormalizing_factor_v_x / xp.reshape(xp.sqrt(EHE_scaling_vx), (nv_recon, 1, nx, 1, 1)))
im_fista_vrxyz = sp.to_device(im_fista_vrxyz * EHb_renormalizing_factor_v_x / xp.reshape(xp.sqrt(EHE_scaling_vx), (nv_recon, 1, nx, 1, 1)))

print("Reconstruction took %.2f seconds" % (time.time() - start))    

# +
im_fista_normalized_vrxyz = im_fista_vrxyz / np.mean(np.abs(im_fista_vrxyz), axis=(1, 2, 3, 4), keepdims=True) # normalize answer so plots are nicer
descr_string = ' pnw ' + str(do_phasenav_weighting) + ' win ' + phasenav_window_type

slice_to_iterable = lambda s: range(s.start, s.stop, 1 if s.step is None else s.step)

def show_adjoint(vol_to_plot, slices_to_plot, resp_to_plot=0):
    show_volume(im_adjoint_vrxyz, vol_to_plot, slices_to_plot, resp_to_plot)
    plt.title('adjoint' + descr_string)

def show_cg(vol_to_plot, slices_to_plot, resp_to_plot=0):
    if cg_iters > 0:
        montage(np.abs(im_cg_vrxyz[vol_to_plot, resp_to_plot, :, :, slices_to_plot]), normalize=True)
        plt.title('cg')

def show_fista(vol_to_plot, slices_to_plot, resp_to_plot=0):
    if fista_iters > 0:
        show_volume(im_fista_normalized_vrxyz, vol_to_plot, slices_to_plot, resp_to_plot)
        plt.title('fista')

def show_volume(im_vrxyz, vol_to_plot, slices_to_plot, resp_to_plot):
    assert (type(slices_to_plot) is slice) ^ (type(resp_to_plot) is slice)

    is_plot_along_z = type(slices_to_plot) is slice

    im_vrxyz = np.flip(im_vrxyz, axis=2)    
    im_to_plot = np.abs(im_vrxyz[vol_to_plot, resp_to_plot, :, :, slices_to_plot])
    
    if type(vol_to_plot) is slice:
        im_to_plot = np.vstack(tuple(im_to_plot[d, ...] for d in slice_to_iterable(vol_to_plot)))

    if is_plot_along_z:
        montage(im_to_plot, normalize=True)
    else:
        montage(im_to_plot.transpose(1, 2, 0), normalize=True)


# -

vol_to_plot = min(nv - 1, 1)
slices_to_plot = slice(3, 22, 1)

if is_jupyter:
    nv_to_plot = nv_recon
    for zz in slice_to_iterable(slices_to_plot):
        show_fista(slice(0, nv_to_plot), zz, slice(0, nr))
        plt.clim([-.02, .5])
        plt.title('fista slice %d' % zz)


# +
if is_jupyter:    
    plt.figure()
    plt.subplot(131)
    plt.plot(np.sum(sos(im_adjoint_vrxyz, (-2, -1)), 1).T)
    plt.title('Energy per slice adjoint')
    plt.legend(['vol '  + str(vv) for vv in range(nv_recon)])
    plt.subplot(132)
    plt.plot(np.sum(sos(im_fista_vrxyz, (-2, -1)), 1).T)
    plt.title('Energy per slice FISTA')
    plt.subplot(133)
    resp_phase_to_plot = 0
    energy_first_resp_phase_vx = np.squeeze(sos(im_fista_vrxyz[:, resp_phase_to_plot, ...], (-2, -1)))
    plt.plot(energy_first_resp_phase_vx[1, :] / energy_first_resp_phase_vx[0, :])
    plt.plot(energy_first_resp_phase_vx[2, :] / energy_first_resp_phase_vx[0, :])
    plt.plot(energy_first_resp_phase_vx[1, :] / energy_first_resp_phase_vx[2, :])
    plt.title('ratio energies')
    plt.legend(['no pn weight', 'pn weight', 'no pn weight / pn weight'])
    plt.show()

    plt.figure()
    for vv in range(nv_recon):
        plt.subplot(131)
        plt.plot(sp.to_device(max_eigenvalues_EHE_vx[vv, :]), label=('vol ' + str(vv)))
    plt.legend()
    plt.title('max eig EHE')        
    for vv in range(nv_recon):
        plt.subplot(132)
        plt.plot(sp.to_device(max_eigenvalues_EHE_smoothed_vx[vv, :]), label=('vol smooth ' + str(vv)))
    plt.title('max eig EHE smoothed')      
    for vv in range(nv_recon):
        plt.subplot(133)
        plt.plot(sp.to_device(max_eigenvalues_EHE_normalized_vx[vv, :]), label=('vol smooth ' + str(vv)))
    plt.title('max eig EHE normalized')      
    
    plt.xlabel('x location')
    plt.show()

    plt.figure()
    for vv in range(nv_recon):
        plt.plot(sp.to_device(oneH_EHE_one_vx[vv, :]), label=('vol ' + str(vv)))
    plt.legend()
    plt.title('1HEHE1')

    plt.figure()
    for vv in range(nv_recon):
        plt.subplot(121)
        plt.plot(sp.to_device(trace_EHE_vx[vv, :]), label=('vol ' + str(vv)))
    plt.legend()
    plt.title('trace EHE')        
    for vv in range(nv_recon):
        plt.subplot(122)
        plt.plot(sp.to_device(trace_EHE_smoothed_vx[vv, :]), label=('vol smooth ' + str(vv)))
    
    plt.xlabel('x location')
    plt.show()

    if EHb_scaling_type == EHbScalingType.PER_READOUT_SLICE:
        plt.figure()
        for vv in range(nv_recon):
            plt.plot(sp.to_device(xp.squeeze(EHb_renormalizing_factor_v_x)[vv, :]), label=('vol ' + str(vv)))
        plt.legend()
        plt.title('EHb per readout slice')


    
# -

if is_jupyter and do_save_fista_obj:
    plt.figure()
    for vv in range(0, nv_recon):
        plt.subplot(nv_recon, 1, vv + 1)
        plt.plot(fista_obj[vv, :, :].T)
    plt.show()

if sequence_opts['all_in_one']:
    # only compute adc for first resp phase
    adc_map_xyz = calculate_adc_map(im_fista_vrxyz[0, 0, ...], im_fista_vrxyz[1, 0, ...], sequence_opts['bvalue'][-1])
    _ = montage(np.flip(adc_map_xyz[:, :, 4:12], axis=0))
    plt.title('adc map')
    plt.clim([0, 3e-3])

