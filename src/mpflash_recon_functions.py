import ctypes
import enum
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.mri as mr

from utils import center_crop, match_device, uncenter_crop


class PhaseNavWeightingType(enum.Enum):
    NONE = 0
    WLSQ = 1
    WLSQ_DC = 2
    WLSQ_W_SQUARED = 3
    SENSE_LIKE = 4

class EHEScalingType(enum.Enum):
    NONE = 0
    MAX_EIG_EHE = 1
    TRACE_EHE = 2

class EHbScalingType(enum.Enum):
    NONE = 0
    BY_FIRST_VOLUME = 1
    PER_VOLUME = 2
    PER_READOUT_SLICE = 3

def pe_to_grid(data_vsxpec, shot_mask_vspecoord, pe_grid_size):
    assert(data_vsxpec.ndim == 5)
    assert(shot_mask_vspecoord.shape[-1] == 2)
    assert(data_vsxpec.shape[0] == shot_mask_vspecoord.shape[0] and data_vsxpec.shape[1] == shot_mask_vspecoord.shape[1])
    
    # do the looping on the same device as shot_mask_vspecoord since it is faster but return it on the same device as the input
    # doing this on cpu seems a bit faster but to make it non blocking on gpu it's maybe better to do it on gpu
    input_device = sp.backend.get_device(data_vsxpec)
    nv, ns, nx, npe, nc = data_vsxpec.shape
    ny = pe_grid_size[0]
    nz = pe_grid_size[1]
    xp = sp.backend.get_array_module(data_vsxpec)

    # in numpy, when using arange indexing it gets pushed to the front so need to reshape the data so that the variables being set are continunous
    # with npe at front
    data_in = xp.transpose(data_vsxpec, (0, 1, 3, 2, 4))
    data_out = xp.zeros((nv, ns, ny * nz, nx, nc), dtype=np.complex64) 
    shot_mask_vspecoord = match_device(shot_mask_vspecoord, data_in)

    for vv in range(0, nv):
        for shot in range(0, ns):
            ky_indices = xp.squeeze(shot_mask_vspecoord[vv, shot, :, 0])
            kz_indices = xp.squeeze(shot_mask_vspecoord[vv, shot, :, 1])
            linear_pe_indices = xp.ravel_multi_index((ky_indices, kz_indices), (ny, nz))
            data_out[vv, shot, linear_pe_indices, :, :] = data_in[vv, shot, :, :, :]
    
    # copy to contiguous array only if not copying to gpu
    f = lambda x: np.ascontiguousarray(x) if input_device == sp.Device(-1) else sp.to_device(x, input_device)

    return f(xp.transpose(xp.reshape(data_out, (nv, ns, ny, nz, nx, nc)), (0, 1, 4, 2, 3, 5)))

def test_pe_to_grid(shot_mask_vspecoord, shot_mask_vsyz):
    nv, ns, npe, _ = shot_mask_vspecoord.shape
    _, _, ny, nz = shot_mask_vsyz.shape
    test_mask_vsxyz_ = pe_to_grid(np.ones_like(shot_mask_vsyz, shape=(nv, ns, 1, npe, 1)), shot_mask_vspecoord, (ny, nz))
    difference = np.abs(np.sum(np.ravel(test_mask_vsxyz_) - np.ravel(shot_mask_vsyz)))
    if(difference != 0):
        warnings.warn("pe to grid does not match input mask")

    shots_to_plot = range(6)    
    plt.figure()
    for idx, shot_to_plot in enumerate(shots_to_plot):
        plt.subplot(1, len(shots_to_plot), idx + 1)
        plt.imshow(np.squeeze(np.abs(test_mask_vsxyz_[0, shot_to_plot, 0, :, :, 0] + sp.to_device(shot_mask_vsyz[0, shot_to_plot, np.newaxis, :, :]))))
    plt.title("Masks generated by pe_to_grid")
    plt.show()

def triangle_window(inner_matrix_size, full_matrix_size=None):    
    # returns a ramp triangle window within inner_matrix_size, zeros everywhere else    
    if full_matrix_size is None:
        full_matrix_size = inner_matrix_size
    assert(all(tuple((inner_matrix_size[dim] <= full_matrix_size[dim],) for dim in range(len(inner_matrix_size)))))
    coords = np.meshgrid(*tuple(((np.linspace(-1, 1, inner_matrix_size[sz]),) for sz in range(len(inner_matrix_size)))), indexing='ij')

    triangle_window = np.ones(inner_matrix_size, dtype=np.float32)
    for dim in range(len(coords)):    
        triangle_window = triangle_window * np.abs(1 - np.abs(coords[dim]))
    
    return uncenter_crop(triangle_window, full_matrix_size)

def gauss_window(inner_matrix_size, inner_matrix_sigma, full_matrix_size=None):
    # inner_matrix_sigma is standard deviation across the inner matrix shape 
    if full_matrix_size is None:
        full_matrix_size = inner_matrix_size    
    coords = np.meshgrid(*tuple(((np.linspace(-1, 1, inner_matrix_size[sz]),) for sz in range(len(inner_matrix_size)))), indexing='ij')

    gauss_window = np.ones(inner_matrix_size, dtype=np.float32)    

    for dim, sigma in enumerate(inner_matrix_sigma):
        gauss_window = gauss_window * np.exp(-np.square(coords[dim]) / (2 * (sigma ** 2)))

    return uncenter_crop(gauss_window, full_matrix_size)


def reconstruct_phase_navigators(im_sx_ksp_pec, shot_mask_specoord, sens_xyzc, window_xyz, phase_nav_recon_size, n_lowres_lines):
    # phase nav recon size is a tuple of length 3
    # n_lowres_lines tells how much of each echo train to take 
    device = sp.get_device(im_sx_ksp_pec)
    xp = sp.get_array_module(im_sx_ksp_pec)
    nx, ny, nz, nc = sens_xyzc.shape
    ns = im_sx_ksp_pec.shape[0]

    sens_cxyz = xp.transpose(sens_xyzc, ((3, 0, 1, 2)))
    sens_downsampled_cxyz = sp.ifft(center_crop(sp.fft(sens_cxyz, axes=(1, 2, 3)), (-1,) + phase_nav_recon_size), axes=(1, 2, 3))

    with device:
        im_phasenavs_lowres_xyzs = xp.zeros(phase_nav_recon_size + (ns,), dtype=np.complex64)        
        im_phasenavs_xyzs = xp.zeros((nx, ny, nz, ns), dtype=np.complex64)

        for shot in range(ns):

            lowres_data_for_shot = sp.fft(im_sx_ksp_pec[np.newaxis, shot, np.newaxis, :, 0:n_lowres_lines, :], axes=(2,)) # input is in hybid space, take it to all ksp
            lowres_coords_for_shot = shot_mask_specoord[np.newaxis, shot, np.newaxis, 0:n_lowres_lines, :]
            
            ksp_lowres_xyzc = xp.squeeze(pe_to_grid(lowres_data_for_shot, lowres_coords_for_shot, (ny, nz)))       
            ksp_lowres_cxyz = xp.transpose(ksp_lowres_xyzc, (3, 0, 1, 2))
            ksp_lowres_cxyz_cropped = center_crop(ksp_lowres_cxyz, (-1,) + phase_nav_recon_size)

            lowres_mask_for_shot_yz = xp.squeeze(pe_to_grid(xp.ones((1, 1, 1, n_lowres_lines, 1), dtype=np.complex64), lowres_coords_for_shot, (ny, nz)))        
            lowres_mask_for_shot_yz_cropped = center_crop(lowres_mask_for_shot_yz, ksp_lowres_cxyz_cropped.shape[-2::])[np.newaxis, np.newaxis, :, :]        
            n_phasenav_cg_iters = 5
            app = mr.app.SenseRecon(ksp_lowres_cxyz_cropped, sens_downsampled_cxyz, weights=lowres_mask_for_shot_yz_cropped, device=device, show_pbar=(shot == None), max_iter=n_phasenav_cg_iters)
            im_phasenav_xyz_lowres = app.run()
            im_phasenavs_lowres_xyzs[..., shot] = im_phasenav_xyz_lowres
            im_phasenavs_xyzs[..., shot] = sp.ifft(window_xyz * uncenter_crop(sp.fft(im_phasenav_xyz_lowres, axes=(0, 1, 2)), (nx, ny, nz)), axes=(0, 1, 2))

    return im_phasenavs_xyzs, im_phasenavs_lowres_xyzs

def get_diffusion_forward_model(mask_rsyz_, sens_xyzc, im_phasenavs__xyzs, im_phasenav_weightings__xyzs, do_phasenav, do_phasenav_weighting, power_iters=10):
    nx, ny, nz, nc = sens_xyzc.shape
    nr = mask_rsyz_.shape[0]
    ns_per_r = mask_rsyz_.shape[1]
    xp = sp.get_array_module(sens_xyzc)

    trace_EHE_x = xp.zeros((nx,), dtype=np.float32)
    linear_operators_x = [0 for x in range(nx)]
    max_eigenvalues_EHE_x = xp.zeros((nx,), dtype=np.float32)

    for xx in range(nx):
        
        sens_yzc = xp.reshape(sens_xyzc[xx, :, :, :], (1, 1, ny, nz, nc))        
        
        if do_phasenav_weighting is not PhaseNavWeightingType.NONE:
            im_phasenav_weighting_rsyz_ = xp.reshape(xp.transpose(im_phasenav_weightings__xyzs[:, xx, np.newaxis, ...], (0, 4, 2, 3, 1)), (nr, ns_per_r, ny, nz, 1))
        else:
            im_phasenav_weighting_rsyz_ = 1.0

        if do_phasenav:
            im_phasenav_angle_rsyz_ = xp.exp(1j * xp.angle(xp.reshape(xp.transpose(im_phasenavs__xyzs[:, xx, np.newaxis, ...], (0, 4, 2, 3, 1)), (nr, ns_per_r, ny, nz, 1))))
        else:
            im_phasenav_angle_rsyz_ = xp.ones((nr, ns_per_r, ny, nz, 1), dtype=np.complex64) # allocate full size to preserve shape
               
        S = sp.linop.Multiply((nr, ns_per_r, ny, nz, 1), sens_yzc)
        F = sp.linop.FFT(S.oshape, axes=(-3, -2))
        D = sp.linop.Multiply(F.oshape, mask_rsyz_)

        W_shot = sp.linop.Identity(D.oshape)
        W_sqrt_shot = sp.linop.Identity(D.oshape)
       
        if do_phasenav_weighting == PhaseNavWeightingType.NONE:
            P = sp.linop.Multiply((nr, 1, ny, nz, 1), im_phasenav_angle_rsyz_)
        elif do_phasenav_weighting in  [PhaseNavWeightingType.WLSQ, PhaseNavWeightingType.WLSQ_W_SQUARED, PhaseNavWeightingType.WLSQ_DC]:
            P = sp.linop.Multiply((nr, 1, ny, nz, 1), im_phasenav_angle_rsyz_)

            if do_phasenav_weighting == PhaseNavWeightingType.WLSQ:
                im_weighting_rsyz_ = im_phasenav_weighting_rsyz_
            elif do_phasenav_weighting == PhaseNavWeightingType.WLSQ_DC:
                im_weighting_rsyz_ = xp.mean(im_phasenav_weighting_rsyz_, axis=(2, 3), keepdims=True)
            elif do_phasenav_weighting == PhaseNavWeightingType.WLSQ_W_SQUARED:
                im_weighting_rsyz_ = xp.square(im_phasenav_weighting_rsyz_)

            W_shot = sp.linop.Compose((F, sp.linop.Multiply(F.oshape, im_weighting_rsyz_), F.H))
            W_sqrt_shot = sp.linop.Compose((F, sp.linop.Multiply(F.oshape, xp.sqrt(im_weighting_rsyz_)), F.H))
        elif do_phasenav_weighting == PhaseNavWeightingType.SENSE_LIKE:
            P = sp.linop.Multiply((nr, 1, ny, nz, 1), im_phasenav_weighting_rsyz_ * im_phasenav_angle_rsyz_)
        else:
            assert False, "Invalid do_phasenav_weighting"

        E = sp.linop.Compose((W_sqrt_shot, D, F, S, P))
        EHE = sp.linop.Compose((P.H, S.H, F.H, D.H, W_shot, D, F, S, P))

        if power_iters > 0:
            max_eigenvalue = sp.app.MaxEig(EHE, device=sens_xyzc.device, max_iter=power_iters, dtype=np.complex64, show_pbar=False).run()
            max_eigenvalues_EHE_x[xx] = 1 if np.isnan(max_eigenvalue) else max_eigenvalue
        
        if do_phasenav_weighting in [PhaseNavWeightingType.WLSQ, PhaseNavWeightingType.WLSQ_W_SQUARED, PhaseNavWeightingType.WLSQ_DC]:
            # slight modification to other Trace, where we get a nice expression if we assume the spatial weights are low resolution (so their fourier transform
            # is band limited), it's proportional to the DC value per readout slice of the weights
            psf_dc_rs = xp.abs(xp.sum(mask_rsyz_, axis=(2, 3), keepdims=True)) * nc * xp.mean(xp.abs(im_weighting_rsyz_), axis=(2, 3), keepdims=True)
        else:
            # implement equation from Evan's paper, can be derived by writing block matrices
            # Equation 11 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5840375/
            psf_dc_rs = xp.abs(xp.sum(mask_rsyz_, axis=(2, 3), keepdims=True)) * nc #  dc value of psf for each shot, abs to make real and eliminate cast warning, multiply by ncc since mask does not include c dimension


        trace_EHE_x[xx] = xp.sum((xp.abs(S.mult * P.mult) ** 2) * psf_dc_rs) / (ny * nz * ns_per_r * nr * nc) # divide by this so that we have orthonormal defintion of fft
        trace_EHE_x[xx] = 1.0 if trace_EHE_x[xx] < 1e-5 else trace_EHE_x[xx] # guard against small values when coil sens is zero in a readout slice

        linear_operators_x[xx] = (E, EHE, W_sqrt_shot)

    return linear_operators_x, trace_EHE_x, max_eigenvalues_EHE_x