import cupy as cp
import sigpy as sp
import numpy as np
import enum


class PhaseNavPerVoxelNormalizationType(enum.Enum):
    NONE = 0
    MEAN_ACROSS_ALL_RESP_PHASES = 1 # makes the mean across all shots equal to one
    MEAN_PER_RESPIRATORY_PHASE = 2 # makes mean across each respiratory phase one    
    MEAN_ACROSS_FIRST_RESP_PHASE = 3 # makes mean across first resp phase equal to one (less sensitive to outliers, no weird contrast changes)
    MEAN_ACROSS_MAX_RESP_PHASE = 4 # finds the mean for each phase and selects the resp phase with max to equal 1
    MAX_PER_RESPIRATORY_PHASE = 5 # slightly sensitive to outliers 
    PERCENTILE_PER_RESPIRATORY_PHASE = 6

class PhaseNavNormalizationType(enum.Enum):
    NOOP = 0
    SQUARE = 1
    SQRT = 2
    EXP = 3


def normalize_phasenav_weights(im_phasenav_weightings_vxyzs, phasenav_weighting_normalization, phasenav_weighting_per_voxel_normalization,
                               nv, nx, ny, nz, nr, ns_per_r, resp_phase_slicer):

    xp = sp.get_array_module(im_phasenav_weightings_vxyzs)

    per_voxel_mean = lambda x: x / xp.mean(x, axis=-1, keepdims=True)
    per_voxel_max = lambda x: x / xp.max(x, axis=-1, keepdims=True)
    per_voxel_sos = lambda x: x / xp.sum(xp.abs(x)**2, axis=-1, keepdims=True)
    per_yz_slice_mean = lambda x: x / xp.mean(x, axis=(-3, -2, -1), keepdims=True)
    per_global_mean = lambda x: x / xp.mean(x, axis=(-4, -3, -2, -1), keepdims=True)

    print("Phase nav weighting normalization %s per voxel %s" % (phasenav_weighting_normalization, phasenav_weighting_per_voxel_normalization))

    if phasenav_weighting_normalization != PhaseNavNormalizationType.NOOP:
        # prenormalizing makes weighting functions easier to tune
        im_phasenav_weightings_vxyzs = per_voxel_mean(xp.abs(im_phasenav_weightings_vxyzs))

    if phasenav_weighting_normalization == PhaseNavNormalizationType.NOOP:
        pass
    elif phasenav_weighting_normalization == PhaseNavNormalizationType.SQUARE:
        im_phasenav_weightings_vxyzs = xp.square(im_phasenav_weightings_vxyzs)
    elif phasenav_weighting_normalization == PhaseNavNormalizationType.SQRT:
        im_phasenav_weightings_vxyzs = xp.sqrt(im_phasenav_weightings_vxyzs)
    elif phasenav_weighting_normalization == PhaseNavNormalizationType.EXP:
        im_phasenav_weightings_std_vxyz_ = xp.std(im_phasenav_weightings_vxyzs, axis=-1, keepdims=True)
        im_phasenav_weightings_std_vxyz_ = im_phasenav_weightings_std_vxyz_ / xp.mean(xp.ravel(im_phasenav_weightings_std_vxyz_))    
        im_phasenav_weightings_vxyzs = xp.power(im_phasenav_weightings_vxyzs, im_phasenav_weightings_std_vxyz_)
    else:
        assert False, "Invalid normalization type"

    if phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.MEAN_ACROSS_ALL_RESP_PHASES:
        im_phasenav_weightings_vxyzs = per_voxel_mean(xp.abs(im_phasenav_weightings_vxyzs))
    elif phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.MEAN_PER_RESPIRATORY_PHASE or \
        phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.MAX_PER_RESPIRATORY_PHASE or \
        phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.PERCENTILE_PER_RESPIRATORY_PHASE:
        im_phasenav_weightings_vxyzrs = xp.abs(xp.reshape(im_phasenav_weightings_vxyzs[..., 0:(nr* ns_per_r)], (nv, nx, ny, nz, nr, ns_per_r)))

        if phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.MEAN_PER_RESPIRATORY_PHASE:
            scaling_per_vxyzr_ = xp.mean(im_phasenav_weightings_vxyzrs, axis=-1, keepdims=True)
        elif phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.MAX_PER_RESPIRATORY_PHASE:
            scaling_per_vxyzr_ = xp.max(im_phasenav_weightings_vxyzrs, axis=-1, keepdims=True)
        elif phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.PERCENTILE_PER_RESPIRATORY_PHASE:
            percentiles_per_r = xp.array([60, 60, 40, 20], dtype=np.float32) / 100 # 100 is use all weights for calculating mean
            
            samples_to_normalize_against_per_r = xp.floor(percentiles_per_r * ns_per_r)

            sorted_phasenav_weightings_vxyzrs = xp.sort(im_phasenav_weightings_vxyzrs, axis=-1) # sorts ascending
            sorted_phasenav_weightings_vxyzrs = xp.flip(sorted_phasenav_weightings_vxyzrs, axis=-1) # make descending

            scaling_per_vxyzr_ = xp.ones((nv, nx, ny, nz, nr, 1))
            for rr, slicer in enumerate(resp_phase_slicer):
                scaling_per_vxyzr_[..., rr, :] = xp.mean(sorted_phasenav_weightings_vxyzrs[..., rr, 0:samples_to_normalize_against_per_r[rr]], axis=-1, keepdims=True)

        scaling_per_vxyzs = xp.ones_like(im_phasenav_weightings_vxyzs)

        for rr, slicer in enumerate(resp_phase_slicer):
            scaling_per_vxyzs[..., slicer] = scaling_per_vxyzr_[..., rr, :]

        im_phasenav_weightings_vxyzs = im_phasenav_weightings_vxyzs / scaling_per_vxyzs

    elif phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.MEAN_ACROSS_FIRST_RESP_PHASE:
        scaling = xp.mean(xp.abs(im_phasenav_weightings_vxyzs[..., 0:ns_per_r]), axis=-1, keepdims=True)
        im_phasenav_weightings_vxyzs = im_phasenav_weightings_vxyzs / scaling 
    elif phasenav_weighting_per_voxel_normalization == PhaseNavPerVoxelNormalizationType.MEAN_ACROSS_MAX_RESP_PHASE:
        im_phasenav_weightings_vxyzrs = xp.abs(xp.reshape(im_phasenav_weightings_vxyzs[..., 0:(nr* ns_per_r)], (nv, nx, ny, nz, nr, ns_per_r)))
        scaling_per_vxyzr = xp.mean(im_phasenav_weightings_vxyzrs, axis=-1)
        scaling_per_vxyz_ = xp.max(scaling_per_vxyzr, axis=-1, keepdims=True)
        im_phasenav_weightings_vxyzs = im_phasenav_weightings_vxyzs / scaling_per_vxyz_
    else:
        assert False, "Invalid normalization type"

    return im_phasenav_weightings_vxyzs


def walsh_method(im_xyzc, window_shape, window_stride, alg_str='eig'):
    xp = sp.get_array_module(im_xyzc)
    with sp.get_device(im_xyzc):
        im_cxyz = xp.transpose(im_xyzc, (3, 0, 1, 2))
        nchannels = im_cxyz.shape[0]
        B = sp.linop.ArrayToBlocks(im_cxyz.shape, window_shape, window_stride)
        R1 = sp.linop.Reshape((nchannels, np.prod(B.oshape[1:4]), np.prod(B.oshape[4::])), B.oshape)

        blocks_cbbbwww = B * im_cxyz
        spatial_channel_matrices_cbr = R1 * blocks_cbbbwww  # b = block r = position

        spatial_channel_matrices_bcr = xp.transpose(spatial_channel_matrices_cbr, (1, 0, 2))

        # this is faster than using svd because can assume hermitian symmetry, smaller block size
        spatial_channel_matrices_hermitian_bcc = xp.matmul(spatial_channel_matrices_bcr, xp.conj(xp.transpose(spatial_channel_matrices_bcr, (0, 2, 1)))) 

        if alg_str == 'eig':
            spatial_channel_eigenvalues, spatial_channel_eigenvectors = xp.linalg.eigh(spatial_channel_matrices_hermitian_bcc)
            spatial_channel_largest_vector_bc = spatial_channel_eigenvectors[:, :, -1]
        elif alg_str == 'power':
            nblocks = spatial_channel_matrices_hermitian_bcc.shape[0]
            M = sp.linop.MatMul((nblocks, nchannels, 1), spatial_channel_matrices_hermitian_bcc)
            # change norm_func so that eigenvector is computed per block instead of across the whole linop
            power_alg = sp.alg.PowerMethod(M, xp.ones((nblocks, nchannels, 1), dtype=np.complex64), max_iter=20, norm_func=lambda x: sos(x, (1, 2), keepdims=True))
            while not power_alg.done():
                power_alg.update()

            spatial_channel_largest_vector_bc = xp.squeeze(power_alg.x)

        spatial_channel_largest_vector_bsr = xp.tile(xp.abs(spatial_channel_largest_vector_bc[..., np.newaxis]), (1, 1, spatial_channel_matrices_bcr.shape[-1]))
        spatial_channel_largest_vector_sbr = xp.transpose(spatial_channel_largest_vector_bsr, (1, 0, 2))
        blocks_largest_only_cbbbwww = R1.H * spatial_channel_largest_vector_sbr
        im_weightings_cxyz = B.H * blocks_largest_only_cbbbwww
        im_weightings_xyzc = xp.transpose(im_weightings_cxyz, (1, 2, 3, 0))    

        return xp.ascontiguousarray(im_weightings_xyzc)    