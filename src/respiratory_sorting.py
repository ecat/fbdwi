import warnings

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from utils import uncenter_crop


def sort_respiratory_navigators(im_respnav_sxyc, crop_to_left_liver=True, do_plot=False):
    xp = sp.get_array_module(im_respnav_sxyc)

    ns, nx, ny_respnav, nc = im_respnav_sxyc.shape

    with sp.get_device(im_respnav_sxyc):

        im_respnav_xycs = xp.transpose(im_respnav_sxyc, (1, 2, 3, 0))
        ny_full = 4 * ny_respnav # zeropad in kspace to this resolution
        im_respnav_full_xycs = xp.abs(sp.ifft(uncenter_crop(sp.fft(im_respnav_xycs, axes=(0, 1)), (-1, ny_full, -1, -1)), axes=(0, 1))).astype(np.float32)
        
        # take half liver with most motion
        ny_liver = ny_full // 2 if crop_to_left_liver else ny_full
        im_respnav_liver_xycs = im_respnav_full_xycs[:, 0:ny_liver, :, :]

        def get_respiratory_signal(im_shots_last_axis):
            _, _, vt = xp.linalg.svd(np.reshape(im_shots_last_axis, (-1, im_shots_last_axis.shape[-1])), full_matrices=False)
            return (vt.T)[:, 0]

        dominant_component_all_coils = get_respiratory_signal(im_respnav_liver_xycs)

        dominant_component_per_coil = xp.zeros((ns, nc), dtype=np.float32)

        for coil in range(nc):
            dominant_component_per_coil[:, coil] = get_respiratory_signal(im_respnav_liver_xycs[:, :, coil, :])

        # cluster the coils that represent respiratory phase well https://onlinelibrary.wiley.com/doi/10.1002/mrm.25858
        # extract dominant motion state from each coil and make the coil correlation matrix
        correlation_threshold = 0.95
        coil_correlation_matrix = xp.corrcoef(dominant_component_per_coil, rowvar=False)
        thresholded_coil_correlation = coil_correlation_matrix * (coil_correlation_matrix > correlation_threshold).astype(np.float32)
        thresholded_coil_correlation_u, _, _ = xp.linalg.svd(thresholded_coil_correlation, full_matrices=False)

        good_coil_indices = cp.asnumpy(xp.argwhere(xp.abs(thresholded_coil_correlation_u[:, 0]) > 0.1))

        if len(good_coil_indices) == 0:
            good_coil_indices = np.array([16, 17, 18, 19])
            warnings.warn('coil correlation method failed, using preset coils')

        print("good coil indices " + str(np.squeeze(np.array(good_coil_indices))))

        im_respnav_liver_good_coils_xycs = im_respnav_liver_xycs[:, :, good_coil_indices, :]
        dominant_component_good_coils = get_respiratory_signal(im_respnav_liver_good_coils_xycs)

        sorted_shot_indices = cp.asnumpy(xp.argsort(dominant_component_good_coils))

        im_respnav_liver_sorted_xycs = cp.asnumpy(im_respnav_liver_xycs[:, :, :, sorted_shot_indices])

    if do_plot:
        shots_to_plot = range(0, ns, 5)

        
        plt.figure()
        plt.imshow(np.reshape(np.transpose(np.squeeze(im_respnav_liver_sorted_xycs[:, :, good_coil_indices[0], shots_to_plot]), (0, 2, 1)), (nx, -1)), cmap='gray')
        plt.title('respnav')

        plt.figure()
        plt.subplot(221)
        plt.imshow(cp.asnumpy(coil_correlation_matrix))
        plt.subplot(222)
        plt.imshow(cp.asnumpy(thresholded_coil_correlation))
        plt.subplot(212)
        plt.plot(cp.asnumpy(dominant_component_all_coils), label='all')
        plt.plot(cp.asnumpy(dominant_component_good_coils), label='good')
        plt.legend()
        plt.show()

    return sorted_shot_indices.tolist(), im_respnav_liver_sorted_xycs
