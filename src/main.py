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
from mpflash_recon_functions import gauss_window, triangle_window, pe_to_grid, load_mp_flash_scanarchive, test_pe_to_grid, walsh_method, reconstruct_phase_navigators, get_diffusion_forward_model
from mpflash_recon_functions import PhaseNavWeightingType, PhaseNavNormalizationType, EHEScalingType, EHbScalingType, PhaseNavPerVoxelNormalizationType
from coil_compression import get_gcc_matrices_3d, apply_gcc_matrices, get_cc_matrix, apply_cc_matrix
from respiratory_sorting import sort_respiratory_navigators
from cfl import writecfl, readcfl
