import sys, os
from pathlib import Path
import numpy.typing as npt
from skimage.io import imread # type: ignore
from skimage.morphology import erosion, disk
from skimage.measure import regionprops_table, block_reduce
from skimage.transform import rescale
from skimage.filters import gaussian
import napari # type: ignore
import numpy as np
import pandas as pd
import trackpy as tp
import scipy.ndimage as ndi
from scipy.signal import medfilt


def projection(im_array: np.ndarray, projection_type: str):

    if im_array.shape[0] % 2 == 0:
        center_index = im_array.shape[0] // 2 - 1
    else:
        center_index = im_array.shape[0] // 2

    range = center_index // 2

    try:
        assert projection_type in ["max", "min", "average"]
    except AssertionError:
        print("Projection type was not valid, valid types include: max, min, mean")

    if projection_type == "max":
        projected_image = np.max(
            im_array[center_index - range : center_index + range], axis=0
        )
    elif projection_type == "average":
        projected_image = np.mean(
            im_array[center_index - range : center_index + range], axis=0
        )
    elif projection_type == "min":
        projected_image = np.min(
            im_array[center_index - range : center_index + range], axis=0
        )

    return np.array(projected_image)
    
def gen_intensity_correction_map(image: npt.NDArray) -> npt.NDArray:
    """
    From Anish
    Computes the intensity map for flouresence microscopy intensity normalization if the input is a blank with flourescent media
    ----------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        image: npt.NDArray
    OUTPUTPS:
        intensity_map: npt.NDArray
    """
    mean_plane = projection(image, "average")
    smoothed_mean_plane = gaussian(mean_plane, 45)
    intensity_correction_map = smoothed_mean_plane / (np.max(smoothed_mean_plane))

    return intensity_correction_map

def gen_background_correction_map(background_stack: npt.NDArray) -> npt.NDArray:
    '''
    Newly written to avoid too much smoothing. The cMOS camera has a persistent noise pattern
    therefore, it is better to keep the corrections local. 
    '''

    background_correction_map = np.zeros_like(background_stack, dtype=int)
    footprint = footprint=np.ones((3,3))
    for i in np.arange( background_stack.shape[0]):
        background_correction_map[i,:,:] = ndi.median_filter(background_stack[i,:,:], footprint=footprint)

    return background_correction_map


def mean_signal_from_mask(img: npt.NDArray, mask: npt.NDArray):
    '''
    '''
    pixels = img[np.nonzero(mask)]
    if pixels.any():
        mean_signal = np.mean(pixels)
    else:
        mean_signal = np.NAN

    return mean_signal



def calculate_signal(semantic, signal, bkg_corr, int_corr):
    '''
    utility function for calculating signal from the given semantic, signal, and bkg traces
    '''
    # I also noticed that the signal goes up during metaphase. 
    # THerefore, multiply the signal trace with the semantic label.
    # in semantic, 100 = mitotic, 1 = non-mitotic
    # semantic = (semantic - 1)/99
    semantic = medfilt(semantic, 3) # Need to add to the class
    semantic = (semantic - 1)/99

    if signal.any():
        signal = np.mean(signal[np.where(semantic)])
    else:
        signal = 0
    
    if bkg_corr.any():
        bkg_corr = np.mean(bkg_corr[np.where(semantic)])
    else:
        bkg_corr = 0

    if int_corr.any():
        int_corr = np.mean(int_corr[np.where(semantic)])
    else:
        int_corr = 1

    return signal, bkg_corr, int_corr