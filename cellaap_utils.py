import numpy.typing as npt
from skimage.filters import gaussian
import numpy as np
import scipy.ndimage as ndi
from scipy.signal import medfilt
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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
    # med_filtered_mean_plane = ndi.median_filter(mean_plane, 9)
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
        mean_signal = np.nan

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


def fit_model(xy_data: pd.DataFrame, plot: True, quant_fraction = None, bin_size = None) -> (pd.DataFrame, dict): # type: ignore
    '''
    Function to fit the dose-response data with a 4-parameter sigmoid.
    Bin range is determined by quantiles. Default is 0.025 and 0.85. The data
    typically contain outliers on the high side, but not the low side. Hence the 
    default values are aysmmetric.
    Inputs:
    xy_data        - dataframe w/ dose as the first column and response as 
                     the second column
    plot           - Boolean to enable plotting
    quant_fraction - quantiles to determine bin range; 
    bin_size       - size of each bin, default is 2.5 (empirical)
    Outputs:
    xy_data        - the input dataframe with bin labels added as a new column
    fit_pars       - dictionary containing fit parameters
    '''
    if quant_fraction is None:
        quant_fraction = [0.025, 0.85]
    quants = np.round(xy_data.iloc[:,0].quantile(quant_fraction)).tolist()

    # 
    if bin_size is None:
        bin_size = 2.5
    bins   = np.arange(quants[0], quants[-1], bin_size)

    labels, _ = pd.cut(xy_data.iloc[:, 0], bins, retbins=True)
    xy_data["bins"] = labels

    bin_means = xy_data.groupby("labels").mean()
    # bin_stds  = xy_data.groupby("labels").std()
    # bin_n     = xy_data.groupby("labels").count()

    fits, _ = curve_fit(sigmoid_4par, bin_means.iloc[:,0], bin_means.iloc[:,-1], 
                        p0 = [bin_means.iloc[:,1].min(), bin_means.iloc[:,1].max(), 
                              5, (quants[0] + quants[-1])/ 4
                             ]
                       )

    if plot:
        plt.plot(bin_means.iloc[:,0], bin_means.iloc[:,1], 'ro')
        plt.plot(xy_data.iloc[:,0], xy_data.iloc[:,1], 'r.')
        plt.plot(bin_means.iloc[:,0], sigmoid_4par(bin_means.iloc[:,0],
                                       fits[0], fits[1], fits[2], fits[3]), 'b-')
    
    fit_values = { 'min_duration' : fits[0],
                   'max_duration' : fits[1],
                   'Hill_exponent': fits[2],
                   'EC50'         : fits[3]
                 }
    
    return xy_data, fit_values

def sigmoid_4par(x, base, top, exponent, ec50):

    return base + (top - base)*(x**exponent)/(x**exponent+ec50**exponent)