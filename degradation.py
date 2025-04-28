import os
from pathlib import Path
from itertools import groupby
from operator import itemgetter

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.optimize import fmin
from scipy.signal import find_peaks
from scipy.ndimage import zoom

from typing import Tuple, List, Optional

from statsmodels.nonparametric.kernel_regression import KernelReg

import skimage
from skimage.morphology import (
    binary_closing,
    white_tophat,
    binary_dilation,
    binary_erosion,
    closing,
    disk,
)
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
from skimage.restoration import unsupervised_wiener
import findiff


def gkern(l=5, sig=1.0):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def retrieve_traces(
    analysis_df: pd.DataFrame,
    wl: str,
    frame_interval: int,
    clean: Optional[bool] = True,
) -> list[npt.NDArray]:
    """
    Retrieves traces for a given wavelength from the "analysis.xlsx" cellapp-analysis output spreadsheet
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        analysis_df: pd.DataFrame, analysis.xlsx output from https://github.com/ajitpj/cellapp-analysis
        wl: str
        clean_spurious: bool
    OUTPUTS:
        traces: list[npt.NDArray]
    """

    traces = []
    ids = []
    for id in analysis_df["particle"].unique():
        trace = analysis_df.query(f"particle=={id}")[[f"{wl}", "semantic_smoothed"]]
        bkg = analysis_df.query(f"particle=={id}")[f"{wl}_bkg_corr"]
        intensity = analysis_df.query(f"particle=={id}")[f"{wl}_int_corr"]

        trace = trace.to_numpy()

        if clean:
            t_char = 30 // frame_interval
            padded_semantic = np.append(trace[:, 1], np.zeros(3))
            _, props = find_peaks(padded_semantic, plateau_size=1)
            peak_widths = [
                width
                for i, width in enumerate(props["plateau_sizes"])
                if width >= t_char
            ]
            if len(peak_widths) == 1 and (np.sum(props["plateau_sizes"]) - peak_widths[0]) < t_char:
                if padded_semantic[0] != 1:
                    trace[:, 0] = (trace[:, 0] - bkg) * intensity
                    traces.append(trace)
                    ids.append(id)
        else:
            traces.append(trace)
            ids.append(id)

    return traces, ids


def curvature(trace: npt.NDArray, spacing: Optional[int] = 1) -> npt.NDArray:
    """
    Computes the curvature of a discrete valued function
    -----------------------------------------------------
    INPUTS:
        trace: npt.NDArray
        spacing: int
    """

    d2_dx2 = findiff.FinDiff(0, spacing, 2, acc=6)
    d3_dx3 = findiff.FinDiff(0, spacing, 3, acc=6)

    return d2_dx2(trace) / (1 + d3_dx3(trace)) ** (3 / 2)


def degradation_interval(
    trace: npt.NDArray,
    prominence: float,
    spacing: Optional[int] = 1,
) -> tuple[int]:
    """
    Computes the interval over which a signal is diminishing. The function expects certain things to be true:
    1. There is a singular global maxima
    2. There is a qualitative "stop" to degradation (degradation finishes at some point during the trace)
    ---------------------------------------------------------------------------------------------------------

    INPUTS:
        trace: npt.ndarray,
        prominence: float,
        spacing: int
    OUTPUTS:
        start: int, index of the global maxima
        end: int, index of the most prominent maxima of curvature that occurs after the global maxima
    """

    start = fmin(-trace, trace.shape[0] // 2)
    k = curvature(trace, spacing)
    curvature_maxima, _ = find_peaks(k[start:], prominence=prominence)
    iter = 1
    while (
        len(curvature_maxima) != 1 and iter <= 50
    ):  # continue trying to find peaks until 1 is found
        iter += 1
        curvature_maxima, _ = find_peaks(k[start:], prominence=prominence / iter)

    if len(curvature_maxima) == 1:
        end = curvature_maxima[0]
        return start, end
    else:
        return None, None



def qual_deg(traces: npt.NDArray, frame_interval: int) -> tuple[npt.NDArray]:
    """
    Returns the signal from some arbitrary fluorophore begining after a cell-aap mitosis call
    ------------------------------------------------------------------------------------------
    INPUTS:
        traces: npt.NDarray, output of retrieve_traces()
        frame_interval: int, time between successive frames
    OUTPUTS:
        intensity_container: npt.NDArray
        semantic_container: npt.NDArray
    """

    intensity_traces = []
    semantic_traces = []
    first_tp = []
    t_char = 30 // frame_interval
    for trace in traces:
        padded_semantic = np.append(trace[:, 1], np.zeros(3))
        peaks, props = find_peaks(padded_semantic, plateau_size=t_char)
        if props["plateau_sizes"][0] % 2:
            first_mitosis = peaks[0] - (props["plateau_sizes"][0] // 2)
        else:
            first_mitosis = peaks[0] - (props["plateau_sizes"][0] // 2 - 1)
        intensity_trace = trace[:, 0]
        semantic_trace = trace[:, 1]
        intensity_traces.append(intensity_trace)
        semantic_traces.append(semantic_trace)
        first_tp.append(first_mitosis)

    return intensity_traces, semantic_traces, first_tp


def adjust_zoom_factor(chromatin_shape: tuple[int, int, int], instance_shape: tuple[int, int, int]) -> float:
    """
    Adjust zoom factor based on input shapes.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        chromatin_shape: tuple
        instance_shape: tuple
    OUTPUTS:
        zoom_factor: float
    """
    try:
        assert chromatin_shape[1] / instance_shape[1] == chromatin_shape[2] / instance_shape[2]
        return chromatin_shape[2] / instance_shape[2]
    except AssertionError:
        raise ValueError("Chromatin and Instance must be square arrays")


def prepare_cell_image(chromatin: np.ndarray, frame: int, bbox_coords: tuple[int, int, int, int]) -> np.ndarray:
    """
    Prepares the cell image by cropping and applying top-hat filtering and gaussian smoothing.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        chromatin: np.ndarray
        frame: int
        bbox_coords: tuple (rmin, rmax, cmin, cmax)
    OUTPUTS:
        cell: np.ndarray
        nobkg_cell: np.ndarray
    """
    rmin, rmax, cmin, cmax = bbox_coords
    cell = chromatin[frame, rmin:rmax, cmin:cmax]
    nobkg_cell = skimage.morphology.white_tophat(cell, disk(5))
    return cell, gaussian(nobkg_cell, sigma=1.5)


def get_largest_signal_regions(nobkg_cell, cell: np.ndarray, num_regions: int = 1) -> tuple[list[regionprops], np.ndarray]:
    """
    Segments the cell and returns the brightest regions.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
        num_regions: int
    OUTPUTS:
        sorted_regions: List[regionprops]
        labeled: np.ndarray
    """
    thresh = threshold_otsu(nobkg_cell) #
    labeled, num_labels = label(nobkg_cell > thresh, return_num=True, connectivity=1)
    labels = np.linspace(1, num_labels, num_labels).astype(int)
    region_intensities = [np.nansum(cell[labeled == (lbl)]) for lbl in labels]
    max_intensity = max(region_intensities)
    max_intensity_lbl = labels[region_intensities.index(max_intensity)]
    
    if len(region_intensities) > 1:
        second_max_intensity = sorted(region_intensities)[-2]
        nxt_max_intensity_lbl = labels[
                    region_intensities.index(second_max_intensity)
                ]
        
        intensity_diff_ratio = (max_intensity - second_max_intensity) / max_intensity
        
    else:
        nxt_max_intensity_lbl = None
        intensity_diff_ratio = 1        
        
    return labeled, max_intensity_lbl, nxt_max_intensity_lbl, intensity_diff_ratio


def remove_regions(labels: list[int], labeled: np.ndarray) -> np.ndarray:
    """
    Removes specified labeled regions by dilating their masks.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        regions: List[regionprops]
        labeled: np.ndarray
    OUTPUTS:
        removal_mask: np.ndarray
    """
    removal_mask = np.zeros_like(labeled, dtype=bool)
    for lbl in labels:
        removal_mask[labeled == lbl] = 1
    return binary_dilation(removal_mask, disk(9))


def remove_metaphase_if_eccentric(lbl:int, labeled: np.ndarray) -> np.ndarray:
    """
    Removes the metaphase plate only if its eccentricity exceeds threshold.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        region: regionprops
        labeled: np.ndarray
    OUTPUTS:
        removal_mask: np.ndarray
    """
    region_mask = np.zeros_like(labeled, dtype=bool)
    region_mask[labeled == lbl] = 1
    eccentricity = regionprops(label(region_mask.astype(int)))[0].eccentricity
    if eccentricity > 0.7:
        print('metaphase; removing plate')
        return binary_dilation(region_mask, disk(9))
    else:
        print('metaphase; NOT removing plate')
        return np.zeros_like(labeled, dtype=bool)


def segment_unaligned_chromosomes(cell: np.ndarray, removal_mask: np.ndarray, min_area: int) -> tuple[int, int, int]:
    """
    Segments and measures properties of unaligned chromosomes.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
        removal_mask: np.ndarray
        min_area: int
    OUTPUTS:
        total_area: int
        total_intensity: int
        object_count: int
    """
    perfect_psf = np.zeros((19, 19))
    perfect_psf[9, 9] = 1
    psf = gaussian(perfect_psf, 2)
    deconv_cell = unsupervised_wiener(cell, psf, clip=False)[0]
    cell_minus_struct = np.copy(deconv_cell)
    cell_minus_struct[removal_mask] = 0
    
    print(np.nansum(cell))
    print(np.std(psf))
    print(np.nansum(deconv_cell))
    print(np.nansum(removal_mask))
    

    thresh = threshold_otsu(cell_minus_struct)
    labeled, num_labels = label(cell_minus_struct > thresh, return_num=True, connectivity=1)
    labels = np.linspace(1, num_labels, num_labels).astype(int)
    labeled = clear_border(labeled)
    
    areas, intensities = [], []
    for lbl in labels:
        area = np.nansum(labeled[labeled == lbl])
        if area >= min_area:
            intensity = np.nansum(cell[labeled==lbl])
            areas.append(area)
            intensities.append(intensity)
                

    return np.nansum(areas), np.nansum(intensities), len(areas)


def measure_whole_cell(cell: np.ndarray) -> Tuple[int, float, float]:
    """
    Measures whole cell area, total intensity, and average intensity.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
    OUTPUTS:
        area: int
        intensity: float
        avg_intensity: float
    """
    thresh = threshold_otsu(cell)
    labeled = clear_border(label(cell > thresh))
    mask = labeled > 0
    return np.nansum(mask),np.nansum(cell[mask]), np.nanmean(cell[mask])


def unaligned_chromatin(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: np.ndarray,
    chromatin: np.ndarray,
    min_chromatin_area: Optional[int] = 4
) -> tuple[list[int], list[int], list[float], list[int], list[float], list[int], int]:
    """
    Given an image capturing histone fluoresence, returns the area emitting of signal emitting regions minus the
    area of the largest signal emitting region (corresponds with unaligned chromosomes in metaphase)
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        identity: int
        analysis_df: pd.DataFrame
        instance: np.ndarray
        chromatin: np.ndarray
        min_chromatin_area: Optional[int]
    OUTPUTS:
        area_signal: List[int]
        intensity_signal: List[int]
        whole_cell_intensity: List[float]
        num_signals: List[int]
        whole_cell_avg_intensity: List[float]
        whole_cell_area: List[int]
        first_anaphase: int
    """
    zoom_factor = adjust_zoom_factor(chromatin.shape, instance.shape)
    frames_data = analysis_df.query(f"particle == {identity}")
    semantics = frames_data["semantic_smoothed"].tolist()

    results = []
    anaphase_indices = []
    print(f'Working on cell {identity}')
    
    for idx, row in frames_data.iterrows():
        f, l, semantic = int(row["frame"]), int(row["label"]), int(row["semantic_smoothed"])

        mask = instance[f] == l
        if zoom_factor != 1:
            mask = zoom(mask, zoom_factor, order=0)
        mask = binary_dilation(mask, disk(3))

        bbox_coords = bbox(mask)
        cell, nobkg_cell = prepare_cell_image(chromatin, f, bbox_coords)

        if semantic == 1:
            labeled_regions, max_lbl, second_lbl, intensity_diff_ratio = get_largest_signal_regions(nobkg_cell, cell, num_regions=2)

            if second_lbl:
                to_check = idx+9 if (idx+9) < len(semantics) else -1
                near_end_of_mitosis = any(s == 0 for s in semantics[idx:to_check]) 
                
                if intensity_diff_ratio < (1 / 3) and near_end_of_mitosis:
                    print('anaphase; removing blobs')
                    removal_mask = remove_regions([max_lbl, second_lbl], labeled_regions)
                    anaphase_indices.append(idx)
                else:
                    removal_mask = remove_metaphase_if_eccentric(max_lbl, labeled_regions)
            else:
                removal_mask = remove_metaphase_if_eccentric(max_lbl, labeled_regions)

            if len(anaphase_indices) > 0:
                consecutives = np.split(
                    anaphase_indices, np.where(np.diff(anaphase_indices) != 1)[0] + 1
                )
                for sublist in consecutives:
                    if len(sublist) > 1:
                        first_anaphase = sublist[0]
                        break
                    else:
                        first_anaphase = anaphase_indices[0]
            else:
                # anaphase may not be detected due to finite temporal resolution
                first_anaphase = None

            area_sig, int_sig, num_sig = segment_unaligned_chromosomes(cell, removal_mask, min_chromatin_area)
        else:
            area_sig, int_sig, num_sig = 0, 0, 0

        whole_area, whole_intensity, whole_avg_intensity = measure_whole_cell(cell)

        results.append((area_sig, int_sig, whole_intensity, num_sig, whole_avg_intensity, whole_area))
        
    area_signal, intensity_signal, whole_cell_intensity, num_signals, whole_cell_avg_intensity, whole_cell_area = zip(*results)

    return (
        list(area_signal),
        list(intensity_signal),
        list(whole_cell_intensity),
        list(num_signals),
        list(whole_cell_avg_intensity),
        list(whole_cell_area),
        first_anaphase,
    )

def bbox(img: npt.NDArray) -> tuple[int]:
    """
    Returns the minimum bounding box of a boolean array containing one region of True values
    ------------------------------------------------------------------------------------------
    INPUTS:
        img: npt.NDArray
    OUTPUTS:
        rmin: float, lower row index
        rmax: float, upper row index
        cmin: float, lower column index
        cmax: float, upper column index
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def watershed_split(
    binary_image: npt.NDArray, sigma: Optional[float] = 3.5
) -> npt.NDArray:
    """
    Splits and labels touching objects using the watershed algorithm
    ------------------------------------------------------------------
    INPUTS:
        binary_img: image where 1s correspond to object regions and 0s correspond to background
        sigma: standard deviation to use for gaussian kernal smoothing
    OUTPUTS:
        labels: image of same size as binary_img but labeled
    """

    # distance transform
    distance = ndi.distance_transform_edt(
        binary_closing(binary_image, skimage.morphology.disk(9))
    )
    blurred_distance = gaussian(distance, sigma=sigma)

    # finding peaks in the distance transform
    coords = peak_local_max(blurred_distance, labels=binary_image)

    # creating markers and segmenting
    mask = np.zeros(binary_image.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    labels = watershed(-blurred_distance, markers, mask=binary_image)

    return labels


def extract_montages(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: npt.NDArray,
    chromatin: npt.NDArray,
    cmap: Optional[str] = None,
    well_pos: Optional[str] = None,
    save_path: Optional[str] = None,
    mode: Optional[str] = "seg",
):
    """
    Function for saving montages of ROIs
    ------------------------------------
    INPUTS:
        identity: int,
        analysis_df: pd.DataFrame.
        instance: npt.NDArray,
        chromatin: npt.NDArray,
        well_pos: str,
        save_path:str
    TODO:
        color masks by area extracted
    """

    if chromatin.shape != instance.shape:
        try:
            assert (
                chromatin.shape[1] / instance.shape[1]
                == chromatin.shape[2] / instance.shape[2]
            )
            zoom_factor = chromatin.shape[1] / instance.shape[1]
        except AssertionError:
            print("Chromatin and Instance must be square arrays")
            return

    rois = []
    frames = analysis_df.query(f"particle=={identity}")["frame"].tolist()
    markers = analysis_df.query(f"particle=={identity}")["label"].tolist()
    semantic = analysis_df.query(f"particle=={identity}")["semantic"].tolist()

    for index, zipped in enumerate(zip(frames, markers, semantic)):

        f, l, classifier = zipped
        # expand mask and capture indices
        mask = instance[f, :, :] == l
        if zoom_factor != 1:
            mask = ndi.zoom(mask, zoom_factor, order=0)
        zoom_mask = binary_dilation(mask, skimage.morphology.disk(3))
        rmin, rmax, cmin, cmax = bbox(zoom_mask)

        # trim cell and mask for efficiency
        cell = chromatin[f, rmin:rmax, cmin:cmax]
        nobkg_cell = white_tophat(cell, skimage.morphology.disk(5))
        nobkg_cell = gaussian(nobkg_cell, sigma=1.5)
        zoom_mask = zoom_mask[rmin:rmax, cmin:cmax]

        if mode != "cell":
            if classifier == 1:
                # find first aggresive threshold
                thresh = threshold_otsu(nobkg_cell)

                # threshold and label bkg subtracted cell
                thresh_cell, num_labels = label(
                    nobkg_cell > thresh, return_num=True, connectivity=1
                )
                labels = np.linspace(1, num_labels, num_labels).astype(int)
                # find the label corresponding to the maximum cummulative intensity image
                region_intensities = [
                    np.sum(cell[thresh_cell == (lbl)]) for lbl in labels
                ]
                max_intensity = max(region_intensities)
                max_intensity_lbl = labels[region_intensities.index(max_intensity)]
                if len(region_intensities) > 1:
                    second_max_intensity = sorted(region_intensities)[-2]
                    nxt_max_intensity_lbl = labels[
                        region_intensities.index(second_max_intensity)
                    ]
                    intensity_diff = (
                        max_intensity - second_max_intensity
                    ) / max_intensity

                # if we are two or one timepoint away from cytokensis
                index_to_check = (
                    index + 8 if len(semantic) > index + 8 else len(semantic) - 1
                )
                if (
                    semantic[index_to_check] == 1
                    and intensity_diff < (7 / 8)
                    and len(region_intensities) > 1
                ):
                    anaphase_blobs_mask = np.zeros_like(thresh_cell)
                    anaphase_blobs_mask[thresh_cell == max_intensity_lbl] = 1
                    anaphase_blobs_mask[thresh_cell == nxt_max_intensity_lbl] = 1
                    anaphase_blobs_mask = binary_dilation(
                        anaphase_blobs_mask, skimage.morphology.disk(9)
                    )
                    to_remove_mask = anaphase_blobs_mask
                    print("anaphase; removing blobs")

                else:
                    # grab the metaphase plate
                    metphs_plate_mask = np.copy(thresh_cell)
                    metphs_plate_mask[thresh_cell != max_intensity_lbl] = 0
                    # dilate the metaphase plate
                    metphs_plate_mask = binary_dilation(
                        metphs_plate_mask, skimage.morphology.disk(9)
                    )
                    if np.sum(metphs_plate_mask) != 0:
                        eccen = regionprops(label(metphs_plate_mask))[0]["eccentricity"]
                        if eccen < 0.7:
                            metphs_plate_mask = np.zeros_like(metphs_plate_mask)
                            print("metaphase; not removing metaphase plate")
                        else:
                            print("metaphase; removing")
                    to_remove_mask = metphs_plate_mask
            else:
                to_remove_mask = np.zeros_like(cell)

            # deconvolve and create cell image with metaphase plate removed
            perfect_psf = np.zeros((25, 25))
            perfect_psf[12, 12] = 1
            psf = gaussian(perfect_psf, 2)
            deconv_cell = skimage.restoration.unsupervised_wiener(
                cell, psf, clip=False
            )[0]
            cell_minus_struct = np.copy(deconv_cell)
            cell_minus_struct[to_remove_mask] = 0

            # threshold cell with metaphase plate removed
            thresh2 = threshold_otsu(cell_minus_struct)
            thresh_cell2, num_labels2 = label(
                cell_minus_struct > thresh2, return_num=True, connectivity=1
            )
            thresh_cell2 = clear_border(thresh_cell2)
            labels2 = np.linspace(1, num_labels2, num_labels2).astype(int)

            borders = binary_dilation(thresh_cell2) ^ binary_erosion(thresh_cell2)
            with_borders = np.copy(cell)
            with_borders[borders] = 0
            rois.append(with_borders)
        else:
            rois.append(cell)

    max_rows = max([roi.shape[0] for roi in rois])
    max_cols = max([roi.shape[1] for roi in rois])

    rois_new = []
    for roi in rois:
        template = np.zeros((max_rows, max_cols))
        template[: roi.shape[0], : roi.shape[1]] = roi
        rois_new.append(template)

    rois = np.asarray(rois_new)

    if not well_pos:
        display_save(rois, save_path, identity, cmap=cmap, step=2)
    else:
        display_save(rois, save_path, identity, step=2, cmap=cmap, well_pos=well_pos)

    return rois


def display_save(
    rois: npt.NDArray,
    path: str,
    identity: int,
    cmap="gray",
    step=2,
    well_pos: Optional[str] = None,
):
    """
    Function for creating montages
    ------------------------------
    INPUTS
        rois: npt.NDArray, array with shape = (t, x, y) where t is the number of timepoints
        path: str
        identity: int
        cmap: str
        step: int
        well_pos: str

    """
    data_montage = skimage.util.montage(rois[0::step], padding_width=4, fill=np.nan)
    fig, ax = plt.subplots(figsize=(40, 40))
    ax.imshow(data_montage, cmap=cmap)
    title = f"Cell {identity} from {well_pos}" if well_pos else f"Cell {identity}"
    ax.set_title(title)
    ax.set_axis_off()

    if path != None:
        fig.savefig(path, dpi=300)


def cycb_chromatin_batch_analyze(
    positions: list[str],
    analysis_path_templ: str,
    instance_path_templ: str,
    chromatin_path_templ: str,
) -> tuple[pd.DataFrame]:

    for pos in positions:
        instance_path = instance_path_templ.format(pos, pos)
        analysis_path = analysis_path_templ.format(pos, pos)
        chromatin_path = chromatin_path_templ.format(pos)

        try:
            instance = tiff.imread(instance_path)
            chromatin = tiff.imread(chromatin_path)
            analysis_df = pd.read_excel(analysis_path)
        except FileNotFoundError:
            print(
                f"Could not find either the instance movie, chromatin movie, or analysis dataframe for {pos}"
            )
            continue

        analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        analysis_df.dropna(inplace=True)

        print(f"Working on position: {pos}")
        traces, ids = retrieve_traces(analysis_df, "GFP", 4)
        intensity, semantic, first_tps = qual_deg(traces, 4)

        un_chromatin = []
        un_intensity = []
        tot_intensity = []
        un_number = []
        tot_avg_intensity = []
        tot_area = []
        first_anaphase = []
        unaligned_chromosomes = [
            unaligned_chromatin(identity, analysis_df, instance, chromatin)
            for i, identity in enumerate(ids)
        ]
        for i, data_tuple in enumerate(unaligned_chromosomes):
            un_chromatin.append(data_tuple[0])
            un_intensity.append(data_tuple[1])
            tot_intensity.append(data_tuple[2])
            un_number.append(data_tuple[3])
            tot_avg_intensity.append(data_tuple[4])
            tot_area.append(data_tuple[5])
            first_anaphase.append(data_tuple[6])

        cycb = pd.DataFrame(intensity)
        classification = pd.DataFrame(semantic)
        un_chromatin_area = pd.DataFrame(un_chromatin)
        un_chromatin_intensity = pd.DataFrame(un_intensity)
        un_chromatin_number = pd.DataFrame(un_number)
        total_intensity = pd.DataFrame(tot_intensity)
        total_avg_intensity = pd.DataFrame(tot_avg_intensity)
        total_area = pd.DataFrame(tot_area)

        d_temp = {
            "ids": ids,
            "first_mitosis": first_tps,
            "first_anaphase": first_anaphase,
            "raw_data_paths": [instance_path, analysis_path, chromatin_path],
        }
        other_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d_temp.items()]))

        save_dir = os.path.dirname(analysis_path)
        if not os.path.isdir(save_dir):
            save_dir = os.getcwd()
        save_path = os.path.join(save_dir, f"cycb_chromatin_{pos}.xlsx")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            cycb.to_excel(writer, sheet_name="cycb")
            classification.to_excel(writer, sheet_name="classification")
            un_chromatin_area.to_excel(writer, sheet_name="unaligned chromatin area")
            un_chromatin_intensity.to_excel(
                writer, sheet_name="unaligned chromatin intensity"
            )
            total_intensity.to_excel(writer, sheet_name="total chromatin intensity")
            un_chromatin_number.to_excel(
                writer, sheet_name="number of unaligned chromosomes (approx.)"
            )
            total_avg_intensity.to_excel(
                writer, sheet_name="average chromatin intensity"
            )
            total_avg_intensity.to_excel(writer, sheet_name="total chromatin area")
            other_data.to_excel(writer, sheet_name="analysis_info")


if __name__ == "__main__":
    positions = [
        "A02_s2",
        "A02_s3",
        "A02_s5",
        "B02_s1",
        "B02_s7",
        "B02_s8",
        "E02_s1",
        "E02_s3",
        "E02_s4",
        "F02_s3",
        "F02_s9",
        "F02_s10",
        "G02_s1",
        "G02_s2",
        "H02_s1",
        "H02_s4",
        "H02_s10",
    ]
    analysis_path_templ = "/scratch/ajitj_root/ajitj99/anishjv/for_analysis_complete/20250203_20250203-CycB-pFF1_{}_phs_HeLa_1800_0.35_inference/20250203_20250203-CycB-pFF1_{}_analysis.xlsx"

    instance_path_templ = "/scratch/ajitj_root/ajitj99/anishjv/for_analysis_complete/20250203_20250203-CycB-pFF1_{}_phs_HeLa_1800_0.35_inference/20250203_20250203-CycB-pFF1_{}_instance_movie.tif"

    chromatin_path_templ = "/scratch/ajitj_root/ajitj99/anishjv/for_analysis/1/20250203_20250203-CycB-pFF1_{}_Texas Red.tif"

    cycb_chromatin_batch_analyze(
        positions, analysis_path_templ, instance_path_templ, chromatin_path_templ
    )
