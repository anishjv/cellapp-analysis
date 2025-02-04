import os, tifffile
from pathlib import Path, PurePath
from skimage.io import imread # type: ignore
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from cellaap_utils import *


min_width = 3


def summarize_data(tracked, save_flag: True):
    '''
    Summarizes data stored in the tracked dataframe; operates on all measured channels.
    Filtering - Two strict filters are applies. Cell is summarized only if:
                * If the cell is labeled as mitotic in frame 0 or frame -1
                * If the label trace has only one peak
    Inputs - 
    save_flag : whether to export the data as an xlsx file

    Outputs - 
    None
    '''
    # Select only those tracks where mitosis was observed
    idlist    = list(set(tracked[tracked.mitotic==1].particle))
    mitosis   = []
    mito_start= []
    cell_area = []
    particle  = []
    channels = []
    # Check which channels have been measured. If none, return only "mitotic duration"
    # Need to find a better way to code this.

    if "GFP" in tracked.columns:
        channels.append('GFP')
    if "Texas_Red" in tracked.columns:
        channels.append("Texas_Red")
    if "Cy5" in tracked.columns:
        channels.append("Cy5")
    signal_storage = {}
    for channel in channels:
        signal_storage[f'{channel}'] = []
        signal_storage[f'{channel}_std'] = []
        signal_storage[f'{channel}_bkg_corr'] = []
        signal_storage[f'{channel}_bkg_corr_std'] = []
        signal_storage[f'{channel}_int_corr'] = []
        signal_storage[f'{channel}_int_corr_std'] = []

    for id in idlist:
        semantic = tracked[tracked.particle==id].semantic
        _, props = find_peaks(semantic, width=min_width)
        
        # Only select tracks that have one peak in the semantic trace
        # This will bias the analysis to smaller mitotic durations
        if props["widths"].size == 1:
            mitosis.append(props["widths"][0])
            mito_start.append(props['left_bases'][0])
            cell_area.append(tracked[tracked.particle==id].area.mean())
            particle.append(id)
            
        
            for channel in channels:
                signal, bkg_corr, int_corr, signal_std, bkg_std, int_std = calculate_signal(
                                                                semantic, 
                                                                tracked[tracked.particle==id][f'{channel}'].to_numpy(), 
                                                                tracked[tracked.particle==id][f'{channel}_bkg_corr'].to_numpy(), 
                                                                tracked[tracked.particle==id][f'{channel}_int_corr'].to_numpy(),
                                                                min_width
                                                                )
                signal_storage[f'{channel}'].append(signal)
                signal_storage[f'{channel}_std'].append(signal_std)
                signal_storage[f'{channel}_bkg_corr'].append(bkg_corr)
                signal_storage[f'{channel}_bkg_corr_std'].append(bkg_std)
                signal_storage[f'{channel}_int_corr'].append(int_corr)
                signal_storage[f'{channel}_int_corr_std'].append(int_std)
            
    
    # Construct summary DF
    other_storage = {
                    "particle"  : particle,
                    "cell_area" : cell_area,
                    "mito_start": mito_start,
                    "mitosis"   : mitosis,
                    }
    
    summary_storage = other_storage | signal_storage
    summaryDF = pd.DataFrame(summary_storage)

    return summaryDF