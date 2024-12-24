import sys, os
from pathlib import Path
# import napari # type: ignore
# import numpy as np
# import pandas as pd
# import trackpy as tp
# from scipy import ndimage as ndi
import matplotlib.pyplot as plt
plt.ion()

# sys.path.append('/Users/ajitj/Library/CloudStorage/GoogleDrive-ajitj@umich.edu/My Drive/ImageAnalysis/cell_analysis')
sys.path.append('/Users/ajitj/Google Drive/ImageAnalysis/cell_analysis')
import cellaap_analysis

root_path = Path('/Users/ajitj/Desktop/current/')
folder_path = Path('/Users/ajitj/Desktop/current/20241205_Bub1 ppg1 ppg2 pps121_D12_s1_phs_inference')
# folder_path = Path('/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/Ajit_Joglekar/M6_spacer/20230619/pPS22-pPS124/2023-06-19/20259/20230619_pPS22-pPS124_B09_s3_phs_HeLa_2000_0.25_inference')
t = cellaap_analysis.analysis(root_path)
t.files(folder_path)
tracks = t.track_centroids('HeLa', False)
tracks = t.measure_signal('Texas Red', True, -1)
# tracks = t.measure_signal('GFP', True, -1)
summary = t.summarize_data(True)
