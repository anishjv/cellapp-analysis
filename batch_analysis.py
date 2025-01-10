import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import napari
# sys.path.append('/Users/ajitj/Library/CloudStorage/GoogleDrive-ajitj@umich.edu/My Drive/ImageAnalysis/cell_analysis')
sys.path.append('/Users/ajitj/Google Drive/ImageAnalysis/cell_analysis')
import cellaap_analysis

root_path = Path('')

# map_dict = {"GFP_background"      :'',
#             "GFP_intensity"       :'',
#             "TexasRed_background" :'',
#             "TexasRed_intensity"  :''}

root_path = Path('/Users/ajitj/Desktop/RPE1-U2OS')
t = cellaap_analysis.analysis(root_path, analysis_only = False)
# t.create_correction_maps(map_dict)
inference_dirs = []
for dir in root_path.glob('*A04_s1*inference'):
    inference_dirs.append(dir)

for dir in inference_dirs:
    if hasattr(t, 'summaryDF'):
       delattr(t, 'summaryDF')
       delattr(t, 'tracked')
    t.files(dir, cell_type = 'rpe1')
    tracks = t.track_centroids(False)
    # tracks = t.measure_signal('Texas Red', True, -1)
    # tracks = t.measure_signal('GFP', True, -1)
    # summary = t.summarize_data(True)


# folder_path = Path('/Users/ajitj/Desktop/current/20241205_Bub1 ppg1 ppg2 pps121_F12_s3_phs_HeLa_2000_0.25_inference')
# folder_path = Path('/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/Ajit_Joglekar/M6_spacer/20230619/pPS22-pPS124/2023-06-19/20259/20230619_pPS22-pPS124_B09_s3_phs_HeLa_2000_0.25_inference')



