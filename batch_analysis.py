import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import napari
sys.path.append('/Users/ajitj/Google Drive/ImageAnalysis/cell_analysis')
import cellaap_analysis

# map_dict = {"GFP_background"      :Path('/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/precious/20250130/pPS28 Halo and pPS18 HelLa dox and/2025-01-30/20341/20250130_pPS28 Halo and pPS18 HelLa dox and_H07_s2_GFP.tif'),
#             "GFP_intensity"       :Path('/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/precious/20250130/pPS28 Halo and pPS18 HelLa dox and/2025-01-30/20341/20250130_pPS28 Halo and pPS18 HelLa dox and_G07_s1_GFP.tif'),
#             "TexasRed_background" :Path('/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/precious/20250130/pPS28 Halo and pPS18 HelLa dox and/2025-01-30/20341/20250130_pPS28 Halo and pPS18 HelLa dox and_H07_s2_Texas Red.tif'),
#             "TexasRed_intensity"  :Path('/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/precious/20250130/pPS28 Halo and pPS18 HelLa dox and/2025-01-30/20341/20250130_pPS28 Halo and pPS18 HelLa dox and_G07_s1_Texas Red.tif')}

root_path = Path('/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/precious/20250130/pPS28 Halo and pPS18 HelLa dox and/2025-01-30/20341')
t = cellaap_analysis.analysis(root_path, plotting_only= False)
# t.create_correction_maps(map_dict)

inference_dirs = []
for dir in root_path.glob('*_E07_*_inference'):
    inference_dirs.append(dir)
for dir in root_path.glob('*_F07_*_inference'):
    inference_dirs.append(dir)

for dir in inference_dirs:
    if hasattr(t, 'summaryDF'):
       delattr(t, 'summaryDF')
       delattr(t, 'tracked')
    t.files(dir, cell_type = 'u2os')
    tracks = t.track_centroids(False)
    tracks = t.measure_signal('Texas Red', True, -1)
    tracks = t.measure_signal('GFP', True, -1)
    summary = t.summarize_data(True)



# def main(root_path, create_correction_maps, inference_dirs, cell_type, to_measure, map_dict):
#     t = cellaap_analysis.analysis(root_path, analysis_only = False)
#     if create_correction_maps:
#         t.create_correction_maps(map_dict)


#     for dir in inference_dirs:
#         if hasattr(t, 'summaryDF'):
#             delattr(t, 'summaryDF')
#             delattr(t, 'tracked')
#         t.files(dir, cell_type = cell_type)
#         t.track_centroids(save_flag=True)
#         for wavelength in to_measure:
#             tracks = t.measure_signal(wavelength, True, -1)

#         summary = t.summarize_data(True)

# if __name__ == "__main__":
#     main()




