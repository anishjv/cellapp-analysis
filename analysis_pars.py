import os
import pandas as pd
import numpy as np
from skimage.morphology import disk

class analysis_pars:

    def __init__(self, cell_type = "hela"):
        self.cell_types = ["hela", "u2os", "rpe1", "ht1080"]

        try:
            if cell_type.lower() in self.cell_types:
                self.current_cell_type = cell_type
        except:
            raise ValueError(f"Choose one of {self.cell_types}")
        
        # Parameters for pre-processing inferred cell segmentations
        self.erode_footprint = disk(3)
        self.max_cell_size = 4000
        self.min_cell_size = 500

        # Median filter size for smoothing semantic label trace
        self.min_mitotic_duration = 30 # minutes
        self.frame_interval = 10 #  time step in min
        # Must be odd for median filtering to work
        self.min_mitotic_duration_in_frames = self.min_mitotic_duration // self.frame_interval # number of frames
        self.semantic_footprint = np.ones(self.min_mitotic_duration_in_frames)

        # trackpy parameters
        self.max_pixel_movement = 20
        self.tracking_memory    = 2
        self.min_track_length   = 30 # min track length

        self.adaptive_tracking = False

        if cell_type.lower() == "ht1080":
            self.max_pixel_movement = 40
            self.max_cell_size = 5000
            self.adaptive_tracking = True

        if cell_type.lower() == "u2os":
            self.max_pixel_movement = 30
            self.max_cell_size = 5000
            self.adaptive_tracking = True

        if cell_type.lower() == "rpe1":
            self.max_pixel_movement = 30
            self.max_cell_size = 5000
            self.adaptive_tracking = True