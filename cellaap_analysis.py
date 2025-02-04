import os, tifffile
from pathlib import Path, PurePath, PosixPath
from skimage.io import imread # type: ignore
from skimage.morphology import erosion
from skimage.filters.rank import minimum
from skimage.measure import regionprops_table, block_reduce
import napari # type: ignore
import numpy as np
import pandas as pd
import trackpy as tp
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from analysis_pars import analysis_pars
from cellaap_utils import *
from skimage.util import img_as_uint

class analysis:
    
    def __init__(self, root_folder: Path, plotting_only: False):
        '''
        Object initializes with default parameter values and definitions of 
        root and inference folders. It also reads in tif files with either 
        intensity or background in their names as the corresponding, channel-
        specific correction maps.
        '''
        # self.__dict__.update((key, False) for key in self.suffixes)
        # self.__dict__.update((key, value) for key, value in file_dict.items() if key in self.suffixes)

        ## Default parameters
        self.paths = {}  # Dictionary stores stack paths
        self.stacks = {} # Dictionary stores image stacks

        ##
        try:
            root_folder.exists()
        except:
            raise ValueError(f"{root_folder} not a valid path")
        
        self.root_folder = root_folder
        self.inference_folders = [] # list of folders with _inference in their names
        for directory in os.scandir(root_folder):
            if "_inference" in directory.name:
                self.inference_folders.append(directory)

        ## Check for and set path names for correction maps
        # These maps are loaded with the root directory because they apply to all 
        # segmentations in the root directory.
        self.intensity_map_present = False
        self.background_map_present = False
        
        if not plotting_only:
            for name in self.root_folder.glob("*_map.tif"):
                map_type = name.name.split('_')[-2]
                channel_name = name.name.split('_')[-3]
                if channel_name == "Texas Red":
                    channel_name = "Texas_Red"

                match map_type:
                    case "intensity":
                        self.paths["intensity_map"] = Path(name)
                        self.stacks[channel_name + "_intensity_map"] = tifffile.imread(Path(name))
                        self.intensity_map_present = True
                        print(f"{name.name} used as the {channel_name} intensity map")

                    case "background":
                        self.paths["background_map"] = Path(name)
                        self.stacks[channel_name+"_background_map"] = imread(Path(name))
                        self.background_map_present = True
                        print(f"{name.name} used as the {channel_name} background map")
        else:
            print(f"Opening {root_folder} in plotting only mode.")


    def files(self, cellaap_dir: Path, cell_type: str):
        '''
        Inputs:
        cellaap_dir: directory containing cellapp inference; must contain "instance" and "semnatic" tif files
        cell_type: specify the cell type so approprite default pars are set
        '''
        # Process the path objects to retrieve the parent directories and suffixes
        try:
            self.cellaap_dir = cellaap_dir
            self.paths["instance"] = Path([name for name in cellaap_dir.glob('*.tif') if "instance" in name.name][0])
            self.paths["semantic"] = Path([name for name in cellaap_dir.glob('*.tif') if "semantic" in name.name][0])
            # Keep the name stub to infer other file names
            self.name_stub = self.paths["instance"].name.split('_phs')[0]
            self.defaults = analysis_pars(cell_type=cell_type)
        except:
            raise ValueError("Instance and/or semantic segmentations not found!")

        # Set path names for existing channel files
        self.data_dir = self.cellaap_dir.parent
        for name in self.root_folder.glob(self.name_stub+'*.tif'):
            print(f"{name.name} can be used for analysis and display")
            channel = name.stem.split('_')[-1]
            match channel:
                case "phs":
                    self.paths["phase"] = Path(name)
                case "GFP":
                    self.paths["GFP"] = Path(name)
                case "Texas Red":
                    self.paths["Texas_Red"] = Path(name)
                case 'Cy5':
                    self.paths["Cy5"] = Path(name)
                
        # Only read instance and semantic stacks
        self.stacks["semantic"] = imread(self.paths["semantic"])
        self.stacks["instance"] = imread(self.paths["instance"])
        
        # Apply erosion before zooming to avoid duplicate computations
        instance_shape = self.stacks["instance"].shape
        # Saving the number of planes for track filtering (summarize_data)
        self.max_timepoints = instance_shape[0]

        instance_zoomed = np.zeros((instance_shape[0], instance_shape[1]*2, instance_shape[2]*2))
        print(f"Computing zoomed and eroded instance mask...")
        for i in np.arange(instance_shape[0]):
            pre_zoom = erosion(self.stacks["instance"][i,:,:].astype(np.int16),
                               self.defaults.erode_footprint)
            instance_zoomed[i, :, :] = ndi.zoom(pre_zoom, 2, order=0)
            
        self.stacks["instance_zoomed"] = instance_zoomed
        print(f"Finished computing zoomed and eroded instance mask!")
        return
    

    def create_correction_maps(self, type_file_dict: dict):
        '''
        Function uses the assigned background and intensity mapping stacks
        to create the corresponding background and intensity correction maps.
        the position_name_channel dictionary should include key-value pairs as:
        key: intensity or background , value: Path_to_file
        Currently, has to be used for each channel individually.
        '''

        for key, value in type_file_dict.items():
            if type(value) is PosixPath:
                if "intensity" in key:
                    intensity_map_name = value.parent / Path(value.stem + "_intensity_map.tif")
                    channel_map = value.stem.split('_')[-1] + "_intensity_map.tif"
                    self.stacks[channel_map] = gen_intensity_correction_map(imread(value))
                    tifffile.imsave(intensity_map_name, self.stacks[channel_map].astype(np.float16))
                    self.paths[channel_map] = intensity_map_name
                    print(f"Intensity map saved in the data dir. as {intensity_map_name}")

                elif "background" in key:
                    background_map_name = value.parent / Path(value.stem + "_background_map.tif")
                    channel_map = value.stem.split('_')[-1] + "_background_map.tif"
                    self.stacks[channel_map] = gen_background_correction_map(imread(value))
                    tifffile.imsave(background_map_name, self.stacks[channel_map].astype(np.int16))
                    self.paths[channel_map] = background_map_name
                    print(f"Background map saved in the data dir. as {background_map_name}")
            
            else:
                print(f"Dictionary values must be Path objects; try again.")

        return



    def track_centroids(self, save_flag: False, memory = None, max_pixel_movement = None) -> pd.DataFrame:
        '''
        Function uses the instance segmentation file generated by cell_aap and trackpy
        to track cell centroids.
        Inputs:
        saveflag  : Save the pandas dataframe as a csv file.
        Optionally, track memory and min. track length can be changed.
        Outputs:
        csv_name  : name of the csv file saved
        dataframe : dataframe with tracking information
        '''
        
        frames = self.stacks["instance"].shape[0] # type: ignore

        property_list  = ['centroid','area','eccentricity','bbox', 'label']

        df_list = []

        for i in np.arange(frames):
            props = regionprops_table(self.stacks["instance"][i,:,:], properties=property_list)
            props_table = pd.DataFrame(props)
            props_table["centroid-0"] = props_table["centroid-0"].apply(lambda x: int(x))
            props_table["centroid-1"] = props_table["centroid-1"].apply(lambda x: int(x))
            props_table["frame"] = i #trackpy needs this to track.
            df_list.append(props_table)

        track_table = pd.concat(df_list)

        # trackpy requirements for the dataframe it will use to link tracks
        track_table.rename(columns={"centroid-0":"x", "centroid-1":"y"}, inplace=True)
        # remove large and small cells
        track_table = track_table[(track_table.area < self.defaults.max_cell_size) & 
                                  (track_table.area > self.defaults.min_cell_size)]

        # Check if default values have been changed
        if memory is None:
            memory = self.defaults.tracking_memory
        if max_pixel_movement is None:
            max_pixel_movement = self.defaults.max_pixel_movement

        # Adaptive linking for cells that move a lot between two timesteps
        if self.defaults.adaptive_tracking:
            self.tracked = tp.link(track_table, 
                               max_pixel_movement, adaptive_stop=10, adaptive_step=0.9,
                               memory=memory)
        else:
            self.tracked = tp.link(track_table, max_pixel_movement, memory=memory)
        
        self.tracked = tp.filter_stubs(self.tracked, self.defaults.min_track_length) 
        
        # tracked annoying use the frame number as the row index.
        # So drop the frame column and reset the index to recreate it.
        self.tracked.drop(columns=["frame"], inplace=True)
        self.tracked.reset_index(inplace=True)
        # For displaying tracks in naapri
        self.tracked.sort_values(by=["particle","frame"],inplace=True)

        # Obtain the semantic label
        # This drops the old index and uses serial numbers
        self.tracked.reset_index(inplace=True)

        semantic_label = []
        for i in np.arange(len(self.tracked)):
            semantic_label.append(self.stacks["semantic"][self.tracked.loc[i,"frame"],
                                                      self.tracked.loc[i,"x"],  
                                                      self.tracked.loc[i,"y"]])
        self.tracked["semantic"] = semantic_label

        # classify the cells as dividing or non-dividing
        # observed division = 1; no division = 0

        for id in list(set(self.tracked.particle)):
            index  = self.tracked[self.tracked.particle==id].index

            if np.isin(100, self.tracked[self.tracked.particle==id].semantic):
                self.tracked.loc[index, "mitotic"] = 1
            else:
                self.tracked.loc[index, "mitotic"] = 0
        
        if save_flag:
            self.tracked.to_excel(self.cellaap_dir / Path(self.name_stub+"_tracks.xlsx"))

        return self.tracked
    
    def _display_tracks(self, img_stack = None):
        '''
        '''
        if "viewer" not in self.__dict__.keys():
            self.viewer = napari.Viewer()
        
        if img_stack is None:
            self.stacks["phase"] = imread(self.paths["phase"])
            phase_binned = np.zeros_like(self.stacks["instance"], dtype=int)
            for i in np.arange(phase_binned.shape[0]):
                phase_binned[i,:,:]=block_reduce(self.stacks["phase"][i,:,:,],block_size=(2,2), func=np.max)
        else:
            pass

        self.viewer.add_image(phase_binned)
        self.viewer.add_labels(self.stacks["semantic"])
        self.viewer.add_tracks(self.tracked[["particle","frame","x","y"]].to_numpy())

        return self.viewer

    
    def measure_signal(self, channel: str, save_flag: False, id = -1,):
        '''
        Measures the average cell signal over the eroded cell masks. Also
        calculates the position-dependent correction factors for background fluorescence and 
        excitation intensity variation. 
        Inputs:
        channel   - Channel Name; must be one of: (phase, GFP, Texas_Red, Cy5)
        save_flag - Whether to export dataframe as xlsx
        id        - ID of the cell (assigned by trackpy); -1 will analyze all cells
        '''
        try:
            if channel in []:
                pass
        except:
            raise ValueError(f"")
        
        if channel == "Texas Red":
            channel = "Texas_Red"
        
        # Read image stack
        channel_stack = imread(self.paths[channel])

        ##
        # if particle id is set to -1 - measure all particles. 
        # otherwise, just the specified particle.
        if id > -1:
            if type(id) == int:
                try:
                    if id in set(self.tracked.particle):
                        id_list = []
                        id_list.append(id)
                except:
                    raise ValueError(f"id not in the tracking list")
            
            elif type(id) == list:
                id_list = id
        else:
            id_list = list(set(self.tracked[self.tracked.mitotic==1].particle))
        
        # Default values for all entries
        self.tracked[channel] = np.nan
        self.tracked[channel+"_int_corr"] = 1.
        self.tracked[channel+"_bkg_corr"] = 0. 
        
        
        for id in id_list:
            print(f"Processing cell #{id}...")
            semantic = self.tracked[self.tracked.particle==id].semantic.to_list()
            # Measurement decision
            measure_cell = False
            if len(semantic) == self.max_timepoints:
                if semantic[0] == 1 & semantic[-1] == 1:
                    measure_cell = True
                else:
                    print(f"cell #{id} not measured; mitotic at start or end")
            else:
                if semantic[0] == 1:
                    measure_cell = True
                else:
                    print(f"cell #{id} not measured; mitotic at start")
            
            if measure_cell:
                frames = self.tracked[self.tracked.particle==id].frame.tolist()
                labels = self.tracked[self.tracked.particle==id].label.tolist()
                index  = self.tracked[self.tracked.particle==id].index
                
                signal = np.zeros(len(frames))
                background_correction = np.zeros_like(signal)
                intensity_correction = np.ones_like(signal)
                counter = 0
                for f,l in zip(frames, labels):
                    # print("Processing frame # {f}...")
                    mask = self.stacks["instance_zoomed"][f,:,:]==l

                    signal[counter] = mean_signal_from_mask(channel_stack[f,:,:], mask)
                    
                    if self.background_map_present:
                        map_name = channel + '_background_map'
                        background_correction[counter] = mean_signal_from_mask(self.stacks[map_name][f,:,:].astype(float), mask)
                    
                    if self.intensity_map_present:
                        map_name = channel + '_intensity_map'
                        intensity_correction[counter] = mean_signal_from_mask(self.stacks[map_name], mask)

                    counter = counter + 1
                
                self.tracked.loc[index, channel] = signal
                self.tracked.loc[index, channel+"_bkg_corr"] = background_correction
                self.tracked.loc[index, channel+"_int_corr"] = intensity_correction


        if save_flag:
            with pd.ExcelWriter(self.cellaap_dir / Path(self.name_stub+'_analysis.xlsx')) as writer:  
                self.tracked.to_excel(writer, sheet_name='cell_data')
                # There are no scalars - so turn into list; transform
                pd.DataFrame([self.paths]).T.to_excel(writer,   sheet_name='file_data')
            # self.tracked.to_excel()

        return self.tracked
    
    
    def summarize_data(self, save_flag: True):
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
        idlist    = list(set(self.tracked[self.tracked.mitotic==1].particle))
        mitosis   = []
        mito_start= []
        cell_area = []
        particle  = []
        channels = []
        # Check which channels have been measured. If none, return only "mitotic duration"
        # Need to find a better way to code this.

        if "GFP" in self.tracked.columns:
            channels.append('GFP')
        if "Texas_Red" in self.tracked.columns:
            channels.append("Texas_Red")
        if "Cy5" in self.tracked.columns:
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
            semantic = self.tracked[self.tracked.particle==id].semantic
            _, props = find_peaks(semantic, width=self.defaults.min_width)
            
            # Only select tracks that have one peak in the semantic trace
            # This will bias the analysis to smaller mitotic durations
            if props["widths"].size == 1:
                mitosis.append(props["widths"][0])
                mito_start.append(props['left_bases'][0])
                cell_area.append(self.tracked[self.tracked.particle==id].area.mean())
                particle.append(id)
                
           
                for channel in channels:
                    signal, bkg_corr, int_corr, signal_std, bkg_std, int_std = calculate_signal(
                                                                  semantic, 
                                                                  self.tracked[self.tracked.particle==id][f'{channel}'].to_numpy(), 
                                                                  self.tracked[self.tracked.particle==id][f'{channel}_bkg_corr'].to_numpy(), 
                                                                  self.tracked[self.tracked.particle==id][f'{channel}_int_corr'].to_numpy(),
                                                                  self.defaults.min_width
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
        self.summaryDF = pd.DataFrame(summary_storage)


        if save_flag:
            self.summaryDF.to_excel(self.cellaap_dir / Path(self.name_stub+"_summary.xlsx"))

        return self.summaryDF
    
    
    def gather_plot_summaries(self, well_position: list) -> pd.DataFrame:
        '''
        Collects and concatenates the summary xslx spreadsheets from the designated well_position list.
        
        Inputs:
        well_position : list with entries of the form r"[A-Z][dd]+_+[a-z][d+]"
        Output: 
        data_summary  : dataframe with the data concatenated; well_pos - column designating well_pos
        '''
        if well_position:
            if not hasattr(self, 'inf_folder_list'):
                # Assemble the folder list when function called for the first time
                self.inf_folder_list = [f for f in self.root_folder.glob('*_inference')]
            
            df_list = []
            experiment = self.root_folder.parents[-1]
            for wp in well_position:
                for f in self.inf_folder_list:
                    if wp in f.name:
                        xls_file_name = [file for file in f.glob('*_summary.xlsx')]
                        if xls_file_name:
                            df = pd.read_excel(xls_file_name[0]) #assumes only one
                            df["well_pos"] = wp #assign well-position identifier
                            df["experiment"] = experiment
                            df_list.append(df)
                            print(f"{wp} loaded")
                        else:
                            print(f"No summary file found in {wp}")

            data_summary = pd.concat(df_list)

        return data_summary