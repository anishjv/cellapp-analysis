import os, tifffile
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
from scipy.signal import find_peaks, medfilt
# import matplotlib.pyplot as plt
import napari
from analysis_pars import analysis_pars


class analysis:
    
    def __init__(self, root_folder: Path, analysis_only: False):
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
        
        if not analysis_only:
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
                        print(f"{name.name} used as {channel_name} intensity map")

                    case "background":
                        self.paths["background_map"] = Path(name)
                        self.stacks[channel_name+"_background_map"] = imread(Path(name))
                        self.background_map_present = True
                        print(f"{name.name} used as {channel_name} background map")
        else:
            print(f"Opening {root_folder} in analysis mode.")


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
        instance_zoomed = np.zeros((instance_shape[0], instance_shape[1]*2, instance_shape[2]*2))
        print(f"Computing zoomed and eroded instance mask...")
        for i in np.arange(instance_shape[0]):
            pre_zoom = erosion(self.stacks["instance"][i,:,:],self.defaults.erode_footprint)
            instance_zoomed[i, :, :] = ndi.zoom(pre_zoom, 2, order=0)
            
        self.stacks["instance_zoomed"] = instance_zoomed
        print(f"Finished computing zoomed and eroded instance mask!")
        return
    
    def _projection(self, im_array: np.ndarray, projection_type: str):

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
    
    def _gen_intensity_correction_map(self, image: npt.NDArray) -> npt.NDArray:
        """
        From Anish
        Computes the intensity map for flouresence microscopy intensity normalization if the input is a blank with flourescent media
        ----------------------------------------------------------------------------------------------------------------------------
        INPUTS:
            image: npt.NDArray
        OUTPUTPS:
            intensity_map: npt.NDArray
        """
        mean_plane = self._projection(image, "average")
        # med_filtered_mean_plane = ndi.median_filter(mean_plane, 9)
        smoothed_mean_plane = gaussian(mean_plane, 45)
        intensity_correction_map = smoothed_mean_plane / (np.max(smoothed_mean_plane))

        return intensity_correction_map
    
    def _gen_background_correction_map(self, background_stack: npt.NDArray) -> npt.NDArray:
        '''
        Newly written to avoid too much smoothing. The cMOS camera has a persistent noise pattern
        therefore, it is better to keep the corrections local. 
        '''

        background_correction_map = np.zeros_like(background_stack, dtype=int)
        footprint = footprint=np.ones((3,3))
        for i in np.arange( background_stack.shape[0]):
            background_correction_map[i,:,:] = ndi.median_filter(background_stack[i,:,:], footprint=footprint)

        return background_correction_map

    def create_correction_maps(self, type_file_dict: dict):
        '''
        Function uses the assigned background and intensity mapping stacks
        to create the corresponding background and intensity correction maps.
        the position_name_channel dictionary should include key-value pairs as:
        key: intensity or background , value: Path_to_file
        Currently, has to be used for each channel individually.
        '''

        for key, value in type_file_dict.items():
            if type(value) is Path:
                if "intensity" in key:
                    intensity_map_name = value.parent / Path(value.stem + "_intensity_map.tif")
                    channel_map = value.stem.split('_')[-1] + "_intensity_map.tif"
                    self.stacks[channel_map] = self._gen_intensity_correction_map(imread(value))
                    tifffile.imsave(intensity_map_name, self.stacks[channel_map].astype(np.float16))
                    self.paths[channel_map] = intensity_map_name
                    print(f"Intensity map saved in the data dir. as {intensity_map_name}")

                elif "background" in key:
                    background_map_name = value.parent / Path(value.stem + "_background_map.tif")
                    channel_map = value.stem.split('_')[-1] + "_background_map.tif"
                    self.stacks[channel_map] = self._gen_background_correction_map(imread(value))
                    tifffile.imsave(background_map_name, self.stacks[channel_map].astype(np.int16))
                    self.paths[channel_map] = background_map_name
                    print(f"Background map saved in the data dir. as {background_map_name}")
            
            else:
                print(f"Wrongly formatted dictionary; try again!")

            # match key:

            #     case "intensity":
            #         intensity_map_name = value.parent / Path(value.stem + "_intensity_map.tif")
            #         channel_map = value.stem.split('_')[-1] + "_intensity_map.tif"
            #         self.stacks[channel_map] = self._gen_intensity_correction_map(imread(value))
            #         tifffile.imsave(intensity_map_name, self.stacks[channel_map].astype(np.float16))
            #         self.paths[channel_map] = intensity_map_name
            #         print(f"Intensity map saved in the data dir. as {intensity_map_name}")
                
            #     case "background":
            #         background_map_name = value.parent / Path(value.stem + "_background_map.tif")
            #         channel_map = value.stem.split('_')[-1] + "_background_map.tif"
            #         self.stacks[channel_map] = self._gen_background_correction_map(imread(value))
            #         tifffile.imsave(background_map_name, self.stacks[channel_map].astype(np.int16))
            #         self.paths[channel_map] = background_map_name
            #         print(f"Background map saved in the data dir. as {background_map_name}")

        return



    def track_centroids(self, saveflag: False, memory = None, max_pixel_movement = None) -> pd.DataFrame:
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

        self.tracked = tp.link(track_table, 
                               self.defaults.max_pixel_movement, 
                               memory=memory)
        
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
        
        if saveflag:
            self.tracked.to_excel(self.cellaap_dir / Path(self.name_stub+"_tracks.xlsx"))

        return self.tracked
    
    def _display_tracks(self):
        '''
        '''
        self.viewer = napari.Viewer()
        self.stacks["phase"] = imread(self.paths["phase"])
        phase_binned = np.zeros_like(self.stacks["instance"], dtype=int)
        for i in np.arange(phase_binned.shape[0]):
            phase_binned[i,:,:]=block_reduce(self.stacks["phase"][i,:,:,],block_size=(2,2), func=np.max)

        self.viewer.add_image(phase_binned)
        self.viewer.add_labels(self.stacks["semantic"])
        self.viewer.add_tracks(self.tracked[["particle","frame","x","y"]].to_numpy())

        return self.viewer

    def _mean_signal_from_mask(self, img: npt.NDArray, mask: npt.NDArray):
        '''

        '''
        pixels = img[np.nonzero(mask)]
        if pixels.any():
            mean_signal = np.mean(pixels)
        else:
            mean_signal = np.nan

        return mean_signal
    
    def measure_signal(self, channel: str, save_flag: False, id = -1,):
        '''

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
            if semantic[0] == 1 & semantic[-1] ==1:
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

                    signal[counter] = self._mean_signal_from_mask(channel_stack[f,:,:], mask)
                    
                    if self.background_map_present:
                        map_name = channel + '_background_map'
                        background_correction[counter] = self._mean_signal_from_mask(self.stacks[map_name][f,:,:].astype(float), mask)
                    
                    if self.intensity_map_present:
                        map_name = channel + '_intensity_map'
                        intensity_correction[counter] = self._mean_signal_from_mask(self.stacks[map_name], mask)

                    counter = counter + 1
                
                self.tracked.loc[index, channel] = signal
                self.tracked.loc[index, channel+"_bkg_corr"] = background_correction
                self.tracked.loc[index, channel+"_int_corr"] = intensity_correction

            else:
                print(f"cell #{id} not processed; in mitosis at start or end")

        if save_flag:
            with pd.ExcelWriter(self.cellaap_dir / Path(self.name_stub+'_analysis.xlsx')) as writer:  
                self.tracked.to_excel(writer, sheet_name='cell_data')
                # There are no scalars - so turn into list; transform
                pd.DataFrame([self.paths]).T.to_excel(writer,   sheet_name='file_data')
            # self.tracked.to_excel()

        return self.tracked
    
    def _calculate_signal(self, semantic, signal, bkg_corr, int_corr):
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
    
    def summarize_data(self, save_flag: True):
        '''
        Currently applying conservative filters to focus only on mitotic
        cells with the expected pattern of inferred labels.
        '''
        # Select only those tracks where mitosis was observed
        idlist    = list(set(self.tracked[self.tracked.mitotic==1].particle))
        mitosis   = []
        mito_start= []
        cell_area = []
        particle  = []
        # Check which channels have been measured. If none, return only "mitotic duration"
        # Need to find a better way to code this.
        GFP_exists = False
        Texas_Red_exists = False
        Cy5_exists = False
        if "GFP" in self.tracked.columns:
            GFP_exists = True
            GFP = []
            GFP_bkg_corr = []
            GFP_int_corr = []
        if "Texas_Red" in self.tracked.columns:
            Texas_Red_exists = True
            Texas_Red = []
            Texas_Red_bkg_corr = []
            Texas_Red_int_corr = []
        if "Cy5" in self.tracked.columns:
            Cy5_exists = True
            Cy5 = []
            Cy5_bkg_corr = []
            Cy5_int_corr = []

        for id in idlist:
            semantic = self.tracked[self.tracked.particle==id].semantic
            _, props = find_peaks(semantic, width=self.defaults.min_mitotic_duration)
            
            # Only select tracks that have one peak in the semantic trace
            
            if props["widths"].size == 1:
                mitosis.append(props["widths"][0])
                mito_start.append(props['left_bases'][0])
                cell_area.append(self.tracked[self.tracked.particle==id].area.mean())
                particle.append(id)
                
                if Texas_Red_exists:
                    signal, bkg_corr, int_corr = self._calculate_signal(semantic, 
                                                                        self.tracked[self.tracked.particle==id].Texas_Red.to_numpy(), 
                                                                        self.tracked[self.tracked.particle==id].Texas_Red_bkg_corr.to_numpy(), 
                                                                        self.tracked[self.tracked.particle==id].Texas_Red_int_corr.to_numpy())
                    Texas_Red.append(signal)
                    Texas_Red_bkg_corr.append(bkg_corr)
                    Texas_Red_int_corr.append(int_corr)
                
                if GFP_exists:
                    signal, bkg_corr, int_corr = self._calculate_signal(semantic, 
                                                                        self.tracked[self.tracked.particle==id].GFP.to_numpy(), 
                                                                        self.tracked[self.tracked.particle==id].GFP_bkg_corr.to_numpy(), 
                                                                        self.tracked[self.tracked.particle==id].GFP_int_corr.to_numpy())
                    GFP.append(signal)
                    GFP_bkg_corr.append(bkg_corr)
                    GFP_int_corr.append(int_corr)
                
                if Cy5_exists:
                    signal, bkg_corr, int_corr = self._calculate_signal(semantic, 
                                                                        self.tracked[self.tracked.particle==id].Cy5.to_numpy(), 
                                                                        self.tracked[self.tracked.particle==id].Cy5_bkg_corr.to_numpy(), 
                                                                        self.tracked[self.tracked.particle==id].Cy5_int_cor.to_numpy())
                    Cy5.append(signal)
                    Cy5_bkg_corr.append(bkg_corr)
                    Cy5_int_corr.append(int_corr)
        
        # Construct summary DF
        self.summaryDF = pd.DataFrame({"particle"  : particle,
                                       "cell_area" : cell_area,
                                       "mito_start": mito_start,
                                       "mitosis"   : mitosis,})
        if GFP_exists:
            self.summaryDF["GFP"] = GFP
            self.summaryDF["GFP_bkg_corr"] = GFP_bkg_corr
            self.summaryDF["GFP_int_corr"] = GFP_int_corr
        if Texas_Red_exists:
            self.summaryDF["Texas_Red"] = Texas_Red
            self.summaryDF["Texas_Red_corr"] = Texas_Red_bkg_corr
            self.summaryDF["Texas_Red_int_corr"] = Texas_Red_int_corr
        if Cy5_exists:
            self.summaryDF["Cy5"] = Cy5
            self.summaryDF["Cy5_bkg_corr"] = Cy5_bkg_corr
            self.summaryDF["Cy5_int_corr"] = Cy5_int_corr

        if save_flag:
            self.summaryDF.to_excel(self.cellaap_dir / Path(self.name_stub+"_summary.xlsx"))

        return self.summaryDF
    
    
    def gather_plot_summaries(self, well_position: list) -> pd.DataFrame:
        '''
        Inputs:
        well_position : list with entries of the form r[A-Z][\d]+_+[a-z][\d+]
        '''
        if well_position:
            if not hasattr(self, 'inf_folder_list'):
                # Assemble the folder list when function called for the first time
                self.inf_folder_list = [f for f in self.root_folder.glob('*_inference')]
            
            # selected_dirs = []
            df_list = []
            for wp in well_position:
                for f in self.inf_folder_list:
                    if wp in f.name:
                        xls_file_name = [file for file in f.glob('*_summary.xlsx')] 
                        df = pd.read_excel(xls_file_name[0]) #assumes only one
                        df["well_pos"] = wp #assign well-position identifier
                        df_list.append(df)

            data_summary = pd.concat(df_list)

        return data_summary