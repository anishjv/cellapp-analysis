# cellapp-analysis module
A python module to track cells and measure fluorescence and mitotic duration using cellapp inference files and raw fluorescence images. It runs trackpy on the instance segmentation file first, and then measures cell state (mitotic/non-mitotic) from the semantic segmentation and raw mean fluorescence value from the specific fluorescence channel. The module also summarizes the analysis by calculating the mean fluorescence for each cell ove4r the duration of mitosis, the total mitotic duration, and correction factors based on intensity and background correction maps (must be acquired using empty wells with DMEM and fluorobrite respectively).
This analysis and summary are saved as separate excel spreadsheets.

# Usage
1. Specify the root folder (this folder must contain cellapp-generated inference folders and the raw intensity stacks.) This must be input as a Path.
experiment1 = cellapp_analysis.analysis(Path(root_folder), analysis_only: False)
If the boolean input is set true, the experiment1 object will look for and read in correction maps (if they are present; not used otherwise)

2. **Analysis_only mode**: One can create multiple objects corresponding to multiple repeats of an experiment.
e.g., 

experiment1 = cellapp_analysis.analysis(Path(root_folder_1), analysis_only: False)
experiment2 = cellapp_analysis.analysis(Path(root_folder_2), analysis_only: False)

In this mode, the module is used to compile all data corresponding to positions and wells belonging to the same treatment/cell line into one dataframe. e.g.,

HeLa_wells = ["A02_s1", "H12_s5"] # Note the exact formant (\W\d+_s\d)
HeLa_data = experiment1.gather_plot_summaries(HeLa_wells)  

3. 
