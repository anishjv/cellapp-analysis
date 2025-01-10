# cellapp-analysis module
A python module to track cells and measure fluorescence and mitotic duration using cellapp inference files and raw fluorescence images. It runs trackpy on the instance segmentation file first, and then measures cell state (mitotic/non-mitotic) from the semantic segmentation and raw mean fluorescence value from the specific fluorescence channel. The module also summarizes the analysis by calculating the mean fluorescence for each cell ove4r the duration of mitosis, the total mitotic duration, and correction factors based on intensity and background correction maps (must be acquired using empty wells with DMEM and fluorobrite respectively).
This analysis and summary are saved as separate excel spreadsheets.

## Usage
1. Specify the root folder (this folder must contain cellapp-generated inference folders and the raw intensity stacks.) This must be input as a Path.

```python
experiment1 = cellapp_analysis.analysis(Path(root_folder), analysis_only: False)
```
If the boolean input is set true, the experiment1 object will look for and read in correction maps (if they are present; not used otherwise)

2. **Analysis mode**: One can create multiple objects corresponding to multiple repeats of an experiment.
e.g., 

```python
experiment1 = cellapp_analysis.analysis(Path(root_folder_1), analysis_only: True)

experiment2 = cellapp_analysis.analysis(Path(root_folder_2), analysis_only: True)
```

In this mode, the module is used to compile all data corresponding to positions and wells belonging to the same treatment/cell line into one dataframe. e.g.,

```python
HeLa_wells = ["A02_s1", "H12_s5"] # Note the exact formant (\W\d+_s\d)
HeLa_data = experiment1.gather_plot_summaries(HeLa_wells)  
```

3. **Measurement mode**: Set the analysis_only mode to False. With this, analysis object instatiation will detect image stacks with either "channel_background" and "channel_intensity" in the filename. 

```python
exp_analysis = cellaap_analysis.analysis(Path(root_folder), False)
```

If present, these stacks will be loaded. The analysis object can also create the maps by reading the images stacks from a user-provided dictionary. The dictionary must have "channel_intensity" and "channel_background" as the keys and Path objects pointing to the corresponding image stacks (DMEM and Fluorobrite respectively) that are present in the root folder. Multiple channels can be specified as long as the "intensity" and "background" keywords are in the dictionary keys. e.g.,

```python
map_dict = {'GFP_background': Path(GFP_fluorobrite_stk),
            'GFP_intensity' : Path(GFP_DMEM_stk), 
            'TR_background': Path(TRed_fluorobrite_stk),
            'TR_intensity':Path(TRed_DMEM_stk)}

exp_analysis.create_correction_maps(map_dict)
```

These correction maps are optional; analysis will progress without them.

Step 1: Point the analysis object to a specific inference folder by providing a path to it. This will read the instance segmentation file from this folder. 

Step 2: Use the **track_centroids** method; it will erode the instance segmentation with the default footprint (needs to be customized for different cells), track the resultant masks using trackpy, and then determine the cell-state by reading the semantic stack. The intermediate padas dataframe can be saved to excel using the flag. Currently, the 'HeLa' input does not do anything; the plan is to use cell-line-specific parameters for trackpy (e.g., when some cells crawl around)

Step 3: Use the **measure_signal** function to measure the fluorescence from the specified channel. The channel string must match the channel name in the file names. The "id = -1" will make the function measure data for all cells that went through a complete mitosis during the time lapse. Optionally, one can provide a list with cell numbers (development only). Thus, signal in cells that remained in interphase is not measured. Any cell that is in mitosis at the beginning or end of the movie is filtered out. Additionally, any cells that show that show multiple peaks in the semantic label timeseries are also filtered out. For these cells, cellapp has not generated high confidence inference. This filter likely biases the data towards cells with shorted mitotic duration.

Step 4: Use the **summarize_data** function to create the summary Excel file that lists the average signals measured for all channels, duraion of mitosis, and the correction factors to account for background and excitation intensity variation.

Step 5: Use the **gather_plot_summaries** function to collect multiple wells and/or positions that represent the same experiment. The function requires a list as the input. Each entry in the list must be a string encoding the well and position identifier. Notice the capitalization and well number convention used in the example below. The output is a dataframe with an additional column for the well+position designation. In the future, one more column indicating the experiment will be added.

Step 6: Use the **fit_model** function to fit a 4-parameter sigmoid to binned data. The function expects input data as a dataframe with the first column containing the fluorescence signal and the second column containing the time in mitosis. For this model to work, the 0 dosage response must be defined as a positive value. This value must be obtained from a -rapamycin well or otherwise supplied. If it is unavailable, perform a rough background subtraction as shown below on a temporary basis.
quant_fraction must a list that specifies the quantiles to be evaluated for the dosage. The bin range is based on the quantile values of the dosage values. Remember that the eSAC dosage distribution is asymmetric (it should be possible to fit it with a log-normal distribution). Therefore, the default quantile values (used below) are asymmetric. bin_size is arbitrarily defined and can also be adjusted if necessary. Don't use lower values (the default shown below is empirically defined).

```python
exp_analysis.files(Path(to_inference_folder), cell_type = "HeLa")
exp_analysis.track_centroids(save_flag = False)
tracks = exp_analysis.measure_signal('GFP', save_flag = False, id = -1)
tracks = exp_analysis.measure_signal('Texas_Red', save_flag = True, id = -1)
tracks = exp_analysis.measure_signal('Cy5', save_flag = True, id = -1) #if needed
summary = exp_analysis.summarize_data(True)
well_pos_list = ["A01_s1", "A12_s5", "B05_s3"]
compiled_data = exp_analysis.gather_plot_summaries(well_pos_list)

# plotting and curve-fitting compiled data
dose_response = compiled_data.loc[:, ('Texas_Red', 'mitosis')]
# rough background subtraction 
dose_response.Texas_Red = dose_response.Texas_Red - dose_response.Texas_Red.min()
xy_data, bin_means, fit_values = fit_model(dose_response, plot: True, quant_fraction = [0.025, 0.85], bin_size = 2.5)
```
