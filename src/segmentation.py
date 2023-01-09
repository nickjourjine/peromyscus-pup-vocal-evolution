# this file contains functions for segmenting vocalizations

import glob
import os
import json
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp2d
from scipy.signal import stft
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import shutil	
import seaborn as sns 
from scipy.ndimage.filters import gaussian_filter

#definitely keep - documented now on evernote 20230105
def get_amplitude_segments(audio_dir, save_dir, seg_params, species = None, thresholds_path=None, intersyll_threshold = None, duration_threshold = None, path_list = None):
    """
    Segment audio files using get_onsets_offsets() from AVA. Writes a separate csv for non-vocal pups.

    Parameters
    ----------
    audio_dir (string): the path to the directory containing the raw wav files to be segmented

    save_dir (string): the path to the directory where the start/stop csvs will be written (one for each recording)

    species (string): optionally segment just one species, indicated by the two letter code (which should begin the name of each raw wav file)

    thresholds_path (string): a path to a csv containing a noise floor for each file in one column and the file name in another. Optional.

    intersyll_threshold (float): detected syllables separated by less than this number of seconds will be merged into one vocalization

    duration_threshold (float): detected syllables shorter than this value will be ignored

    seg_params (dict): the segmenting parameters

    path_list (list): a list of paths to the audio to analyze (optionally use this if you don't want to segment every vocalization in the directory)

    Returns
    -------
    None

    """
    all_wavs = [i for i in os.listdir(audio_dir) if i.endswith('.wav')]
    done = [i for i in os.listdir(save_dir) if i.endswith('.csv')]

    #just take a list of paths as input if you want
    if path_list != None and audio_dir == None:
        wav_files = path_list
        not_done = [i for i in wav_files if i.split('/')[-1] not in done]
        print('Segmenting the audio in the list you provided...')

    #subset by species if you want
    elif species != None:
        wav_files = [audio_dir+i for i in os.listdir(audio_dir) if i.endswith('.wav') and not i.startswith('.') and i.startswith(species)]
        not_done = [i for i in wav_files if i.split('/')[-1] not in done]
        print('Segmenting species', species)

    #or just do everything
    else:
        wav_files = [audio_dir+i for i in os.listdir(audio_dir) if i.endswith('.wav') and not i.startswith('.')]
        not_done = [i for i in wav_files if i.split('/')[-1] not in done]
        print('Segmenting everything...')

    #get the noise thresholds file if you have one
    if thresholds_path != None:
        thresholds = pd.read_csv(thresholds_path)
        print('Segmenting WITH per recording noise floors...')
        print(len(done), 'recordings have been segmented out of', len(all_wavs))
    else:
        print('Segmenting WITHOUT per recording noise floors...')
        print(len(done), 'recordings have been segmented out of', len(all_wavs))

    counter=0
    total_vocs_found = []
    for file in wav_files:

        print(file)
        if file.split('/')[-1].split('.wav')[0]+'.csv' in os.listdir(save_dir):
            print('DONE WITH...', file.split('/')[-1].split('.wav')[0])
            continue

        else:
            print('PROCESSING...', file)

            #if there is a thresholds file get the noise threshold for this pup and update the segmenting parameters dictionary with it
            if thresholds_path != None:

                if not 'clip' in file:
                    pup = file.split('/')[-1]
                elif 'clip' in file:
                    pup = file.split('/')[-1].split('_clip')[0]+'.wav'
                    print(pup)


                thresh = np.float(thresholds['noise_floor'].loc[thresholds['source_file'] == pup])
                seg_params['spec_min_val'] = thresh
                print('reset spec_min_val to', thresh)

            _, audio = wavfile.read(file)

            print('SEGMENTING...', file.split('/')[-1])
            print('spec_min_val is', seg_params['spec_min_val'])
            onsets, offsets, _, _ = get_onsets_offsets(audio=audio, p=seg_params)


            if len(onsets) != 0:
                temp = pd.DataFrame()
                temp['start_seconds'] = onsets
                temp['stop_seconds'] = offsets
                temp['source_file'] = [file]*len(onsets)
                temp = temp.drop_duplicates()
                temp = prune_segments(temp, 
                               intersyll_threshold = intersyll_threshold, 
                               duration_threshold = duration_threshold)
                csv_save_name = os.path.join(save_dir,file.split('/')[-1][:-4]+'.csv')
                temp.to_csv(csv_save_name, index=False)
                print('...FOUND', len(temp), 'vocalizations')
                counter += 1
                total_vocs_found.append(len(temp))

            else:
                print('...no vocalizations found')
                nonvocal = pd.DataFrame()
                nonvocal['start_seconds'] = None
                nonvocal['stop_seconds'] = None
                nonvocal['source_file'] = file
                nonvocal.to_csv(save_dir+file.split('/')[-1][:-4]+'.csv', index=False)

    print('segmented', counter, 'files')
    print('#######################')
    print('total segments detected:', sum(total_vocs_found))
    print('#######################')
    print('done.')








    
    
    
    
def get_onsets_offsets(audio, p):
	"""
	Segment the spectrogram using thresholds on its amplitude. From https://autoencoded-vocal-analysis.readthedocs.io/en/latest/_modules/ava/segmenting/amplitude_segmentation.html#get_onsets_offsets

	A syllable is detected if the amplitude trace exceeds `p['th_3']`. An offset
	is then detected if there is a subsequent local minimum in the amplitude
	trace with amplitude less than `p['th_2']`, or when the amplitude drops
	below `p['th_1']`, whichever comes first. Syllable onset is determined
	analogously.

	Note
	----
	`p['th_1'] <= p['th_2'] <= p['th_3']`

	Parameters
	----------
	audio : numpy.ndarray
		Raw audio samples.
	p : dict
		Parameters.
	return_traces : bool, optional
		Whether to return traces. Defaults to `False`.

	Returns
	-------
	onsets : numpy array
		Onset times, in seconds
	offsets : numpy array
		Offset times, in seconds
	traces : list of a single numpy array
		The amplitude trace used in segmenting decisions. Returned if
		`return_traces` is `True`.
	amplitudes: a list of spectrogram amplitudes
	"""

	if len(audio) < p['nperseg']:
		if return_traces:
			return [], [], None
		return [], []
	spec, dt, _ = get_spec(audio, p)
	min_syll_len = int(np.floor(p['min_dur'] / dt))
	max_syll_len = int(np.ceil(p['max_dur'] / dt))
	th_1, th_2, th_3 = p['th_1'], p['th_2'], p['th_3'] # thresholds
	onsets, offsets = [], []
	too_short, too_long = 0, 0

	# Calculate amplitude and smooth.
	amps = np.sum(spec, axis=0)
	amps = gaussian_filter(amps, p['smoothing_timescale']/dt)

	# Find local maxima greater than th_3.
	local_maxima = []
	print('getting maxima')
	for i in range(1,len(amps)-1,1):
		if amps[i] > th_3 and amps[i] == np.max(amps[i-1:i+2]):
			local_maxima.append(i)

	# Then search to the left and right for onsets and offsets.
	for local_max in local_maxima:
		if len(offsets) > 1 and local_max < offsets[-1]:
			continue
		i = local_max - 1
		while i > 0:
			if amps[i] < th_1:
				onsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				onsets.append(i)
				break
			i -= 1
		if len(onsets) != len(offsets) + 1:
			onsets = onsets[:len(offsets)]
			continue
		i = local_max + 1
		while i < len(amps):
			if amps[i] < th_1:
				offsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				offsets.append(i)
				break
			i += 1
		if len(onsets) != len(offsets):
			onsets = onsets[:len(offsets)]
			continue
	
	# Throw away syllables that are too long or too short.
	new_onsets = []
	new_offsets = []
	for i in range(len(offsets)):
		t1, t2 = onsets[i], offsets[i]
		new_onsets.append(t1 * dt)
		new_offsets.append(t2 * dt)

	# Return decisions.
	return new_onsets, new_offsets, [amps], spec


#modified from https://autoencoded-vocal-analysis.readthedocs.io/en/latest/_modules/ava/segmenting/utils.html?highlight=get_spec

def get_background_clips(raw_wavs_dir, save_location, all_segments_df, margin, start_column, stop_column, label_column = None, species = None, units = 's'):
	"""
	Use amplitude segmentation to generate wav clips of iter-vocalization audio. 
	Similar to get_wav_clips except get the non-vocalization audio clips.

	Parameters
	----------
	
	raw_wavs_dir (string): the path to the directory containing the raw wav files.

	save_location (string): the path to the directory where the clips should be saved.

	all_segments_df (dataframe): the dataframe containing the segments infor for the entire data set - should contain at least the columns ['source_file', 'start_times', 'stop_times']

	margin (float): a margin (in seconds) to be added before each start time and after each stop time

	start_column (string): the name of the column in source_data that contains the vocalization start times

	end_column (string): the name of the column in source_data that contains the vocalization stop times

	label_column (string, optional): the name of the column in source_data that contains labels for each vocalization 

	species (string): optional 2 letter code to process just one species' raw wav files

	units (string): the temporal units for start and stop times (s or ms)

	Returns
	-------
	None.

	"""

	#get the data and optionally subset by species 
	df = pd.read_csv(all_segments_df)
	if species != None:
		df = df.loc[df['species'] == species]

	#get the names of the recording source files 
	source_files = df['source_file'].unique()

	#check which files have already been segmented
	already_processed = [i.split('_backgroundclip')[0] for i in os.listdir(save_location)]
	
	for file in source_files:

		#get the start and stop times for this recording's vocalization segmentation
		sf_df = df.loc[df['source_file'] == file]
		
		#make a dataframe for the start and stop times of the intervocalization periods
		bg_df = pd.DataFrame()
		bg_df['background_start_time'] = [float(i) for i in sf_df[stop_column][:-1]] #ignore background before first vocalization 
		bg_df['background_stop_time'] = [float(i) for i in sf_df[start_column][1:]] #and after last vocalization
		bg_df['source_file'] = [i for i in sf_df['source_file'][1:] ] #so background segments df is one row shorter than vocalization segments df
		bg_df['species'] = [i for i in sf_df['species'][1:] ]
		bg_df['duration'] = bg_df['background_stop_time'] - bg_df['background_start_time']
		
		num_to_process = len(bg_df)
		num_already_processed = len([i for i in already_processed if file in i])
		
		if file in already_processed and num_to_process==num_already_processed:
			print(file, 'already processed, skipping...')

		else:
			path_to_source = raw_wavs_dir + file  
			fs, wav = wavfile.read(path_to_source)
			bg_df['background_clip_number'] = range(num_to_process)

			count = 0
			print('preparing to get', len(bg_df), 'non-vocalization clips from', file.split('/')[-1])
			for idx, _ in bg_df.iterrows(): 
			
				#get the start and stop time
				start, end = bg_df.loc[idx, ('background_start_time')], bg_df.loc[idx, ('background_stop_time')] #get the start and stop time for the clip
				
				#name the clip
				clip_name = bg_df.loc[idx, 'source_file'].split('.wav')[0] + '_' + 'backgroundclip' + '_' + str(bg_df.loc[idx, 'background_clip_number']) + '.wav' #name the clip  
	
				#clip it out and write it
				if units == 's':
					start= int((start - margin)*fs)
					end =  int((end + margin)*fs)
					clip = wav[start:end] #get the clip
					wavfile.write(save_location + clip_name, fs, clip) #write the clip to a wav
					count+=1

				elif units == 'ms':
					start, end = start - margin, end + margin
					start, end = int((start/1000)*fs), int((end/1000)*fs) #convert to sampling units
					clip = wav[start:end] #get the clip
					wavfile.write(save_location + clip_name, fs, clip) #write the clip to a wav
					count+=1
	
			print(' ...got', num_to_process,'wav clips')
	print('done.')
  
    
    
def get_wav_clips(wavs_dir, save_location, source_data, margin, start_column, end_column, label_column = None, species = None, units = 's'):
	"""
	Use start and stop times of vocalizations to save individual clips as .wav files (one per detected voc)

	Parameters
	----------
	wavs_dir (string): the path to the raw wav files that have already been segmented

	save_location (string): the path to the directory where the clips should be saved

	source_data (dataframe): should contain at least the columns ['source_file', 'start_times', 'stop_times']

	margin (float): a margin (in seconds) to be added before each start time and after each stop time

	start_column (string): the name of the column in source_data that contains the vocalization start times

	end_column (string): the name of the column in source_data that contains the vocalization stop times

	label_column (string): the name of the column in source_data that contains labels for each vocalization (optional)

	species (string): optional 2 letter code to process just one species' raw wav files

	units (string): the temporal units for start and stop times (s or ms)

	Returns
	-------
	None.

	"""

	#optionally subset by species 
	if species != None:
		if 'species' not in source_data.columns:
			source_data['species'] = [i.split('_')[0] for i in source_data['source_file']]
		df = source_data.loc[source_data['species'] == species]

	#get the names of the recording source files 
	source_files = df['source_file'].unique()

	#for each recording in df, load the wav, subset the big data frame to get just the start and stop times for this recording, then 
	#for each start and stop time (for each clip), get the clip, name it, and write it to save_location. Note that time is assumed
	#to be in ms here.

	already_processed = [i.split('_clip')[0] for i in os.listdir(save_location)]

	for file in source_files:
		sf_df = df.loc[df['source_file'] == file]
		num_vocs_to_process = len(sf_df)
		num_already_processed = len([i for i in already_processed if file.split('.')[0] in i])

		if file.split('.')[0] in already_processed and num_vocs_to_process==num_already_processed:
			print('all segments from',file, 'already processed, skipping...')

		else:
			path_to_source = wavs_dir + file  
			fs, wav = wavfile.read(path_to_source)
			sf_df['clip_number'] = range(num_vocs_to_process)
			count = 0
			print('preparing to get', len(sf_df), 'clips from', file.split('/')[-1])
			for idx, _ in sf_df.iterrows(): 
				start, end = sf_df.loc[idx, (start_column)], sf_df.loc[idx, (end_column)] #get the start and stop time for the clip
	
				if label_column != None:
					clip_name = ('_').join([sf_df.loc[idx, 'source_file'].split('.wav')[0],'clip',str(sf_df.loc[idx, 'clip_number']),sf_df.loc[idx, label_column]]) + '.wav'   

				else:
					clip_name = ('_').join([sf_df.loc[idx, 'source_file'].split('.wav')[0],'clip',str(sf_df.loc[idx, 'clip_number'])])+'.wav' 
	
				if units == 's':
					start= int((start - margin)*fs)
					end =  int((end + margin)*fs)
					clip = wav[start:end] #get the clip
					wavfile.write(os.path.join(save_location,clip_name), fs, clip) #write the clip to a wav
					count+=1

				elif units == 'ms':
					start, end = start - margin, end + margin
					start, end = int((start/1000)*fs), int((end/1000)*fs) #convert to sampling units
					clip = wav[start:end] #get the clip
		
					wavfile.write(os.path.join(save_location,clip_name), fs, clip) #write the clip to a wav
					count+=1
	
			print(' ...got', num_vocs_to_process,'wav clips')
	print('done.')
	
    
    
    
    
    

def get_intersyllable_intervals(df, annotation=False):
	"""
	Get intersyllable inervals, which is used to determine which vocalizations should be merged and which should not

	Parameters
	----------
	df (dataframe): a dataframe of vocalization start and stop predictions (ie the result of segment wavs)
	
	annotation (bool): if True, include labels for each segment

	Returns
	-------
	all_intersyllables_df (dataframe): a dataframe where each row is a vocalization and the column is the intersyllabl 
									   interval between the end of that vocalization and the next

	"""

	#get the starts and stops
	starts = list(df['start_seconds'])
	stops = list(df['stop_seconds'])
	start_or_stop = ['start']*len(starts)+['stop']*len(stops)
	
	#if you are pruning an annotation get the labels - TODO make this work
	if annotation==True:
		labels = list(test_df['name'])
		names = [i for i in [[i]*2 for i in test_df['name']]]
		names = [item for sublist in names for item in sublist]

	#sort them so they are alternating
	intersyllable = pd.DataFrame()
	intersyllable['times'] = starts + stops
	intersyllable['start_or_stop'] = start_or_stop
	if annotation==True:
		intersyllable['name'] = names
		
	#Occasionally one predicted segment will end at exactly the same time as the next one starts for some reason, which causes problems for sorting. Drop these before sorting.
	intersyllable = intersyllable.drop_duplicates(subset=['times'], keep =False) 
	intersyllable = intersyllable.sort_values(by='times').reset_index(drop=True)

	#get the difference between consecutive rows - these are either intersyllable intervals or durations (values in the same row as a start time are ISIs)
	intersyllable['diff'] = intersyllable['times'].diff()
	just_intersyllable = list(intersyllable['times'].diff())[::2] # this is the just the intersyllable intervals)
	
	return intersyllable, just_intersyllable
	
def prune_segments(predictions_df, intersyll_threshold,duration_threshold,annotation=False):
	"""
	Merge vocalizations with an intersyllable interval below a threshold (ms) from each other and drop vocalizations 
	that are shorter than a threshold (ms)

	Parameters
	----------
	predictions_df (dataframe): a dataframe of vocalization start and stop predictions (ie the result of segment wavs)

	intersyll_threshold (float): the time in seconds below which two vocalizations will be merged into one

	duration_threshold (float): the duration in seconds below which predicted vocalizations will be ignored
	
	annotation (bool): if True, include a column of labels

	Returns
	-------
	all_new_segments (datframe): predictions_df with the mergers and deletions carried out.

	"""

	all_intersyllables = []
	files = []
	dfs = []
	source_files = [i for i in predictions_df['source_file'].unique()]

	print('...merging vocalizations separated by less than', intersyll_threshold, 'seconds..')
	for file in source_files: 

		#get the predictions
		test_df = predictions_df.loc[predictions_df['source_file'] == file]

		#get the intersyllable intervals
		intersyllable, _ = get_intersyllable_intervals(test_df, annotation=annotation)
		
		#get short intersyllables to drop - these are the start times where diff is less than the threshold (diff is the time from the previous start or stop)
		diffs = list(intersyllable['diff'])
		short_intersyll_starts = [i for i in intersyllable.index if intersyllable['diff'].iloc[i] < intersyll_threshold and intersyllable['start_or_stop'].iloc[i] == 'start'] 
		short_intersyll_stops = [i-1 for i in short_intersyll_starts]

		#drop the start times with short preceding intersyllable intervals and the stop time that immediately preceds those starts. This is the "merging" step.
		to_drop = sorted(short_intersyll_starts+short_intersyll_stops)
		pruned_intersyllable = intersyllable.drop(index=to_drop)
		
		#make a new segments_file
		new_segments = pd.DataFrame()
		new_segments['start_seconds'] = list(pruned_intersyllable['times'].loc[pruned_intersyllable['start_or_stop'] == 'start'])
		new_segments['stop_seconds'] = list(pruned_intersyllable['times'].loc[pruned_intersyllable['start_or_stop'] == 'stop'])
		new_segments['source_file'] = [file for i in range(len(new_segments))]
		new_segments['duration'] = new_segments['stop_seconds'] - new_segments['start_seconds']
		new_segments = new_segments.loc[new_segments['duration'] > duration_threshold]
		
		#TODO: get this to work
		#if annotation==True:
		#	new_segments['name'] = list(pruned_intersyllable['name'].loc[pruned_intersyllable['start_or_stop'] == 'start'])
		
		dfs.append(new_segments)

	#drop short
	all_new_segments = pd.concat(dfs)
	print('...dropping vocalizations shorter than', duration_threshold, 'seconds..')
	all_new_segments['duration'] = all_new_segments['stop_seconds'] - all_new_segments['start_seconds']
	all_new_segments = all_new_segments.loc[all_new_segments['duration'] > duration_threshold]

	return all_new_segments

    
    
#probably delete
def prune_segments_testing(predictions_df_path, intersyll_threshold,duration_threshold, save_dir):
	"""
	version of prune_segments() that takes a path instead of a dataframe and doesn't return 
	anything (just saves a csv). Useful for testing on annotated recordings.

	Parameters
	----------
	predictions_df_path (dataframe): path to a dataframe of vocalization start and stop predictions (ie the result of segment wavs)

	intersyll_threshold (float): the time in seconds below which two vocalizations will be merged into one

	duration_threshold (float): the duration in seconds below which predicted vocalizations will be ignored

	save_dir (string): path to the directory to save the csv of pruned segments

	Returns
	-------
	all_new_segments (datframe): predictions_df with the mergers and deletions carried out.

	"""

	predictions_df = pd.read_csv(predictions_df_path)

	all_intersyllables = []
	files = []
	dfs = []
	source_files = [i for i in predictions_df['source_file'].unique()]

	print('merging vocalizations separated by less than', intersyll_threshold, 'milliseonds..')
	for file in source_files:   
		#get the predictions
		test_df = predictions_df.loc[predictions_df['source_file'] == file]

		#get the starts and stops
		starts = list(test_df['start_seconds'])
		stops = list(test_df['stop_seconds'])
		start_or_stop = ['start']*len(starts)+['stop']*len(stops)

		#sort them so they are alternating
		intersyllable = pd.DataFrame()
		intersyllable['times'] = starts + stops
		intersyllable['start_or_stop'] = start_or_stop
		intersyllable = intersyllable.sort_values(by='times').reset_index(drop=True)

		#get the difference between consecutive rows - these are the intersyllable intervals
		intersyllable['diff'] = intersyllable['times'].diff()

		#get the rows to drop
		diffs = list(intersyllable['diff'])

		short_ones_starts = [i for i in range(len(diffs)) if diffs[i] < intersyll_threshold]
		short_ones_stops = [i-1 for i in short_ones_starts]
		to_drop = sorted(short_ones_starts +short_ones_stops)

		#drop them
		pruned_intersyllable = intersyllable.drop(index =to_drop)

		#make a new segments_file
		new_segments = pd.DataFrame()
		new_segments['start_seconds'] = list(pruned_intersyllable['times'].loc[pruned_intersyllable['start_or_stop'] == 'start'])
		new_segments['stop_seconds'] = list(pruned_intersyllable['times'].loc[pruned_intersyllable['start_or_stop'] == 'stop'])
		new_segments['source_file'] = [file for i in range(len(new_segments))]
		dfs.append(new_segments)

	#save
	all_new_segments = pd.concat(dfs)
	print('dropping vocalizations shorter than', duration_threshold, 'milliseonds..')
	all_new_segments['duration'] = all_new_segments['stop_seconds'] - all_new_segments['start_seconds']
	all_new_segments = all_new_segments.loc[all_new_segments['duration'] > duration_threshold]
	all_new_segments.to_csv(save_dir+'all_predictions_pruned.csv', index=False)
	print('done.')

#amplitude segmentation modified from https://autoencoded-vocal-analysis.readthedocs.io/en/latest/_modules/ava/segmenting/amplitude_segmentation.html#get_onsets_offsets
def evaluate_predictions(prediction_csv, annotations_csv, annotations_dir, tolerance, verbose=False):
    """
    Evaluate predictions on annotated recordings.

    Parameters
    ----------
    prediction_csv (string): the path to the csv containing the predictions (start and stop times) for annotated audio

    annotations_csv: the path to the csv containing the containing the hand annoataions (start and stop times) for annotated audio

    annotations_dir: the directory containing annotations_csv (TODO: improve this)

    tolerance (float): the tolerance window in seconds before and after an annotated start or stop within which a predicted start or stop will be counted as correct

    verbose (bool): if True, print out progress status on each annotated file

    Returns
    -------
    df (dataframe): a dataframe of the the accuracy, precision and F1 scores for predictions on each annotated audio clip.

    """

    annotations = pd.read_csv(annotations_csv)
    predictions_df = pd.read_csv(prediction_csv)

    if 'voc_ID' in predictions_df.columns:
        predictions_df = predictions_df.loc[(predictions_df['voc_ID'] != 0) & (predictions_df['voc_ID'] != '0')] # remove noise predictions from das models
        
    annotations['source_file'] = [i.split('_anno')[0] for i in annotations['source_file']]
    predictions_df['source_file'] = [i.split('_clip')[0] for i in predictions_df['source_file']]

    source_files = [i for i in predictions_df['source_file'].unique() if i in annotations['source_file'].unique()] #only evaluate the predictions for which you have annotations
    onsets_errors = []
    offsets_errors = []
    all_onsets_errors = []
    all_offsets_errors = []
    counts = []
    true_pos = []
    false_pos = []
    recordings = []
    predicted_counts = []

    for file in source_files:
        if verbose:
            print('evaluating segmentation on...', file)

        true_pos_count = 0
        false_pos_count = 0

        prediction_start_times = predictions_df['start_seconds'].loc[predictions_df['source_file'] == file]
        prediction_stop_times = predictions_df['stop_seconds'].loc[predictions_df['source_file'] == file]

        annotated_start_times = annotations['start_seconds'].loc[annotations['source_file'] == file]
        annotated_stop_times = annotations['stop_seconds'].loc[annotations['source_file'] == file]

        for prediction_start_time, prediction_stop_time in zip(prediction_start_times, prediction_stop_times):

            start_diffs = annotated_start_times - prediction_start_time
            stop_diffs = annotated_stop_times - prediction_stop_time

            onset_error = np.min(np.abs(start_diffs))
            offset_error = np.min(np.abs(stop_diffs))

            if offset_error < tolerance:
                offsets_errors.append([i for i in stop_diffs if np.abs(i) == offset_error][0])

            if onset_error < tolerance:
                onsets_errors.append([i for i in start_diffs if np.abs(i) == onset_error][0])

            if onset_error < tolerance and offset_error < tolerance:
                true_pos_count+=1

            elif onset_error > tolerance or offset_error > tolerance:
                false_pos_count+=1

        true_pos.append(true_pos_count)
        false_pos.append(false_pos_count)
        predicted_counts.append(len(predictions_df.loc[predictions_df['source_file'] == file]))
        recordings.append(file)
        counts.append(len(annotations.loc[annotations['source_file'] == file]))	

    df = pd.DataFrame()
    df['source_file'] = recordings
    df['true_pos'] = [i for i in true_pos]
    df['false_pos'] = [i for i in false_pos]
    df['predicted_count'] = predicted_counts
    df['count'] = counts
    df['false_neg'] = df['count'] - df['true_pos']
    df['precision'] = df['true_pos']/df['predicted_count']
    df['recall'] = df['true_pos']/df['count']
    df['F1'] = 2*(df['precision']*df['recall'])/(df['precision']+df['recall'])
    df['species'] = [i.split('_')[0] for i in df['source_file']]

    return df

def das_predict(model_name, audio_dir, save_dir, segment_minlen,segment_fillgap):

	"""
	Segment using a das model. This is a backup function for troubleshooting. To do real segmentation with das use das_predict.py

	Parameters
	----------
	model_name (string): the full path to the trained das model

	audio_dir (string): the path to the directory containing the raw wav files to be segmented

	save_dir (string): the path to the directory to save the predicted start and stop times

	segment_minlen: the minimum length of a segment in seconds. Segments below this length will be ignored

	segment_fillgap: the minimum inter-segment interval below which two segments will be merged into one

	Returns
	-------
	None

	"""

	import das,	utils

	#res_name is the path to the directory containing the model and network parameters (params) plus the date_time prefix
	res_name = model_name

	#load the model and parameters
	model, params = das.utils.load_model_and_params(res_name)

	#get wav file paths for das predict
	audio_for_predict = [audio_dir+i for i in os.listdir(audio_dir) if not i.startswith	('.') and i.endswith('wav')]

	#load the wav file(s)
	test = audio_for_predict

	for wav in test:
		print(wav.split('/')[-1].split('.')[0]+'.csv')

		if wav.split('/')[-1].split('.')[0]+'.csv' in os.listdir(save_dir):
			print(wav, 'already processed')
	
		else:
			pup = wav.split('/')[-1][:-4]
			print('loading audio from pup', pup)
			samplerate, x = wavfile.read(wav)
			x = np.atleast_2d(x).T
			print('predicting audio...')
			_, segments, _, _ = das.predict.predict(x, 
													model_save_name=res_name,
													verbose=2,
													segment_minlen=segment_minlen,
													segment_fillgap=segment_fillgap, 
													pad=True)
								

			df = pd.DataFrame(columns=['pup', 'onsets_s', 'offsets_s', 'duration_s' 'label'])
			onsets = segments['onsets_seconds']
			offsets = segments['offsets_seconds']
			durations = segments['durations_seconds']
			labels = segments['sequence']
			name = [pup]*len(onsets)
			df['pup'] = name
			df['start_seconds'] = onsets
			df['stop_seconds'] = offsets
			df['duration'] = durations
			df['label'] = ['cry' if i == 1 else 'whistle' for i in labels	]
			df.to_csv(save_dir+pup+'.csv', index=False)
			print('wrote csv...')
	print('done.')

#TO DO: finish writing this function (currently built into get_background_clips)	
#def segment_background(audio_dir, save_dir, seg_params, species = None, thresholds_path=None, path_list = None):
# 	"""
# 	Segment audio files using amplitude thresholding.
# 
# 	Parameters
# 	----------
# 	audio_dir (string): the path to the directory containing the raw wav files to be segmented
# 
# 	seg_dir (string): the path to the directory where the start/stop csvs will be written (one for each recording)
# 
# 	species (string): optionally segment just one species, indicated by the two letter code (which should begin the name of each raw wav file)
# 
# 	thresholds_path (string): a path to a csv containing a noise floor for each file in one column and the file name in another. Optional.
# 
# 	intersyll_threshold (float): detected syllables separated by less than this number of seconds will be merged into one vocalization
# 
# 	duration_threshold (float): detected syllables shorter than this value will be ignored
# 
# 	seg_params (dict): the segmenting parameters
# 	
# 	path_list (list): a list of paths to the audio to analyze (optionally use this if you don't want to segment every vocalization in the directory)
# 
# 	Returns
# 	-------
# 	None
# 	
# 	"""
	
def save_das_prediction_params(species, model_ID, iteration, min_len, fill_gap, segment_thres, save_dir):
	"""
	save parameters used during das predictions

	Parameters
	----------

	species (string): the two letter code for the species mode to prediction (either same as the species name or AllPero for model trained on all Pero species)

	model_ID (string): the modelID prefix (training date_time stamp) for the model you're going to use for prediction

	iteration (string): in the form 'iteration1', 'iteration2', etc. the number of time syou have run predict with this model using different parameters

	min_len(float): the minimum length in seconds for a segment to be considered for labeling.

	fill_gap (float): the intersyllable interval in seconds below which segments will be merged into 1

	save_dir (sting): the path to the root directory containing predictions for each species (each of which has its own sub-directory)


	Returns
	-------
	A dictionary of the parameters. 

	"""

	params_dict = {}
	params_dict['species'] =species
	params_dict['segment_threshold'] = segment_thres
	params_dict['segment_min_length'] = min_len
	params_dict['segment_fill_gap'] = fill_gap
	params_dict['model_ID'] = model_ID
	params_dict['iteration'] = iteration

	######NOTE: this is the path for prediction on raw wavs!######
	save_dir_root = os.path.join(save_dir,params_dict['species'],params_dict['model_ID'],params_dict['iteration'])+'/'
	
	#make the directories if they don't exit
	if params_dict['species'] not in os.listdir(save_dir):
		os.mkdir(os.path.join(save_dir, params_dict['species']))
	
	if params_dict['model_ID'] not in os.listdir(os.path.join(save_dir, params_dict['species'])):
		os.mkdir(os.path.join(save_dir, params_dict['species'], params_dict['model_ID']))
		
	if params_dict['iteration'] not in os.listdir(os.path.join(save_dir, params_dict['species'], params_dict['model_ID'])):
		os.mkdir(os.path.join(save_dir, params_dict['species'], params_dict['model_ID'], params_dict['iteration']))

	if '00_params' not in os.listdir(os.path.join(save_dir, params_dict['species'], params_dict['model_ID'], params_dict['iteration'])):
		os.mkdir(os.path.join(save_dir_root+'00_params'))
		
	#save
	save_parameters(params_dict, save_dir_root+'00_params/', params_dict['species']+'_'+params_dict['model_ID']+'_das_segmenting_params')

		
def save_parameters(params, save_dir, save_name):
    """
    NOTE this is a copy of the function in utilities.py
    TODO make your .py files into packages so you can import them instead of copying things like this
    
    save a dictionary as .json and double check if you want to overwrite it.

    Parameters
    ----------
    params_dict (dict): the parametes dictionary to be saved

    save_dir (str): the path to the place where the dictionary file will be saved

    save_name (str): the name of the file without any file extension

    Returns
    -------
    None

    """

    save_path = os.path.join(save_dir,save_name)

    if save_name not in [i.split('.')[0] for i in os.listdir(save_dir)]:
        print('making a new params file...')
        with open(save_path+'.json', 'w') as fp:
            json.dump(params, fp, indent=4)
        print('saved the params file to:\n',save_path)
        return

    else: 
        print('This file already exists in save_dir:', save_name)
        val = input('overwrite? y/n')

        if val == 'y':
            val = input('are you sure? y/n')

            if val == 'y':
                with open(save_path+'.json', 'w') as fp:
                    json.dump(params, fp, indent=4)
                print('ok - replaced existing file')
                return

            elif val == 'n':
                print('ok - no file saved')
                return

        elif val == 'n':
                print('ok - no file saved')
                return

        else:
                print('no file saved...')
                return

    return
	
	
def load_parameters(save_dir, save_name):
	"""
	load a dictionary from .json 

	Parameters
	----------
	
	save_dir (str): the path to the place where dictionary was saved

	save_name (str): the name of the .json file (including file extension)
	
	Returns
	-------
	params_dict (dict): the params dictionary you saved

	"""
	
	save_path = os.path.join(save_dir,save_name)
    
	with open(save_path+'.json', 'r') as fp:
			params_dict = json.load(fp)
	print('loaded parameters from:\n\t', save_path)
	return params_dict
