#this file contains functions related to calculating, aggregating, and organizing features of 
#vocalizations and pups who made them

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from librosa import rms
from scipy.io import wavfile
from scipy.signal import stft
from scipy.signal import hilbert
from datetime import date, datetime


def check_file_names(directory):
	"""
	check that file names follow the naming convention used by other functions

	Parameters
	----------
	directory (string): path to the directory containing all of the raw wav files

	Returns
	-------
	bad_lengths (list): a list of file names that don't have the right number of items 
	wrong_order (list): a list of lists containing file names that have the correct number 
    of items but at least one of them is not what it should be at index 0 
    and the location that has incorrect information at index 1.


	"""

	#directory is the directory containing the raw wav files to be checked
	files = [i for i in os.listdir(directory) if not i.startswith('.')] #ignore hidden files

	#check the files that don't have he correct number of entries and keep a list of those that don't
	bad_lengths = []
	for file in files:
		mdata = file.split('_')
		if len(mdata) != 13:
			bad_lengths.append(file)

	#ch2_BK_24224x25894_ltr1_pup1_ch2_3700_m_358_302_fr0_p5_2021-10-22_11-05-10.wav
	#check that the correct info is in the correct position (make sure to fix any files from the above cell before running)

	wrong_order = []
	species = ['BK', 'BW', 'MU', 'NB', 'IS', 'SW', 'LL', 'GO', 'LO', 'PO', 'MZ']

	files = [i for i in os.listdir(directory) if not i.startswith('.')] #ignore hidden files

	for file in files:
		mdata = file.split('_')
		if mdata[0] not in species:
			wrong_order.append([file, 0])

		elif len(mdata[1]) != 11:
			wrong_order.append([file, 1])

		elif not mdata[2].startswith('ltr'):
			wrong_order.append([file, 2])

		elif not mdata[3].startswith('pup'):
			wrong_order.append([file, 3])

		elif not mdata[4].startswith('ch'):
			wrong_order.append([file, 4])

		elif len(mdata[5]) != 4:
			wrong_order.append([file, 5])

		elif mdata[6] not in ['m', 'f']:
			wrong_order.append([file, 6])

		elif len(mdata[7]) != 3:
			wrong_order.append([file, 7])

		elif len(mdata[8]) != 3:
			wrong_order.append([file, 8])

		elif mdata[9] not in ['fr0', 'fr1']:
			wrong_order.append([file, 9])

		elif mdata[10] not in ['p1', 'p3', 'p5','p7', 'p9','p11','p13']:
			wrong_order.append([file, 10])

		elif mdata[11].split('-')[0] not in ['2019', '2020', '2021', '2022']:
			wrong_order.append([file, 11])

		elif len(mdata[12]) != 12:
			wrong_order.append([file, 12])

	return bad_lengths, wrong_order

#get the meta data from checked file names and save it
def get_meta_data(directory):
    """
	Get metadata recorded in a raw recording's file name.

    Arguments:
        Directory (string): path to the directory containing all of the raw wav files

	Returns:
        meta_df (dataframe): a dataframe where rows are recordings and columns are metadata categories.
	"""
    
	files = [i for i in os.listdir(directory) if not i.startswith('.')] #ignore hidden files
	all_rows = []

	for filename in files:

		#split the filename
		parts = filename.split('.')[0].split('_')

		#get the data
		channel = parts[4]
		species = parts[0]
		parents = parts[1]
		litter = parts[2]
		pup = parts[3]

		if parts[5] not in ['nan', 'na']:
			weight = int(parts[5])
		else:
			weight ='NA'

		if parts[6] not in ['nan', 'na']:
			sex = parts[6]
		else:
			sex ='NA'

		if parts[7] not in ['nan', 'na']:
			pre_T = int(parts[7])
		else:
			pre_T ='NA'

		if parts[8] not in ['nan', 'na']:
			post_T = int(parts[7])
		else:
			post_T ='NA'


		removal_flag = parts[9]
		age = int(parts[10][1:])
		date = parts[11]
		time = parts[12]

		#compile the date
		row = [species, parents, litter, pup, channel, weight, sex, pre_T,  post_T, removal_flag, age, date, time]
		all_rows.append(row)

	#put it all in a data frame
	meta_df = pd.DataFrame.from_records(all_rows, columns =['species', 'parents', 'litter', 'pup', 'channel', 'weight_mg', 'sex', 'pre_T', 'post_T', 'removal_flag', 'age', 'date', 'time'])
	return meta_df

def get_filename_keys(dataset):
    """
    Get the category names used to label each file in a dataset

    Arguments:
        dataset (str): the dataset the pup comes from. Must be one of 'development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2'

    Returns:
       filename_keys (list): a list of keys to name the information in each section of each recording's file name
    """
    
    assert dataset in ['development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2']

    #these are keys for each dataset that correspond to information in each file name
    development_keys = ['species', 'breeding_pair','litter_number', 'pup_number', 'mic_channel', 'weight_mg', 'sex', 'start_temp', 'end_temp', 
                        'removal_flag', 'age', 'date', 'time']
    bw_po_cf_keys = ['species', 'breeding_pair', 'pup_number', 'age', 'weight_mg', 'sex', 'litter_size', 'date', 'time']
    bw_po_f1_keys = development_keys.copy()
    bw_po_f2_keys = ['mic_chanel_prefix', 'cross_prefix', 'breeding_pair','family', 'litter_number', 'pup_number', 'mic_channel', 'weight_mg', 
                     'sex', 'start_temp', 'end_temp', 'removal_flag', 'age', 'date', 'time']

    if dataset == 'development':
        keys = development_keys    
    elif dataset == 'bw_po_cf':
        keys = bw_po_cf_keys
    elif dataset == 'bw_po_f1':
        keys = bw_po_f1_keys
    elif dataset == 'bw_po_f2':
        keys = bw_po_f2_keys

    return keys


def get_pup_metadata(source_path, dataset):
    """
    Get meta data for each pup in a dataset: species, dam/sire, litterID, littersize, weight, sex, age, temperatures.

    Arguments:
        source_file (str): the full path to the recording for which you want metdata
        dataset (str): the dataset the pup comes from. Must be one of 'development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2'

    Returns:
        meta_data (dict): a dictionary of the metadata, which can be further aggregated into a dataframe with multiple pups
    """
    #initialize dict and assert dataset is correct
    #"source_file" can refer to a raw recording from a pup (1 pup for each raw recording)
    #or it can refer to a single clip from a single vocalization from a pup
    #here it refers to the latter, but you should fix this potential ambiguity

    pup_dict = {}
    source_file = os.path.split(source_path)[-1]
    assert dataset in ['development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2']

    #get the keys to use for this dataset
    keys = get_filename_keys(dataset)

    #get the values from the file name and add to dict
    values = source_file.split('.')[0].split('_')
    assert len(keys) == len(values), "keys and values don't match"
    for key, value in zip(keys,values):
        pup_dict[key]=value
        pup_dict['pup'] = source_file

    return pup_dict


def aggregate_pup(source_path, features, features_path):
    """
    Aggregate warbleR acoustic features for a single pup

    Arguments:
        features_path (str): full path to the labeled warbleR features to use
        source_path (str): the full path to the recording for which you want metdata
        features (list): list of warbleR features you want to aggregate

    Returns:
        aggregate_data (dict): a dictionary of the metadata, which can be further aggregated into a dataframe with multiple pups

    """

    #get the pup name and assert dataset is correct
    pup = os.path.split(source_path)[-1]
    print(pup)
    assert len(features) == 26

    #get the path to the warbleR features and make sure it's what you expect
    if features_path.endswith('.csv'):
        feats = pd.read_csv(features_path)
    elif features_path.endswith('.feather'):
        feats = pd.read_feather(features_path)

    assert 'BW_hdbscan_label0_withspace.wav.wav' not in feats['source_file'] #TODO fix this for the development dataset
    assert np.any(feats.duplicated()) == 0

    #get the vocalizations from this pup
    assert pup.split('.')[0] in list(feats['pup']), "the pup printed above isn't in the 'pup' column in the features_path file"
    feats = feats.loc[feats['pup'] == pup.split('.')[0]]

    #get the summary stats: cry count, whistle count, scratch count, and mean, median, min, max, and standard deviation of each vocalization of each type

    #count data
    feats_dict = {}
    feats_dict['pup'] = pup
    feats_dict['cry_count'] = len(feats.loc[feats['predicted_label'] == 'cry'])
    feats_dict['whistle_count'] = len(feats.loc[feats['predicted_label'] == 'whistle'])
    feats_dict['scratch_count'] = len(feats.loc[feats['predicted_label'] == 'scratch'])
    feats_dict['total_sounds_detected'] = feats_dict['cry_count']+feats_dict['whistle_count']+feats_dict['scratch_count']
    feats_dict['total_vocalizations_detects'] = feats_dict['cry_count']+feats_dict['whistle_count']


    #use groupby to get the aggregate feature data - note it is not strictly necessayr to group by pup since there is only one pup
    #keeping anyway because it works and I don't have time to break it
    feat_means = feats.groupby(by=['predicted_label']).mean()
    feat_vars = feats.groupby(by=['predicted_label']).aggregate(np.var)
    feat_mins = feats.groupby(by=['predicted_label']).aggregate(np.min)  
    feat_maxs = feats.groupby(by=['predicted_label']).aggregate(np.max)
    feat_meds = feats.groupby(by=['predicted_label']).aggregate(np.median) 

    for feature in features:
        
        variances = feat_vars[feature]
        means = feat_means[feature]
        mins = feat_mins[feature]
        maxs = feat_maxs[feature]
        meds = feat_meds[feature]

        try: cry_mean = means.loc['cry']
        except: cry_mean = float('NaN')

        try: cry_variance = variances.loc['cry']
        except:cry_variance = float('NaN')

        try: cry_min = mins.loc['cry']
        except:cry_min = float('NaN')

        try: cry_max = maxs.loc['cry']
        except:cry_max = float('NaN')

        try: cry_med = meds.loc['cry']
        except:cry_med = float('NaN')

        try: whistle_mean = means.loc['whistle']
        except: whistle_mean = float('NaN')

        try: whistle_variance = variances.loc['whistle']
        except: whistle_variance = float('NaN')

        try: whistle_min = mins.loc['whistle']
        except: whistle_min = float('NaN')

        try: whistle_max = maxs.loc['whistle']
        except: whistle_max = float('NaN')

        try: whistle_med = meds.loc['whistle']
        except: whistle_med = float('NaN')

        try: scratch_mean = means.loc['scratch']
        except: scratch_mean = float('NaN')

        try: scratch_variance = variances.loc['scratch']
        except: scratch_variance = float('NaN')

        try: scratch_min = mins.loc['scratch']
        except: scratch_min = float('NaN')

        try: scratch_max = maxs.loc['scratch']
        except: scratch_max = float('NaN')

        try: scratch_med = meds.loc['scratch']
        except: scratch_med = float('NaN')

        feats_dict[f"cry_{feature}_mean"] = cry_mean
        feats_dict[f"whistle_{feature}_mean"] = whistle_mean
        feats_dict[f"scratch_{feature}_mean"] = scratch_mean

        feats_dict[f"cry_{feature}_variance"] = cry_variance
        feats_dict[f"whistle_{feature}_variance"] = whistle_variance
        feats_dict[f"scratch_{feature}_variance"] = scratch_variance

        feats_dict[f"cry_{feature}_min"] = cry_min
        feats_dict[f"whistle_{feature}_min"] = whistle_min
        feats_dict[f"scratch_{feature}_min"] = scratch_min

        feats_dict[f"cry_{feature}_max"] = cry_max
        feats_dict[f"whistle_{feature}_max"] = whistle_max
        feats_dict[f"scratch_{feature}_max"] = scratch_max

        feats_dict[f"cry_{feature}_med"] = cry_med
        feats_dict[f"whistle_{feature}_med"] = whistle_med
        feats_dict[f"scratch_{feature}_med"] = scratch_med

    return feats_dict



def aggregate_all_pups(source_list, dataset, save, save_name, save_dir, features, features_path):
    """
    For each pup in source_list, aggregate warbleR acoustic features by pup, get pup metdata, then combine them into a single dataframe

    Arguments:
        source_list (list): list of full paths to source_files (one per pup) you want to process
        dataset (str): the dataset the pups come from. Must be one of 'development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2'
        save (boolean): if True, save the datframe as a csv named save_name in the directory save_dir
        save_name (string): name of the csv to save if save is True
        save_dir (string): full path to the directory where the dataframe should be saved if save is true
        raw_dir (string): 

    Returns:
        all_pup_data (dataframe): a dictionary of the metadata, which can be further aggregated into a dataframe with multiple pups

    """

    #assert inputs make sense
    assert dataset in ['development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2'], "dataset must be one of ['development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2']"
    assert isinstance(save, bool), "save must be True or False"
    assert os.path.exists(save_dir), "save_dir doesn't exist"

    #get the metadata for every pup
    all_pup_metadata = []
    for source_path in source_list:
        pup_metadata = get_pup_metadata(source_path=source_path, dataset=dataset)
        pup_metadata = pd.DataFrame.from_records([pup_metadata])
        all_pup_metadata.append(pup_metadata)
    print('done collecting metadata...')

    #get the aggregate features for every pup
    all_pup_features = []
    for source_path in source_list:
        pup_features = aggregate_pup(source_path=source_path, features=features, features_path=features_path)
        pup_features = pd.DataFrame.from_records([pup_features])
        all_pup_features.append(pup_features)
    print('done collecting pup features...')

    #add all the pups toegther
    all_pup_features_df = pd.concat(all_pup_features)
    all_pup_metadata_df = pd.concat(all_pup_metadata)
    
    #
    if save:
        if save_name.endswith('.csv'):
            features_save_name = ('_').join([save_name.split('.')[0], 'agg_features.csv'])
            metadata_save_name = ('_').join([save_name.split('.')[0], 'agg_metadata.csv'])
            
            all_pup_features_df.to_csv(os.path.join(save_dir,features_save_name), index=False)
            all_pup_metadata_df.to_csv(os.path.join(save_dir,metadata_save_name), index=False)
            
            print('saved features to:\n\t', os.path.join(save_dir,features_save_name))
            print('saved metadata to:\n\t', os.path.join(save_dir,metadata_save_name))
            
        elif save_name.endswith('.feather'):
            features_save_name = ('_').join([save_name.split('.')[0], 'agg_features.feather'])
            metadata_save_name = ('_').join([save_name.split('.')[0], 'agg_metadata.feather'])
            
            all_pup_features_df.to_feather(os.path.join(save_dir,features_save_name))
            all_pup_metadata_df.to_csv(os.path.join(save_dir,metadata_save_name))
            
            print('saved features to:\n\t', os.path.join(save_dir,features_save_name))
            print('saved metadata to:\n\t', os.path.join(save_dir,metadata_save_name))

    return all_pup_features_df, all_pup_metadata_df
        


def get_snr(clip_path, noise_path, algorithm = 1):
     """
    Use noise clips generated by get_silence() to calculate signal to noise

    Arguments:
        clip_path (str): the wav clip you want to calculate snr on
        noise_path (str): the background clip for the recording that the vocalization came from (generated by annotation.get_noise_clip())
        algorithm (int): Must be one of 1, 2, or 3 (see comment below for description of each)
    Returns:
        snr (floar): the signal to noise ratio
    """
#use noise clips generated by annotation.get_noise_clip() to calculate signal to noise ratio
#based on snr calculations from warlbeR (https://github.com/maRce10/warbleR)
# 1: ratio of S mean amplitude envelope to N mean amplitude envelope (mean(env(S))/mean(env(N)))
# 2: ratio of S amplitude envelope RMS (root mean square) to N amplitude envelope RMS (rms(env(S))/rms(env(N)))
# 3: ratio of the difference between S amplitude envelope RMS and N amplitude envelope RMS to N amplitude envelope RMS ((rms(env(S)) - rms(env(N)))/rms(env(N)))
    
    
	fs, noise = wavfile.read(noise_path)
	_, signal = wavfile.read(clip_path)
	
	if algorithm == 1:	
		signal_amplitude_envelope = np.abs(hilbert(signal))
		noise_amplitdue_envelope = np.abs(hilbert(noise))
		snr = np.mean(signal_amplitude_envelope)/np.mean(noise_amplitdue_envelope)
		return snr
		
	elif algorithm == 2:
        
        signal_rms = rms(y=np.array(signal, 'float'))
        noise_rms = rms(y=np.array(noise, 'float'))
        snr = np.mean(signal_rms)/np.mean(noise_rms)

	elif algorithm == 3:
		signal_amplitude_envelope = np.abs(hilbert(signal))
		noise_amplitdue_envelope = np.abs(hilbert(noise))
		snr = (rms(y=signal_amplitude_envelope) - rms(y=noise_amplitdue_envelope))/rms(y=noise_amplitdue_envelope)

    return snr

#same as above but iterate through a directory - right now only deal with algorithms 1 and 2
def get_snr_batch(clip_dir, noise_dir, species, algorithm):
    
    """
    Run get_snr() on a batch of vocalization clips in a directory

    Arguments:
        clip_dir (str): the path to the directory containing the wav clips for which you want to get signal to noise
        noise_dir (str): the path to the directory containing the background clip for each wav in clip_dir (one noise clip per recording)
        algorithm (int): Must be one of 1, 2, or 3 (see comment below for description of each)
    Returns:
        snr_df (dataframe): a dataframe where each row is a vocalization and columns are path to vocalization file, snr, and algorithm
    """
	
    #get paths to vocalizations
	if species != None:
		vocs = [i for i in os.listdir(clip_dir) if i.startswith(species)]
	else:
		vocs = [i for i in os.listdir(clip_dir) if not i.startswith('.')]

	sig2noise_list = []
	source_files = []

    #iterate through vocalizations
	for voc in tqdm(vocs):

		#get the audio
		clip_path = clip_dir+voc
		noise_path = noise_dir+voc.split('_clip')[0]+'_silence-clip_.wav'
	
		#get signal to noise
        snr = get_snr(clip_path=clip_path, noise_path=noise_path, algorithm=algorithm)
        
        #update
        sig2noise_list.append(snr)
        source_files.append(voc)

    #write data to dataframe
	snr_df = pd.DataFrame()
	snr_df['source_file'] = source_files
	snr_df['snr'] = sig2noise_list
    snr_df['algorithm'] = algorithm
    
	return snr_df
	
	
#calculate how much of the audio clip is clipped
#test for clipping
#See https://dsp.stackexchange.com/questions/61996/what-are-the-semantics-of-wav-file-sample-values#:~:text=The%2016%20bit%20values%20in,but%20this%20is%20an%20interpretation.
#for where the threshold comes from (our audio is 16-bit and values range from -32768 to 32767)
#threshold is a percent of the maximum possible value the microphone can handle in decimals (ie, .95)
def get_clipping(clip_path, threshold):
    
    """
    Get the percent of wav file that is clipped

    Arguments:
        clip_path (str): the path to the wav file
        threshold (float): percent of maximum value (32767) possible to be read with 16-bit encoding above which you will call audio "clipped"
    Returns:
        percent_clipped (float): percent of frames in the wav file that exceed the clipping threshold
    """

    #check inputs
    assert os.path.exists(wav)
    assert 0 < threshold < 1, "Threshold must be a value between 0 and 1"
    
    #calculate percent clipped
    fs, audio = wavfile.read(clip_path)
    rect_wav = np.abs(audio)
    clipping_limit = threshold*32767
    clipped = rect_wav[rect_wav>clipping_limit]
    percent_clipped = len(clipped)/len(audio)

    return percent_clipped
    
#same as above but iterate through a directory of clips
#returns two lists: the clipping percents and the corres
def get_clipping_batch(clip_dir, threshold, species = None):
    
    """
    Run get_clipping() on a batch of vocalization clips in a directory

    Arguments:
        clip_dir (str): the path to the directory containing the wav clips for which you want to evaluate clipping
        threshold (float): percent of maximum value (32767) possible to be read with 16-bit encoding above which you will call audio "clipped"
        species (str): optional 2 letter code for species if you only want to process one species at a time
    Returns:
        clipping_df (dataframe): a dataframe where each row is wav file (eg vocalization) and columns are path to wav file, % clipped, and threshold
    """
	
	if species != None:
		to_process = [os.path.join(wav_dir,i) for i in os.listdir(wav_dir) if i.startswith(species) and i.endswith('.wav')]
	
	else:
		to_process = [os.path.join(wav_dir,i) for i in os.listdir(wav_dir) if not i.startswith('.') and i.endswith('.wav')]

	source_files = []
	clipping_percents = []
	for wav in tqdm(to_process):
		#get clipping percent
		fs, audio = wavfile.read(wav)
		rect_wav = np.abs(audio)
		clipping_limit = threshold*32767 #32767 is the max value possible for our wav encoding (16 bit)
		clipped = rect_wav[rect_wav>clipping_limit]
		percent_clipped = len(clipped)/len(audio)
	
		#update 
		source_files.append(wav.split('/')[-1])
		clipping_percents.append(percent_clipped)

	clipping_df = pd.DataFrame()
	clipping_df['source_file'] = source_files
	clipping_df['percent_clipped'] = clipping_percents
	clipping_df['clipping_threshold'] = threshold*32767
	return clipping_df
def write_warbleR_job_scripts(dataset, save_root, wav_root, script_dir):
    """
    Write sbatch job files to run warbleR_feature_extraction.R on a computing cluster 
    
    Required processing steps:
        1. You have a csv of all pups to process with a column called species, which will be used to group the features into directories
        2. You have a directory containing one wav clip for every vocalization in the above csv (no subdirectories)
    
    Arguments:
        dataset (str): one of ['bw_po_cf', 'bw_po_f1', 'bw_po_f2', development] (cross foster, F1, F2, development)
        save_root (str): the place where csv of acoustic features will be saved
        wav_root (str): the place containing the wav files (one per vocalization) to get features from
        script_dir (str): the place to save the sbatch scripts (one per species)
    
    Returns
        None
    """
    
    path_to_warbleR_extract = '/n/hoekstra_lab_tier1/Users/njourjine/manuscript/notebooks/00_manuscript/warbleR_extract.R'
    
    
    assert dataset in ['bw_po_cf', 'bw_po_f1', 'bw_po_f2', 'development']
    assert os.path.exists(save_root)
    assert os.path.exists(wav_root)

    #get the species - note that for the non_development data sets these are not strictly species but some other way of grouping the recordings (treatment/mic channel)
    if dataset == 'bw_po_cf':
        source_df=pd.read_csv('/n/hoekstra_lab_tier1/Users/njourjine/manuscript/audio/segments/bw_po_cf/amplitude_segmentated/20220921_030633/all_combined.csv')
        species_list = sorted(source_df['species'].unique())

    elif dataset == 'bw_po_f1':
        source_df=pd.read_csv('/n/hoekstra_lab_tier1/Users/njourjine/manuscript/audio/segments/bw_po_f1/amplitude_segmentated/20220920_072032/all_combined.csv')
        species_list = sorted(source_df['species'].unique())

    elif dataset == 'bw_po_f2':
        source_df=pd.read_csv('/n/hoekstra_lab_tier1/Users/njourjine/manuscript/audio/segments/bw_po_f2/amplitude_segmentated/20220921_040238/all_combined.csv')
        species_list = sorted(source_df['species'].unique())
        
    elif dataset == 'development':
        source_df=pd.read_csv('/n/hoekstra_lab_tier1/Users/njourjine/manuscript/audio/segments/amplitude_segmentation/final/all_predictions.csv')
        species_list = sorted(source_df['species'].unique())

    #make a dictionary for paths
    paths_dict = {}
    for species in species_list:
        paths_dict[species] = {}

    #make the path to the directory where features will be saved
    today = str(date.today())
    today = ('').join(today.split('-'))
    now = str(datetime.now())
    time = now.split(' ')[-1]
    time = ('').join(time.split('.')[0].split(':'))
    save_path = os.path.join(save_root,('_').join([today,time]))
    os.mkdir(save_path)
    
    #populate the dictionary
    for species in species_list:

        #get the path to the raw clips for which features will be calculated
        wav_path = os.path.join(wav_root, species)

        #add the the paths to dictions
        paths_dict[species]['wav_path'] = wav_path
        paths_dict[species]['save_path'] = save_path
        #for each species, write an .sbatch file (no job array here) with
        lines = [
        '#!/bin/bash\n', 
        '#\n', 
        '#SBATCH -J warb # A single job name for the array\n', 
        '#SBATCH -p hoekstra,shared\n', 
        '#SBATCH -c 1 # one core\n', 
        '#SBATCH -t 0-8:00 # Running time of 8 hours\n', 
        '#SBATCH --mem 16000 # Memory request of 24 GB\n', 
        '#SBATCH -o '+species+'_warblExtract_%A_%a.out # Standard output\n', 
        '#SBATCH -e '+species+'_warblExtract_%A_%a.err # Standard error\n',  
        '\n', 
        '#load R\n',
        'module load intel/2017c impi/2017.4.239 FFTW/3.3.8-fasrc01\n',
        'module load R/4.0.2-fasrc01\n',
        'export R_LIBS_USER=$HOME/apps/R_4.0.2:$R_LIBS_USER\n',
        '\n',
        '#set directories\n',
        'SPECIES='+ species + '\n',
        'WAVSDIR='+ paths_dict[species]['wav_path'] +'\n',
        'SAVEDIR='+ paths_dict[species]['save_path'] +'\n',
        '\n',
        '#run warbleR extract\n',
        'Rscript /n/hoekstra_lab_tier1/Users/njourjine/manuscript/notebooks/00_manuscript/warbleR_extract.R'+" $SPECIES"+" $WAVSDIR"+" $SAVEDIR"
        ]

        #write lines
        sbatch_name = 'warbleR_extract_'+species+'.sbatch'
        sbatch_save_path = os.path.join(script_dir, sbatch_name)
        with open(sbatch_save_path, 'w') as f:
            f.writelines(lines)
    
    
    #write a parent .sh script to run all of the species' sbatch scripts
    parent_lines =  []
    for species in species_list:
        parent_lines.extend(['echo ', species, '\n',
                           'sbatch warbleR_extract_', species, '.sbatch\n',
                           'sleep 5\n'])
        
    sh_name = 'warbleR_extract_parent.sh'
    sh_save_path = os.path.join(script_dir, sh_name)
    with open(sh_save_path, 'w') as f:
            f.writelines(parent_lines)
            
    print('wrote job scripts to:\n\t', script_dir)
        




   