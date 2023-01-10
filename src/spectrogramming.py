#this file contains functions for generating and visualizing spectrograms

import glob
import os
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
import time
from sklearn.preprocessing import StandardScaler
import umap
from scipy.signal import butter, lfilter

def get_spectrogram(data, fs, nperseg, noverlap, num_freq_bins, num_time_bins, min_freq, max_freq, fill_value, max_dur, spec_min_val, spec_max_val): 
    """
    Make a spectrogram with a pre-determined number of frequency and time bins. Useful for generating spectrograms that are all the same shape.
    Modified from: https://autoencoded-vocal-analysis.readthedocs.io/en/latest/_modules/ava/preprocessing/utils.html?highlight=spec
        
    Arguments:
        data (numpy array): the wav file you want to make a spectrogram from
        nperseg (int): nperseg for spectrogram generation
        noverlap (int): noverlap for spectrogram generation with scipy.
        num_freq_bins (int): number of frequency bins for spectrogram
        num_time_bins (int): number of time bins for spectrogram
        min_freq (int): minimum frequency for spectrogram
        max_freq (int): maximum frequency for spectrogram
        fill_value (float): the value to use for points outside of the interpolation domain (for scipy.interpolate.interp2d)
        max_dur (float): the duration of the longest vocalization you want to consider - determines the time axis scaling
        spec_min_val (float): maximum spectrogram value
        spec_max_vale (float): minimum spectogram value

    Returns
    -------
        f (numpy array): the frequency bins of the spectrogram
        t (numpy array): the time bins of the spectrogram
        specgram: the spectrogram
    """

	#get the spectrogram
	f,t,specgram = stft(data, fs, nperseg=nperseg, noverlap=noverlap) #default winow is Hann

	#define the target frequencies and times for interpolation
	duration = np.max(t)
	target_freqs = np.linspace(min_freq, max_freq, num_freq_bins)
	shoulder = 0.5 * (max_dur - duration)
	target_times = np.linspace(0 - shoulder, duration+shoulder, num_time_bins)

	#process
	specgram = np.log(np.abs(specgram)+ 1e-12) # make a log spectrogram 
	interp = interp2d(t, f, specgram, copy=False, bounds_error=False, fill_value=fill_value) #define the interpolation object 
	target_times = np.linspace(0 - shoulder, duration+shoulder, num_time_bins) #define the time axis of the spectrogram
	interp_spec = interp(target_times, target_freqs, assume_sorted=True) #interpolate 
	specgram = interp_spec
	specgram -= spec_min_val #normalize
	specgram /= (spec_max_val - spec_min_val) #scale
	specgram = np.clip(specgram, 0.0, 1.0) #clip

	return f,t,specgram

def wavs_to_umap(clips_dir, noise_floors_path, species, noise_floor, spec_params, num_to_process, filtered_clips, version, save_root):
    
    """
    Take a list of wav files, generate spectrograms from those files, then find a umap embedding for those spectrograms and save the coordinates
    
    Arguments:
        clips_dir (str): the directory containing all of the raw wav files
        species (str, optional): optional two letter code for species if you want to just process one species
        filtered_clips (list, optional): optional list of paths to wav files if you want to just process files in a given list
        noise_floors_path (str, optional): optional path to a csv containing noise floors for the clips you want to process 
        noise_floor (float, optional): optional minimum spectrogram value to be used if noise_floors_path not provided
        spec_params (dict): dictionary of parameters for generating spectrograms
        num_to_process (int or 'all'): Number to process. If 'all' process everything. Useful for debugging to set this to 10 or 20 first.
        version (str): version name to keep track of multiple different umap embeddings of the same wavs
        save_root (str): path to the directory where the umap coordinates will be saved
    
    Returns:
        None
    
    """

	#get spectrograms
	specs_list, source_files = specs_from_wavs(clips_dir = clips_dir, 
                                               noise_floors_path=noise_floors_path, 
                                               species = species,
                                               noise_floor = noise_floor,
                                               spec_params=spec_params, 
                                               num_to_process = num_to_process, 
                                               filtered_clips = filtered_clips)
										  
	#linearize
	specs_lin, shape = linearize_specs(specs_list)
	del specs_list #free up space
	print(shape)
	
	#zscore
	df_umap, zscored = zscore_specs(specs_lin, source_files)
	del specs_lin #free up space
	
	#embed
	umap1, umap2 = get_umap(zscored)
	del zscored #free up space
	
	#plot 
	print('plotting...')
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(1,1,1)
	ax = plt.scatter(
		umap1,
		umap2,
		c = 'k',
		s = 1,
		alpha = .25, 
		cmap='viridis')
	plt.xlim([-20,20])
	plt.ylim([-20,20])
	plt.show
	plots_save_dir = save_root+'umap_plots/'
	if 'umap_plots' not in os.listdir(save_root):
		os.mkdir(plots_save_dir)
	plot_save_name = spec_params['species'] + '_black_' + version + '.jpeg'
	plt.savefig(plots_save_dir+plot_save_name, dpi=600)
	
	#save
	print('saving umap coordinates...')
	file_format = '.feather'
	df_umap = df_umap.reset_index(drop=True)
	coordinates_save_dir = save_root+'umap_coordinates/'
	if 'umap_coordinates' not in os.listdir(save_root):
		os.mkdir(coordinates_save_dir)
	coordinates_save_name = spec_params['species'] + version + file_format
	
	save_name = coordinates_save_dir+coordinates_save_name
	df_umap.columns = df_umap.columns.map(str)
	df_umap['umap1'] = umap1
	df_umap['umap2'] = umap2
	df_umap.to_feather(save_name)

def specs_from_wavs(clips_dir, species, filtered_clips, noise_floors_path, noise_floor, spec_params, num_to_process):
    """
    Generate spectrograms for a list of wav files - useful for umap embedding
    
    Arguments:
        clips_dir (str): the directory containing all of the raw wav files
        species (str, optional): optional two letter code for species if you want to just process one species
        filtered_clips (list, optional): optional list of paths to wav files if you want to just process files in a given list
        noise_floors_path (str, optional): optional path to a csv containing noise floors for the clips you want to process 
        noise_floor (float, optional): optional minimum spectrogram value to be used if noise_floors_path not provided
        spec_params (dict): dictionary of parameters for generating spectrograms
        num_to_process (int or 'all'): Number to process. If 'all' process everything. Useful for debugging to set this to 10 or 20 first.
        
    Returns:
        specs_list (list): a list of spectrograms as numpy arrays
        source_files (list): a list of file names for each spectrogram
    """
    specs_list = []
    source_files = []

    # print info about what is about to happen (avoid starting a long process with the wrong parameters)
    if species != None and filtered_clips == None:
        wav_paths = [clips_dir+i for i in os.listdir(clips_dir) if i.startswith(species) and not i.startswith('.')]
        print('generating all specs from', species, '...')

    elif species == None and filtered_clips == None:
        wav_paths = [clips_dir+i for i in os.listdir(clips_dir) if not i.startswith('.') and not i.startswith('00')]
        print('generating specs from every wav in the clips dir...')

    elif species == None and filtered_clips != None:
        wav_paths = filtered_clips
        print('generating specs from the paths in the list filtered_clips...')

    elif species != None and filtered_clips != None:
        print('species and fitered_clips cannot both be None')
        return

    if noise_floors_path != None and noise_floor != None:
        print('Either provide a path to noise floors for each clip or a single noise floor for all clips, not both')
        return

    if num_to_process != 'all':
        to_process = wav_paths[0:num_to_process]
        print('processesing', num_to_process, 'clips')
    else:
        to_process = wav_paths

    if noise_floors_path != None:
        noise_floors = pd.read_csv(noise_floors_path)
        print('processesing using the noise floors provided in noise_floors_path...')

    print('making spectrograms...')
    for path in tqdm(to_process):
        source_files.append(path.split('/')[-1])

        #get the noise floor
        source_file = path.split('/')[-1].split('_clip')[0]+'.wav'

        if noise_floors_path != None:
            noise_floor = list(noise_floors['noise_floor'].loc[noise_floors['source_file'] == source_file])[0]

        #get the wav
        fs, wav = wavfile.read(path)



        #get the spectrogram
        if noise_floors_path != None:
            spec_params['fill_value'] = noise_floor

        f, t, spec = get_spectrogram(data = wav,
                                     fs=spec_params['fs'],
                                     nperseg=spec_params['nperseg'],
                                     noverlap=spec_params['noverlap'],
                                     num_freq_bins=spec_params['num_freq_bins'],
                                     num_time_bins=spec_params['num_time_bins'],
                                     min_freq=spec_params['min_freq'],
                                     max_freq=spec_params['max_freq'],
                                     fill_value = spec_params['fill_value'],
                                     max_dur=spec_params['max_duration'],
                                     spec_min_val=noise_floor, 
                                     spec_max_val=spec_params['spec_max_val'])

        specs_list.append(spec) #downsample time and frequency

    print('done.')
    return specs_list, source_files

def linearize_specs(specs_list):
    """
    Linearize each spectrogram in a list. For UMAP prepprocessing.
    
    Arguments:
    
        specs_list (list): a list of spectrograms as numpy arrays (eg, output of specs from wavs)
        
    Returns:
        shape (numpy array): the shape of np.array(specs list)
        specs_lin (list): a list of linearized spectrograms as numpy arrays
    """
	
	#print some useful info
	specs = np.array(specs_list)
	shape = np.shape(specs)
	print('shape of spectrograms array is:', np.shape(specs))
	
	#linearize
	print('linearizing spectrograms...')
	num_features = specs.shape[-1]*specs.shape[-2]
	specs_lin = specs.reshape([-1, num_features])
	print('done.')
    
	return specs_lin, shape

def zscore_specs(specs_lin, source_files):
    """
    Zscore each spectrogram in a list. For UMAP preprocessing.
    
    Arguments:
        specs_lin (list): a list of linearized spectrograms as numpy arrays (eg, output of linearize_specs)
        
    Returns:
        df_umap (data frame): a dataframe where each row is a spectrogram and each column is a pixel (plus a source_file column with the path to the wav)
        zscored (numpy array): an array of zscored, linearized spectrograms for umap embedding
    """
    
    
	#make a dataframe
	df_umap = pd.DataFrame(specs_lin)
	df_umap['source_file'] = source_files
	
	# Z-score the spectrogams
	print('z scoring...')
	zscored = StandardScaler().fit_transform(df_umap.drop(columns=['source_file']))	
	print('done.')
	return df_umap, zscored

def get_umap(standardized_features):
    """
    Find a umap embedding for a set of spectrograms
    
    Arguments:
        standardized_features (numpy array): an array of linearized, zscored spectrograms (eg output of zscore specs)
        
    Returns:
        umap1 (numpy array): x coordinates of umap embedding for each vocalization
        umap2 (numpy array): y coordinares of umap embedding for each vocalization
    """
	
	#find an embedding
	print('finding a umap embedding...')
	
	start_time = time.time()
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(standardized_features)
	print('embedding took', time.time() - start_time, 'seconds.')
	
	umap1 = embedding[:, 0]
	umap2 = embedding[:, 1]

	return umap1, umap2

def save_umap(df_umap, umap1, umap2, save_name = None, file_format = 'feather'):
	"""
    Save coordinates of a umap embedding and their corresponding source files
    
    Arguments:
        df_umap (dataframe): dataframe of non-zscored spectrograms (eg output of zscore specs)
        umap1 (numpy array): x coordinates of umap embedding for each vocalization
        umap2 (numpy array): y coordinares of umap embedding for each vocalization
        save_name (str): the path to the file you want to write
        file_format (str): must be one of 'csv' or 'feather'. feather is better for very large files
        
        
    Returns:
        None
    """
    
    #check inputs
    assert file_format in ['csv', 'feather']
    
	if file_format == 'csv':
		df_umap['embedding_dim_1'] = umap1
		df_umap['embedding_dim_2'] = umap2
		df_umap.to_csv(save_name, index=False)
		print('done.')
	
	elif file_format == 'feather':
		df_umap['embedding_dim_1'] = umap1
		df_umap['embedding_dim_2'] = umap2
		df_umap.to_feather(save_name)
		print('done.')

def spec_avg_from_list(spec_list):
    """
    Get the average spectrogram from a list of spectrograms
    
    Arguments:
        spec_list (list): a list of nonlinearized spectrograms as numpy arrays
    
    Returns:
        avg_spec_image (numpy array): the average of spectrograms in the list
    
    """
    
    #get average
    avg_spec_image = np.mean(spec_list, axis=0)
        
    return avg_spec_image
     
def show_specs(frame, num_freq_bins, num_time_bins, columns_to_drop):
    """
    Plot spectrogram images from a dataframe
    
    Arguments:
       frame (dataframe): a dataframe of pixel values where each row is a spectrogram and columns are pixel IDs (eg output of save_umap)
       num_freq_bins (int): number of frequency bins the spectrograms have
       num_time_bins (int): number of time bins the spectrograms have
       columns_to_drop (list of strings): columns that are not spectrogram pixel IDs
    
    Returns:
        None
    
    """
    
	for i in range(len(frame)):
		print(frame['source_file'].iloc[i])
		to_plot = frame.drop(columns = columns_to_drop)
		img = to_plot.iloc[i]
		img = np.array(img).reshape((num_freq_bins, num_time_bins))
		plt.imshow(img, origin = 'lower', extent = (num_freq_bins, 0, num_time_bins, 0 ))
		plt.show()

def files_from_umap(frame, umap1_name, umap2_name, umap1_thresh, umap2_thresh, source_dir):
    """
    Get the paths to the wav files in all or a portion of a umap embedding.
    Useful for getting spectrograms and/or spectrograms averages from particular regions of UMAP space.
    
    Arguments:
        frame (dataframe): a dataframe of pixel values where each row is a spectrogram and columns are pixel IDs
        umap1_name (str): the column name in frame for the umap x coordinate
        umap2_name (str): the column name in frame for the umap y coordinate
        umap1_thresh (list of two floats): list of minimum and maximum of UMAP x coordinate from which to get paths
        umap2_thresh (list of two floats): list of minimum and maximum of UMAP y coordinate from which to get paths
        source_dir (str): path to the directory containing all of the wav clips that went into the embedding
    
    Returns:
        source_files (list): list of paths to wall wav clips in the square defined by umap1_thresh and umap2_thresh
    
    """

    #get the spectrograms
    temp = frame[(frame[umap1_name] > umap1_thresh[0]) & (frame[umap1_name] < umap1_thresh[1]) & (frame[umap2_name] > umap2_thresh[0]) & (frame[umap2_name] < umap2_thresh[1])]
    
    #get the paths to their source files
    source_files = [source_dir+i for i in temp['source_file']]
    return source_files

def spec_avg_from_umap(frame, umap1_name, umap2_name, umap1_thresh, umap2_thresh, num_freq_bins, num_time_bins):
    """
    Get an average of all the spectrograms in a region of umap space
    
    Arguments:
        frame (dataframe): a dataframe of pixel values where each row is a spectrogram and columns are pixel IDs
        umap1_name (str): the column name in frame for the umap x coordinate
        umap2_name (str): the column name in frame for the umap y coordinate
        umap1_thresh (list of two floats): list of minimum and maximum of UMAP x coordinate from which to get paths
        umap2_thresh (list of two floats): list of minimum and maximum of UMAP y coordinate from which to get paths
        num_freq_bins (int): number of frequency bins used to generate the spectrogram
        num_time_bins (int): number of time bins used to generate the spectrogram
    
    Returns:
        source_files (list): list of paths to wall wav clips in the square defined by umap1_thresh and umap2_thresh
    
    """

    #get the spectrograms
    temp = frame[(frame[umap1_name] > umap1_thresh[0]) & (frame[umap1_name] < umap1_thresh[1]) & (frame[umap2_name] > umap2_thresh[0]) & (frame[umap2_name] < umap2_thresh[1])]
    
    if 'label' in temp.columns:
        avg_spec_raw = np.array(temp.drop(columns = ['label', 'source_file', umap1_name, umap2_name]))
    else:
        avg_spec_raw = np.array(temp.drop(columns = ['source_file', umap1_name, umap2_name]))
    
    #get the average
    avg_spec_image = np.mean(avg_spec_raw, axis=0).reshape((num_freq_bins, num_time_bins))
    
    #return the individual spectrograms
    return temp, avg_spec_image

def ava_get_spec(audio, p):
	"""
    From https://autoencoded-vocal-analysis.readthedocs.io/en/latest/_modules/ava/segmenting/utils.html?highlight=get_spec#
    Get a spectrogram. Much simpler than ``ava.preprocessing.utils.get_spec``.

	Arguments: 
	   audio (Audio): numpy array of floats
	   p (dict): Spectrogram parameters. Should the following keys: `'fs'`, `'nperseg'`,`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
		`'spec_max_val'`

	Returns:
        spec (numpy array of floats): Spectrogram of shape [freq_bins x time_bins]
        dt (float): Time step between time bins.
        f (numpy.ndarray): Array of frequencies.
	"""
    
	#get log spectrograms between min_freq and max_fre1
	assert len(audio) >= p['nperseg'], "len(audio): " + str(len(audio)) + ", nperseg: " + str(p['nperseg'])
	f, t, spec = stft(audio, fs=p['fs'], nperseg=p['nperseg'], noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	f, spec = f[i1:i2], spec[i1:i2]
	spec = np.log(np.abs(spec))

	#apply thresholds and scale
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0], f
