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

#definitely keep - documented now on evernote 20230105
def get_spectrogram(data, 
                    fs, 
                    nperseg, 
                    noverlap, 
                    num_freq_bins, 
                    num_time_bins, 
                    min_freq, 
                    max_freq, 
                    fill_value, 
                    max_dur, 
                    spec_min_val, 
                    spec_max_val): 
    
	#get the spectrogram
	f,t,specgram = stft(data, fs, nperseg=nperseg, noverlap=noverlap) #default winow is Hann

	#define the target frequencies and times for interpolation
	duration = np.max(t)
	target_freqs = np.linspace(min_freq, max_freq, num_freq_bins)
	shoulder = 0.5 * (max_dur - duration)
	target_times = np.linspace(0 - shoulder, duration+shoulder, num_time_bins)

	#make it pretty
	specgram = np.log(np.abs(specgram)+ 1e-12) # make a log spectrogram 
	interp = interp2d(t, f, specgram, copy=False, bounds_error=False, fill_value=fill_value) #define the interpolation object - what does this do an do you need it??
	target_times = np.linspace(0 - shoulder, duration+shoulder, num_time_bins) #define the time axis of the spectrogram
	interp_spec = interp(target_times, target_freqs, assume_sorted=True) #interpolate -- you need to do this to interpolate the spectrogram values at the 64 frequency and 64 time points you have specified (ie, linearly sampled bw the min and max of time and frequency)
	specgram = interp_spec
	specgram -= spec_min_val #normalize
	specgram /= (spec_max_val - spec_min_val) #normalize
	specgram = np.clip(specgram, 0.0, 1.0) #clip

	return f,t,specgram





def wavs_to_umap(segmenting_option, clips_dir, noise_floors_path, species, noise_floor, spec_params, num_to_process, filtered_clips, interpolate, version, save_root):
#go throught the full pipeline from spectrogram generation to umap plotting
	
		
	#get spectrograms
	specs_list, source_files = specs_from_wavs(clips_dir = clips_dir, 
										  noise_floors_path=noise_floors_path, 
										  species = species,
										  noise_floor = noise_floor,
										  spec_params=spec_params, 
										  num_to_process = num_to_process, 
										  filtered_clips = filtered_clips, 
										  interpolate=interpolate)
										  
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
	
	#TODO: get this to work
	# if coordinates_save_name not in coordinates_save_dir:
# 		save_umap(df_umap, umap1, umap2, save_name = save_name, file_format = file_format)
# 	else:
# 		print('already processed...')



#modified from https://github.com/timsainb/avgn_paper/blob/V2/avgn/signalprocessing/create_spectrogram_dataset.py





def specs_from_wavs(clips_dir, species, filtered_clips, noise_floors_path, noise_floor, spec_params, num_to_process, include_scratch = True, interpolate=False):

#TODO
#document this
#make the variable names especially 'filtered_clips' more intuitive
    specs_list = []
    source_files = []
    nans = [] #what is this line doing? TODO remove it

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

        #get the pretty spectrogram 
        if not interpolate:  

            #bandpass the audio 
            wav = butter_bandpass_filter(wav, 
                                         spec_params['min_freq'], 
                                         spec_params['max_freq'], 
                                         spec_params['fs'], 
                                         order=1)	

            #get the spectrogram
            f, t, spec = scipy_specgram(wav, 
                                  fs=spec_params['fs'], 
                                  nperseg=spec_params['nperseg'], 
                                  noverlap=spec_params['noverlap'], 
                                  thresh=noise_floor, 
                                  scaling_factor = spec_params['log_resize_scaling_factor']) 

            #add downsampled version to specs
            downsampled_spec = spec[:,::spec_params['downsample_by']]
            specs_list.append(downsampled_spec) #downsaple time and frequency 

        elif interpolate:

                #if using this method with noise thresholds, fill value should be set to the recoridng specific threshold
                if noise_floors_path != None:
                    spec_params['fill_value'] = noise_floor

                f, t, spec = get_spectrogram(data = wav,
                  fs=spec_params['fs'],
                  nperseg=spec_params['nperseg'],
                  noverlap=spec_params['noverlap'],
                  num_freq_bins=spec_params['num_freq_bins'],
                  num_time_bins=spec_params['num_time_bins'] ,
                  min_freq=spec_params['min_freq'],
                  max_freq=spec_params['max_freq'],
                  fill_value = spec_params['fill_value'],
                  max_dur=spec_params['max_duration'],
                  spec_min_val=noise_floor, 
                  spec_max_val=spec_params['spec_max_val'])

                specs_list.append(spec) #downsaple time and frequency


    #pad
    if not interpolate:
            print('padding spectrograms...')
            syll_lens = [np.shape(i)[1] for i in specs_list]
            pad_length = np.max(syll_lens)
            specs_list = [pad_spectrogram(i, pad_length) for i in specs_list]

    print('done.')
    return specs_list, source_files



#make spectrograms from a segments dataframe without saving wav clips first - useful for trouble shooting (faster)









def linearize_specs(specs_list):
	
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

#z score linearized spectrograms and return a dataframe of the zscored specs and original specs
def zscore_specs(specs_lin, source_files):
	#make a dataframe
	df_umap = pd.DataFrame(specs_lin)
	df_umap['source_file'] = source_files
	
	# Z-score the spectrogams
	print('z scoring...')
	zscored = StandardScaler().fit_transform(df_umap.drop(columns=['source_file']))	
	print('done.')
	return df_umap, zscored

#take a data frame of zscored spectrograms (no source_file information) and find a umap embedding
#return umap 1 and umap 2 coordinates as separate arrays
def get_umap(standardized_features):
	
	#find an embedding
	print('finding a umap embedding...')
	
	start_time = time.time()
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(standardized_features)
	print('embedding took', time.time() - start_time, 'seconds.')
	
	umap1 = embedding[:, 0]
	umap2 = embedding[:, 1]

	return umap1, umap2

#save a dataframe of all spectrograms and their umap coordinates






#probably keep and make cells to use them in Segmenting and UMAP.ipynb
def save_umap(df_umap, umap1, umap2, save_name = None, file_format = 'feather'):
	
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

		
# make an aggregate spectrogram from  list of source files (wav clips)

def spec_avg_from_list(spec_list):
    #get average
    avg_spec_image = np.mean(spec_list, axis=0)
        
    return avg_spec_image
     
    
def show_specs(frame, num_freq_bins, num_time_bins, columns_to_drop):
	for i in range(len(frame)):
		print(frame['source_file'].iloc[i])
		to_plot = frame.drop(columns = columns_to_drop)
		img = to_plot.iloc[i]
		img = np.array(img).reshape((num_freq_bins, num_time_bins))
		plt.imshow(img, origin = 'lower', extent = (num_freq_bins, 0, num_time_bins, 0 ))
		plt.show()

def files_from_umap(frame, umap1_name, umap2_name, umap1_thresh, umap2_thresh, source_dir):

    #get the spectrograms
    temp = frame[(frame[umap1_name] > umap1_thresh[0]) & (frame[umap1_name] < umap1_thresh[1]) & (frame[umap2_name] > umap2_thresh[0]) & (frame[umap2_name] < umap2_thresh[1])]
    source_files = [source_dir+i for i in temp['source_file']]
    return source_files

def spec_avg_from_umap(frame, umap1_name, umap2_name, umap1_thresh, umap2_thresh, num_freq_bins, num_time_bins):

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
#TODO - properly comment this. Looks like segmenting_option doesn't do anything

	
#probably remove   
def log_resize_spec(spec, scaling_factor):

    resize_shape = [int(np.log(np.shape(spec)[1]) * scaling_factor), np.shape(spec)[0]]
    spec = (spec * 255).astype(np.float32) #make data type and range compatible with Image
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape))
    return resize_spec

def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )

def butter_bandpass(lowcut, highcut, fs, order=5):
    #functions to bandpass the audio 
    #modified from https://timsainburg.com/python-mel-compression-inversion.html#python-mel-compression-inversion
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    #functions to bandpass the audio 
    #modified from https://timsainburg.com/python-mel-compression-inversion.html#python-mel-compression-inversion
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#function to generate spectrograms
def scipy_specgram(data, 
                   fs, 
                   nperseg, 
                   noverlap, 
                   thresh, 
                   scaling_factor = None):
    
	#get the spectrogram
	f,t,specgram = stft(data, fs, nperseg=nperseg, noverlap=noverlap) #default winow is Hann

	#make it pretty
	specgram = np.abs(specgram) #remove complex
	specgram = np.log(specgram)  # take log

	if scaling_factor!= None:
		specgram = log_resize_spec(specgram, scaling_factor= scaling_factor)    
		specgram[specgram < thresh] = thresh
		specgram = (specgram - np.min(specgram)) / (np.max(specgram) - np.min(specgram))

	else: 
		specgram[specgram < thresh] = thresh
		specgram = (specgram - np.min(specgram)) / (np.max(specgram) - np.min(specgram))
		
	return f,t,specgram

#function to generate spectrogram using interpolation a la Goffinet 2021	



def specs_from_segs(seg_df_path,raw_wavs_dir, spec_params, num_to_process, species=None, units= 's', interpolate=False):

	#list to add the spectrograms to
	specs_list = []

	#get the dataframe with the start and stop times (segments)
	seg_df = pd.read_csv(seg_df_path)
	
	#optionally subset by species
	if species != None:
		seg_df = seg_df.loc[seg_df['species'] == species]
		
	#get the names of the raw audio files to be segmented
	source_files = [raw_wavs_dir+i for i in seg_df['source_file'].unique()]
	
	#get the segments
	segments_list = [[i,j] for i,j in zip(seg_df['start_seconds'], seg_df['stop_seconds'])]
	print('ready to make', len(segments_list), 'spectrograms from', len(source_files), 'recordings')
	
	for file in source_files:
	
		#get the information just for this file
		temp = seg_df.loc[seg_df['source_file'] == file]
	
		#get the audio
		fs, wav = wavfile.read(file)
		
		#get the segments
		segments_list = [[i,j] for i,j in zip(temp['start_seconds'], temp['stop_seconds'])]
		
		#get the spectrograms
		for segment in segments_list:
			#get the wav clip
			if units == 's':
				clip = wav[fs*segment[0]:fs*segment[1]]
			elif units == 'ms':
				clip = wav[fs*(segment[0]/1000):fs*(segment[1]/1000)]
				
			#make a spectrogram
			if not interpolate:  

				#bandpass the audio 
				wav = butter_bandpass_filter(wav, 
											 spec_params['min_freq'], 
											 spec_params['max_freq'], 
											 spec_params['fs'], 
											 order=1)	
											
				#get the spectrogram
				f, t, spec = scipy_specgram(wav, 
									  fs=spec_params['fs'], 
									  nperseg=spec_params['nperseg'], 
									  noverlap=spec_params['noverlap'], 
									  thresh=noise_floor, 
									  scaling_factor = spec_params['log_resize_scaling_factor']) 
			
				#add downsampled version to specs
				downsampled_spec = spec[:,::spec_params['downsample_by']]
				specs_list.append(downsampled_spec) #downsaple time and frequency 

			elif interpolate:
					f, t, spec = get_spectrogram(data = wav,
					  fs=spec_params['fs'],
					  nperseg=spec_params['nperseg'],
					  noverlap=spec_params['noverlap'],
					  num_freq_bins=spec_params['num_freq_bins'],
					  num_time_bins=spec_params['num_time_bins'] ,
					  min_freq=spec_params['min_freq'],
					  max_freq=spec_params['max_freq'],
					  fill_value = spec_params['fill_value'],
					  max_dur=spec_params['max_duration'],
					  spec_min_val=noise_floor, 
					  spec_max_val=spec_params['spec_max_val'])

					specs_list.append(spec) #downsaple time and frequency
		
	#pad the spectrograms (only needed if interpolate=False)
	if not interpolate:
			print('padding spectrograms...')
			syll_lens = [np.shape(i)[1] for i in specs_list]
			pad_length = np.max(syll_lens)
			specs_list = [pad_spectrogram(i, pad_length) for i in specs_list]

	print('done.')
	return specs_list, source_files

#linearize spectrograms for umap
def get_spec(audio, p):
	"""
	Get a spectrogram.

	Much simpler than ``ava.preprocessing.utils.get_spec``.

	Raises
	------
	- ``AssertionError`` if ``len(audio) < p['nperseg']``.

	Parameters
	----------
	audio : numpy array of floats
		Audio
	p : dict
		Spectrogram parameters. Should the following keys: `'fs'`, `'nperseg'`,
		`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
		`'spec_max_val'`

	Returns
	-------
	spec : numpy array of floats
		Spectrogram of shape [freq_bins x time_bins]
	dt : float
		Time step between time bins.
	f : numpy.ndarray
		Array of frequencies.
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
    

	

	
	
