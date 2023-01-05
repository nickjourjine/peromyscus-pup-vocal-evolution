# this file contains basic utility functions for dealing with manuscript data


import glob
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp2d
from scipy.signal import stft
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil	
import seaborn as sns 
import json
import pickle


def get_date_time():
    """
    uses datetime to return a string with the format CurrentDate_CurrentTime,  e.g. 20220920_120000
    useful for naming directories

    Parameters
    ----------
    None

    Return
    ------
    The date and time as a string, e.g. 20220920_120000

    """
    
    from datetime import date, datetime
    
    #get the date
    current_date = str(date.today())
    current_date = ('').join(current_date.split('-'))
    
    #get the time
    current_time = str(datetime.now())
    current_time = current_time.split(' ')[-1]
    current_time = ('').join(current_time.split('.')[0].split(':'))
    
    return ('_').join([current_date,current_time])
def combine_dataframes(source_dir, save_dir, save_name, file_format, include_string, exclude_string, paths_list=None):

	"""
	combine multiple csvs from a single directory (default) or list of paths into one. This should replace combine_annotation and combine_prediction.

	Parameters
	----------
	paths_list (list): a list of full paths to the csvs to be combined
	
	source_dir (string): the path to the directory containing the annotation csvs to be combined

	save_dir (string): the path to the directory where the combined csv will be saved
	
	save_name (string): name of the combined csv to be saved
	
	file_format (string): '.csv' or '.feather'
	
	include_sting (string): only combine files with this string in their name
	
	exclude_string (string): ignore files with this in their name

	Returns
	-------
	all_files (dataframe): the combined dataframe

	"""
	assert file_format in ['.csv', '.wav']
    
	if paths_list == None and source_dir != None:
		sources = [os.path.join(source_dir,i) for i in os.listdir(source_dir) if i.endswith(file_format) and exclude_string not in i and not i.startswith('.') and include_string in i]
		combined = []
		
	elif paths_list != None and source_dir == None:
		sources=paths_list
		combined=[]
        
    
		
	elif paths_list == None and source_dir == None:
		print('provide either a list of paths or a directory containing all the files to be combined')
		
	elif paths_list != None and source_dir != None:
		print('provide either a list of paths or a directory containing all the files to be combined, not both')
	
	if file_format == '.csv':

		for i in sources:
			temp = pd.read_csv(i)
			if len(i) != 0:
				combined.append(temp)
			else:
				print(i, 'has no vocalizations')

		all_files = pd.concat(combined)
		all_files.to_csv(os.path.join(save_dir,save_name)+'.csv', index=False)
		print('saved the combined dataframe to',save_dir+save_name+file_format)
		return all_files
	
	elif file_format == '.feather':

		for i in sources:
			temp = pd.read_feather(i)
			if len(i) != 0:
				combined.append(temp)
			else:
				print(i, 'has no vocalizations')

		all_files = pd.concat(combined)
		all_files = all_files.reset_index(drop=True)
		all_files.to_feather(os.path.join(save_dir,save_name)+'.feather')
		print('saved the combined dataframe to', save_dir+save_name+file_format)
		return all_files
			


def save_parameters(params, save_dir, save_name):
    """
    NOTE this is a copy of the function in segmentation.py
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
	
# def aggregate_das_predictions(save_dir, predictions_dir, model_dict, iteration_dict):
# """
# 	Collect predictions csvs from different species and das models into a single csv for evaluation using combine dataframes
# 
# 	Parameters
# 	----------
# 	save_dir (str): full path to the directory where the combined csv will be saved
# 	
# 	predictions_dir (str): the path to the directory containing all the predictions directories for each species, their different models and iterations of prediction from those models
# 
# 	model_dict (dict): dictionary where the keys are species and the values are the models
# 	
# 	iteration_dict (dict): dictionary where the keys are species and the values are the iteration of each model you want to combine
# 
# 	Returns
# 	-------
# 	a dataframe of predictions for all species with a column for model and iteration
# 	also saves this dataframe as a csv to save_dir
# 
# 	"""
# 
# 	
# 
# 
# 
# 	

### functions related to checking and getting info from file names ###
###################################################################################################################################################
def trim_channel(directory):
	"""
	remove the channel prefix from raw wav file names that gets added by avisoft

	some recordings have the channel number pre-pended to the start of the file name. This function checks that this matches
	the file name information in the interior of the file name, then removes the pre-pended channel
	If there are mismatches the code stops and returns a list of mismatches
	In this case the interior channel number should be modified to match the pre-pended channel, which is added automatically by avisoft

	Parameters
	----------
	directory (string): path to the directory containing all of the raw wav files

	Returns
	-------
	None

	"""

	files = os.listdir(directory)
	to_modify = [i for i in files if i.startswith('ch')]
	mismatch = []

	#first check that all the files contain channel information in the expected locations
	for file in to_modify:

		if file.split('_')[0] != [i for i in file.split('_')[1:] if 'ch' in i][0]:
			mismatch.append(file)

	problem = False
	#if channels don't match within any filename, break and return those names
	if len(mismatch) != 0:
		problem = True
		print('The returned files have a mismatch in their channel numbers.')
		return mismatch

	#otherwise trim and rename (note channel numbers are 1-8, so all files with prepended channels look lke 'ch#_' at the start, 4 characters to trim)
	elif not problem:
		for file in to_modify:
			new_name = file[4:]
			os.rename(directory+file, directory+new_name)
	
	
def prepend(root, file, prepend):
	"""
	add a string to the start of a file name - can be useful for renaming so categories in filename match

	Parameters
	----------
	root (string): path to the directory containing all of the raw wav files whose names you want to change
	file (string): the name of the file
	prepend (string): the string to prepend

	Returns
	-------
	None
	"""

	if file.startswith('box'):
		os.rename(root+file, root+prepend+file)


def check_file_names(directory):
	"""
	check that file names follow the naming convention used by other functions

	Parameters
	----------
	directory (string): path to the directory containing all of the raw wav files

	Returns
	-------
	bad_lengths (list): a list of file names that don't have the right number of items (each separated by an underscore)
	wrong_order (list): a list of file names that have the correct number of it but one of them is not what it should be, the location that has incorrect information


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

	### functions for handling wav file clips ###
	########################################################################################################################################################################

	#take a csv of features generated by the old pipeline and extract start, stop, and labels - can be useful for re-segmenting by species
def segments_from_csv(source_csv, save_dir, species, save=True):

	"""
	take a csv of features generated by the old pipeline and extract start, stop, and labels - can be useful for re-segmenting by species

	Parameters
	----------
	source_csv (string): path to the csv of features

	Returns
	-------
	df (dataframe): a dataframe with columns for species, start time, stop time, voc_name (old svm prediction), and source file


	"""

	#read the data
	df = pd.read_csv(source_csv)

	#drop scratches
	#df = df.drop(df.loc[df['voc_name'] == 'scratch'].index)

	#subset by species
	df = df.loc[df['species'] == species]

	#get the info you want
	df = df[['species','start_time', 'end_time','voc_name', 'source_file']]
	df['source_file'] = [i+'.wav' for i in df['source_file'] if not i.endswith('.wav')]
	df = df.dropna(subset = ['start_time']) #drop non-vocal animals

	#save
	if save:
		df.to_csv(save_dir+species+'predictions_from_draft_segmentation_02-2022.csv', index = False)
	else:
		#or just return the dataframe
		return df 	

def copy_wavs(source, destination, species = None):
	#copy raw wav files into the correct folders in the ava directory structure
	#source is the location of the directroy containing the wav files to be moved
	#destination is where they are going
	#optionally just move wavs that start with species

	if species != None:
		source_paths = [source+i for i in os.listdir(source) if i.startswith(species) and not 'scratch' in i]
		destination_paths = [destination+i.split('/')[-1] for i in source_paths]
	else:
		source_paths = [i for i in os.listdir(source) if not i.startswith('.') and not i.startswith('00')]
		destination_paths = [destination+i for i in os.listdir(source) if not i.startswith('.') and not i.startswith('00')]

	already_existed = []
	print('preparing to copy', len(source_paths) - len(os.listdir(destination)), 'files...')
	for source, destination in tqdm(zip(source_paths, destination_paths)):

		#print('destination:', destination)
		if os.path.exists(destination):
			already_existed.append(source)
			continue
		else: 
			print('source:', source)
			shutil.copy2(source, destination) 
	print(len(source_paths) - len(already_existed), 'wavs were copied')            
	# 	print(len(already_existed), "files were already copied and were ignored")

	### functions for handling csv files ###
	########################################################################################################################################################################

	#collect csvs that have been made separately for each species and combine them into one


	### functions for handling features ###
	#########################################################################################################################################################


	#add columns metadata contained in source file names
	#input is a dataframe with a column called source_file
	#example file name for clips BK_24224x25894_ltr1_pup1_ch2_3700_m_358_302_fr0_p5_2021-10-22_11-05-10_clip_0_scratch.wav

def get_metadata(frame):

	columns = frame.columns

	if 'species' in columns:
		print('species column already exists...')
	else:
		frame['species'] = [i[:2] for i in frame['source_file']]

	if 'sex' in columns:
		print('sex column already exists...')
	else:
		frame['sex'] = [i.split('_')[6] for i in frame['source_file']]

	if 'age' in columns or 'age_in_days' in columns:
		print('age column already exists...')
	else:
		frame['age'] = [int(i.split('_')[10][1:]) for i in frame['source_file']]

	if 'individual' in columns:
		print('individual column already exists...')
	else:
		frame['individual'] = [i.split('_clip')[0] for i in frame['source_file']]

	if 'weight' in columns or 'weight_mg' in columns or 'weight_g' in columns:
		print('weight column already exists...')
	else:
		frame['weight_mg'] = [i.split('_')[5] for i in frame['source_file']]

	if 'removal_flag' in columns:
		print('removal flag column already exists...')
	else:
		frame['removal_flag'] = [int(i.split('_')[9][-1]) if not i.split('_')[9] == 'nan' else 'nan' for i in frame['source_file']]

	if 'label' in columns:
		print('label column already exists...')
	else:
		frame['label'] = [i.split('_')[-1].split('.')[0] if i.split('_')[-1].split('.')[0] in ['cry', 'whistle', 'scratch'] else 'nan' for i in frame['source_file']]

	return frame

def get_pup_features(frame):
	by_pup = frame
	return by_pup
