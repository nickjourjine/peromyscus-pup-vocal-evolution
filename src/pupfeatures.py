#this file contains functions for aggregating acoustic features of vocalizations by pup

import os
import pandas as pd
import numpy as np



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
    Aggregate warbleR acoustic features by pup, get pup metdata, then combine them into a single dataframe

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
        
        
        
    
    
    
    
