#this file contains functions for writing and reading parameters and dealing with directories and 
#dataframes for keeping track of different parameter choices for a given analysis


#filesystem
import os

#data
import json
import pandas as pd
from datetime import date, datetime

def get_date_time():
    """
    Uses datetime to return a string with the format CurrentDate_CurrentTime,  e.g. 20220920_120000
    useful for naming directories

    Arguments:
        None

    Returns:
        The date and time as a string in the format 'yyyymmdd_hhmmss'

    """
    
    
    
    #get the date
    current_date = str(date.today())
    current_date = ('').join(current_date.split('-'))
    
    #get the time
    current_time = str(datetime.now())
    current_time = current_time.split(' ')[-1]
    current_time = ('').join(current_time.split('.')[0].split(':'))
    
    return ('_').join([current_date,current_time])

def save(params, save_dir, save_name):
    """  
    Save a dictionary as .json and double check if you want to overwrite it.

    Arguments:
        params_dict (dict): the parametes dictionary to be saved
        save_dir (str): the path to the place where the dictionary file will be saved
        save_name (str): the name of the file without any file extension

    Returns:
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
	
	


    
    
    
def load(save_dir, save_name):
	"""
	Load a dictionary from .json - does the reverse of save_parameters()

	Arguments:
	   save_dir (str): the path to the place where dictionary was saved
	   save_name (str): the name of the .json file (including file extension)
	
	Returns:
	   params_dict (dict): the params dictionary you saved

	"""
	
	save_path = os.path.join(save_dir,save_name)
	
	with open(save_path+'.json', 'r') as fp:
			params_dict = json.load(fp)
	print('loaded parameters from:\n\t', save_path)
	return params_dict

def combine_dataframes(source_dir, save_dir, save_name, file_format, include_string, paths_list=None):

	"""
	Combine multiple csvs from a single directory (default) or list of paths into one. Useful for combining across species
    or treatments while keeping raw data csvs intact.
    
	Arguments:
    
        paths_list (list): a list of full paths to the csvs to be combined
        source_dir (string): the path to the directory containing the annotation csvs to be combined
        save_dir (string): the path to the directory where the combined csv will be saved
        save_name (string): name of the combined csv to be saved
        file_format (string): '.csv' or '.feather'
        include_sting (string): only combine files with this string in their name

	Returns
	-------
	   all_files (dataframe): the combined dataframe

	"""
	assert file_format in ['.csv', '.wav', '.feather']
    
	if paths_list == None and source_dir != None:
		sources = [source_dir+i for i in os.listdir(source_dir) if i.endswith(file_format) and not i.startswith('.') and include_string in i]
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
		all_files.to_csv(save_dir+save_name+'.csv', index=False)
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
		all_files.to_feather(save_dir+save_name+'.feather')
		print('saved the combined dataframe to', save_dir+save_name+file_format)
		return all_files




