#this file contains functions for writing and reading parameters 

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




