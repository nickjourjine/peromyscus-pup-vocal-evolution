import os
import pandas as pd
from datetime import date, datetime


path_to_warbleR_extract = '/n/hoekstra_lab_tier1/Users/njourjine/manuscript/notebooks/00_manuscript/warbleR_extract.R'

def write_warbleR_job_scripts(dataset, save_root, wav_root, script_dir):
    """
    Write sbatch job files to run warbleR_feature_extraction.R on a computing cluster. 
    
    Required processing steps:
        1. You have a csv of all pups to process with a column called species, which will be used to group the features into directories
        2. You have a directory containing one wav clip for every vocalization in the above csv (no subdirectories)
    

    Parameters
    ----------
    dataset (str): one of ['bw_po_cf', 'bw_po_f1', 'bw_po_f2', development] (cross foster, F1, F2, development)
    
    save_root (str): the place where csv of acoustic features will be saved
    
    wav_root (str): the place containing the wav files (one per vocalization) to get features from
    
    script_dir (str): the place to save the sbatch scripts (one per species)
    
    Returns
    -------
    None

    """
    
    
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
        
	
	

