#filesystem
import os
import glob

#data
import numpy as np
import pandas as pd

def gaussian_filter_1d(size,sigma):
    """
    Return a 1D gaussian filter for smoothing trajectories.
    
    Arguments:
        size (int): filter will range from -int(size/2) to int(size/2),size
        sigma (float): sigma value for filter
        
    Returns:
        gaussian_filter (list): the gaussian filter 
    
    """
    
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    
    return gaussian_filter

def calc_distance(X1,Y1,X2,Y2):
    """
    Calculate the distance between two points (eg dam and speaker)
    
    Arguments:
        X1 (float or array of float): x coordinate of point 1
        X2 (float or array of float): y coordinate of point 1
        Y1 (float or array of float): x coordinate of point 2
        Y2 (float or array of float): y coordinate of point 2
        
        
    Returns:
        distance (float): the distance between the points 
    
    """
    
    distance= np.sqrt(np.square(X1-X2) + np.square(Y1-Y2)) 
    return distance

def calc_speed(distanceXY, framerate):
    
    """
    Calculate the speed of the dam between consecutive tracking timepoints
    
    Arguments:
        distanceXY (numoy array): array of dams location (x, y coordinates) 
        framerate (float): camera framerate
        
    Returns:
        speed (float): the speed of the dam
    
    """
    
    speed= np.sqrt(np.square(np.diff(distanceXY[:,0])) + np.square(np.diff(distanceXY[:,1])))/(1/framerate)
    return speed
def get_data(path_to_data, save, save_dir):
    """
    Aggregate all playback data into a single dataframe
    
    Arguments:
        path_to_data (list): the directory containing the data with one subdirectort per dam. Same as path_to_data in Analyze Playback.ipynb
        save (bool): whether or not to save the dataframe as a csv
        save_dir(str): path to the directory to save the dataframe as a csv if save is true
        
    Returns:
        playback_df (dataframe): a data frame with all data for analyzing playback experiments
    
    """

    #get dams whose data you will aggregate
    SessionsToRun = [f for f in os.listdir(path_to_data) if f.startswith('BW')]

    #initialize list to collect data
    playback_df=[]
    
    #print some useful info
    print('Aggregating data from dams (species_ID_yyyy_m_dd):\n')
    for index, dam in enumerate(SessionsToRun): print('\t', dam)

    #get the data
    for Example in SessionsToRun:

        #get metadata
        Species=Example[0:2]
        Sex=Example[3]
        Identity=Example[5:10]
        Date=Example[11:]

        #Grab from Folder the files for Centroid and TTL
        ExampleSessionCentroid=  [f for f in os.listdir(path_to_data+Example) if f.startswith('centroid')] 
        ExampleSessionTTL= [f for f in os.listdir(path_to_data+Example) if f.startswith('Playback_TTL')] 

        #Load Tracking(centroid) and TTL
        tracking=np.genfromtxt([path_to_data+Example+ '/'  +ExampleSessionCentroid[0]][0], delimiter=',')
        TTL=np.genfromtxt([path_to_data+Example+ '/' +ExampleSessionTTL[0]][0], delimiter=',')

        #Rescaling from pixels to cm
        Scale=8.8
        tracking=tracking/Scale

        #Check for equal size of files because Bonsai might miss a row at the end when closing.
        if np.size(tracking,0)!=np.size(TTL):
            minimum=np.min([np.size(tracking,0),np.size(TTL)])
            TTL=TTL[0:minimum]
            tracking=tracking[0:minimum,:]

        #Threshold TTL
        TTL[:]=TTL>1

        #Take consecutive difference between TTL's to find TTL onsets and offsets
        DiffTTL= np.diff(TTL)
        DiffTTL=np.append(DiffTTL,DiffTTL[-1])

        #Analyze the TTL flips
        HighFlips=np.where(DiffTTL>0.5) #HighFlips is an index of the HighFlips

        #DiffFlip is the value of the Diff when the flip is high
        DiffFlips=np.diff(HighFlips)
        DiffFlips=np.append(DiffFlips[0],DiffFlips[0][-1]) #correcting for length

        #Define which TTL flips correspond to which vocalization type based on the InterFlip Interval (DiffFlip)
        CryFlips=( DiffFlips> 28) & (DiffFlips < 32)
        USVFlips=( DiffFlips> 88) & (DiffFlips < 92)
        EndFlips=( DiffFlips> 95)

        #Make a logical vector for when vocalizations are played
        allVector=np.arange(0,np.size(TTL,0),1)
        AllCry=np.interp(allVector, HighFlips[0], CryFlips)
        AllUSV=np.interp(allVector, HighFlips[0], USVFlips)

        #Clean before and after first and last High Flip
        AllUSV[allVector<HighFlips[0][0]]=False
        AllCry[allVector<HighFlips[0][0]]=False
        AllUSV[allVector>HighFlips[0][-1]]=False
        AllCry[allVector>HighFlips[0][-1]]=False

        #Select only the true Highs because of ramp artifact of interp
        AllCrybool=AllCry==1
        AllUSVbool=AllUSV==1

        #Remove jumps in tracking and interpolating tracking
        diffX=np.diff(tracking[:,0])
        diffY=np.diff(tracking[:,1])
        diffX=np.append(diffX,diffX[-1])
        diffY=np.append(diffY,diffY[-1])
        EraseX=np.where(np.abs(diffX)>1.3)
        EraseY=np.where(np.abs(diffY)>1.3)
        Erase=np.append(EraseX, EraseY)
        Erase=np.unique(Erase)

        #Erase outliers
        tracking[Erase,:] = np.NAN
        diffX=np.diff(tracking[:,0])
        diffY=np.diff(tracking[:,1])
        diffX=np.append(diffX,diffX[-1])
        diffY=np.append(diffY,diffY[-1])

        #Interpolate nans
        NaNIndexes=np.where(np.isnan(tracking[:,0]))
        NaNTracking=NaNIndexes
        tracking[NaNIndexes,0]=np.interp(NaNIndexes[0],np.where(~np.isnan(tracking[:,0]))[0],tracking[np.where(~np.isnan(tracking[:,0])),0][0])
        tracking[NaNIndexes,1]=np.interp(NaNIndexes[0],np.where(~np.isnan(tracking[:,1]))[0],tracking[np.where(~np.isnan(tracking[:,1])),1][0])

        # Smoothing X and Y with a gaussian filter
        sigma1 = 5
        Filter=gaussian_filter_1d(size=60,sigma=sigma1)
        tracking[:,0]=np.convolve(tracking[:,0], Filter, 'same')
        tracking[:,1]=np.convolve(tracking[:,1], Filter, 'same')

        #Gaussian filtering the speed and Calculating speed
        TimeVector=np.arange(0,0.0333*np.size(TTL,0),0.0333)
        Speedpre=calc_speed(tracking, 30)
        Speedpre=np.append(Speedpre,Speedpre[-1])
        sigma1 = 10
        Filter=gaussian_filter_1d(size=60,sigma=5)
        Speed=np.convolve(Speedpre, Filter, 'same')

        #Calculate Distances to Nest and Speaker
        DistanceMouseSpeaker=calc_distance(tracking[:,0], tracking[:,1], 5, 22.5)
        DistanceMouseNest=calc_distance(tracking[:,0], tracking[:,1], 31, 6)

        #Median of speeds During Cry and USV
        SpeedCryMedian=np.median(Speed[AllCrybool])
        SpeedUSVMedian=np.median(Speed[AllUSVbool])

        #90 percentile of speeds During Cry and USV
        SpeedCryMax=np.percentile(Speed[AllCrybool], 99)
        SpeedUSVMax=np.percentile(Speed[AllUSVbool], 99)   

        #Trajectories During Vocalization

        #Median Distance to Speaker
        MedianDistSpeakerCry=np.median(DistanceMouseSpeaker[AllCrybool][~np.isnan(DistanceMouseSpeaker[AllCrybool])])
        MedianDistSpeakerWhis=np.median(DistanceMouseSpeaker[AllUSVbool][~np.isnan(DistanceMouseSpeaker[AllUSVbool])])

        # Minimimum Distance to Speaker
        MinDistSpeakerCry=np.percentile(DistanceMouseSpeaker[AllCrybool][~np.isnan(DistanceMouseSpeaker[AllCrybool])], 1)
        MinDistSpeakerWhis=np.percentile(DistanceMouseSpeaker[AllUSVbool][~np.isnan(DistanceMouseSpeaker[AllUSVbool])],1)


        #Calculate Time of response During Cry and USV

        #Indexes where TTL boolean of Cries and USVs goes up or down
        UBoolFlipsup=np.where(np.append(np.diff(1*AllUSVbool), np.diff(1*AllUSVbool)[-1])==1)
        UBoolFlipsdn=np.where(np.append(np.diff(1*(AllUSVbool)), np.diff(1*AllUSVbool)[-1])==-1)
        CBoolFlipsup=np.where(np.append(np.diff(1*AllCrybool), np.diff(1*AllCrybool)[-1])==1)
        CBoolFlipsdn=np.where(np.append(np.diff(1*(AllCrybool)), np.diff(1*AllCrybool)[-1])==-1)

        #Assign a time based on Timevector to those flips indexes
        TimeWFlipUp=TimeVector[UBoolFlipsup]
        TimeWFlipDown=TimeVector[UBoolFlipsdn]
        TimeCFlipUp=TimeVector[CBoolFlipsup]
        TimeCFlipDown=TimeVector[CBoolFlipsdn]

        #Define End of Experiment
        ExtraFrames=300
        if Identity=="31922":
            EndOfExperiment=np.max([ CBoolFlipsdn[0][4], UBoolFlipsdn[0][3]]) +ExtraFrames
        else:
            EndOfExperiment=np.max([ CBoolFlipsdn[0][4], UBoolFlipsdn[0][4]])+ExtraFrames

        #Calculate the Time when the TTL is on for Cries and USVs, only for the first 5 trials of each which correspond to the actual experiment
        TimeCries=TimeCFlipDown[0:5]-TimeCFlipUp[0:5]
        TimeUSVs=TimeWFlipDown[0:5]-TimeWFlipUp[0:5]

        #Median Time per animal per Cry vs USVs
        MedianResponseTimeCry=np.median(TimeCries)
        MedianResponseTimeUSV=np.median(TimeUSVs)

        #Aggregate the data for this iteration of the loop
        playback_df.append({ "species" : Species,
                            "sex" : Sex,
                            "id" : Identity,
                            "date":Date, 
                            "median_time_cry":MedianResponseTimeCry,
                            "median_time_USV": MedianResponseTimeUSV,
                            "max_speed_cry": SpeedCryMax,
                            "max_speed_USV": SpeedUSVMax,
                            "min_dist_cry":MinDistSpeakerCry,
                            "min_dist_USV":MinDistSpeakerWhis,
                            "median_dist_cry":MedianDistSpeakerCry,
                            "median_dist_USV":MedianDistSpeakerWhis,
                            "distance_to_speaker": DistanceMouseSpeaker,
                            "speed": Speed,
                            "nan_tracking": NaNTracking, 
                            "time_vector": TimeVector,
                            "all_cry_bool": AllCrybool,
                            "all_USV_bool": AllUSVbool,
                            "index_onset_cry":CBoolFlipsup,
                            "index_onset_USV":UBoolFlipsup,
                            "end_experiment": EndOfExperiment

        })

    #make the dataframe for all dams
    playback_df = pd.DataFrame(playback_df)  

    #save dataframe
    if save:
        playback_df.to_csv(os.path.join(save_dir, save_name), index=False)
    
    print('done.')
        
    return playback_df
def get_heatmaps(playback_df, feature):
    """
    Get matrices of dam feature (eg distance to speaker) where each row is a trial and each column is a time point.
    
    Arguments:
        playback_df (dataframe): output of get_data
        feature (str): the feature of interest - must be in the columns of playback df
        
    Returns:
        CryMatrix (numpy array): matrix for cry responses
        USVMatrix (numpy array): matrix for USV responses
    
    """
    
    #check inputs
    assert isinstance(playback_df, pd.core.frame.DataFrame), "playback_df must be a pandas dataframe"
    assert feature in playback_df.columns, "feature must be one of the column labels of playback_df"
    
    # hard code information that is the same for all playback trials
    start=0
    TotalTime=3900
    PreTime=300
    iterations=25
    
    #initialize matrices
    CryMatrix=np.zeros((iterations,TotalTime))
    USVMatrix=np.zeros((24,TotalTime))
    
    #get CryMatrix
    counter=0
    for i in playback_df["id"]:
        iteraFrames=playback_df.loc[playback_df["id"]==i, "index_onset_cry"].to_numpy()[0]
        EndOfExp=playback_df.loc[playback_df["id"]==i, "end_experiment"].item()
        iteraFrames2=iteraFrames[0][np.where(iteraFrames[0]<EndOfExp)[0]]

        for ii in iteraFrames2:
            timevector=np.arange(-10,120,1/30)
            CryMatrix[counter,:]=np.reshape(playback_df.loc[playback_df["id"]==i, "distance_to_speaker"].to_numpy()[0][ii-300:ii+3600], TotalTime)
            counter=counter+1

    #get USVMatrix
    counter=0
    for i in playback_df["id"]:
        iteraFrames=playback_df.loc[playback_df["id"]==i, "index_onset_USV"].to_numpy()[0]
        EndOfExp=playback_df.loc[playback_df["id"]==i, "end_experiment"].item()
        iteraFrames2=iteraFrames[0][np.where(iteraFrames[0]<EndOfExp)[0]]

        for ii in iteraFrames2:
            timevector=np.arange(-10,120,1/30)
            USVMatrix[counter,:]=np.reshape(playback_df.loc[playback_df["id"]==i, "distance_to_speaker"].to_numpy()[0][ii-300:ii+3600], TotalTime)
            counter=counter+1
            
    return CryMatrix, USVMatrix 