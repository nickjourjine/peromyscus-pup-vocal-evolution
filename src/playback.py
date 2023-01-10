import numpy as np

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

def CalcDistance(X1,Y1,X2,Y2):
    """
    Calculate the distance between two points
    
    Arguments:
        X1 (float): x coordinate of point 1
        X2 (float): y coordinate of point 1
        Y1 (float): x coordinate of point 2
        Y2 (float): y coordinate of point 2
        
        
    Returns:
        Distance (float): the distance between the points 
    
    """
    
    Distance= np.sqrt(np.square(X1-X2) + np.square(Y1-Y2)) 
    
    return Distance

def CalcSpeed(DistanceXY, Framerate):
    
    """
    Calculate the speed of the dam between consecutive tracking timepoints
    
    Arguments:
        DistanceXY (numoy array): array of dams location (x, y coordinates) 
        Framerate (float): camera framerate
        
    Returns:
        Speed (float): the speed of the dam
    
    """
    
    Speed= np.sqrt(np.square(np.diff(DistanceXY[:,0])) + np.square(np.diff(DistanceXY[:,1])))/(1/Framerate)
    return Speed
