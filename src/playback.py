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
    
    speed= np.sqrt(np.square(np.diff(DistanceXY[:,0])) + np.square(np.diff(DistanceXY[:,1])))/(1/Framerate)
    return speed
