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

def CalcHeadDirectionEars(LeftX,LeftY,RightX, RightY):
    Angle = np.size(LeftX);
    # Angle(Tag > 0) = atan2(RightY-LeftY,RightX-LeftX)*180/pi ;
    Angle = np.arctan2(RightY-LeftY,RightX-LeftX) * 180 / np.pi ;
    return Angle

def CalcHeadDirectionHeadTail(HeadX,HeadY,TailX, TailY):
    dx = (TailX - HeadX)
    dy= (TailY - HeadY)
    return np.arctan2(dy, dx)* 180 / np.pi

def CalcEgocentricDirection(HeadX,HeadY,TailX, TailY, ObjectX, ObjectY):
    AngleEgoLoop=np.empty([1,len(HeadX)])
    for i in np.arange(0,len(HeadX)):
        dx = ( HeadX[i]- TailX[i] )
        dy= ( HeadY[i]- TailY[i])
        VectorAnimal=[dx,dy]/ np.linalg.norm([dx,dy])
        dx1 = ( ObjectX - TailX[i] )
        dy1= (ObjectY - TailY[i])
        VectorObject=[dx1,dy1]/ np.linalg.norm([dx1,dy1])
        dot_animal_object2 = np.dot(VectorAnimal, VectorObject)
        AngleEgoLoop[0,i] =np.arccos(dot_animal_object2)* 180 / np.pi
    return AngleEgoLoop

def CalcSpeed(DistanceXY, Framerate):
    Speed= np.sqrt(np.square(np.diff(DistanceXY[:,0])) + np.square(np.diff(DistanceXY[:,1])))/(1/Framerate)
    return Speed

def RescaleBodyparts(Bodyparts):
    
    #Extract Midspine
    
    CentroidX=Bodyparts[:,18]
    CentroidY=Bodyparts[:,19]
    ConfidenceCentroid=Bodyparts[:,20]
    CentroidConfidenceIndexes=np.where(ConfidenceCentroid>0.9999)

    AbsoluteMaxStretch=15
    
    #ScaleBodyparts
    ScaleX= (np.quantile(CentroidX[CentroidConfidenceIndexes],0.99)-np.quantile(CentroidX[CentroidConfidenceIndexes],0.01))/48.26
    ScaleY=(np.quantile(CentroidY[CentroidConfidenceIndexes],0.99)-np.quantile(CentroidY[CentroidConfidenceIndexes],0.01))/26
    ShiftX=np.quantile(CentroidX[CentroidConfidenceIndexes],0.01)
    ShiftY=np.quantile(CentroidY[CentroidConfidenceIndexes],0.01)
    print(f"scale x  {ScaleX}, Scale Y  {ScaleY}")
    figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    print(f'Centroid {CentroidX}')
    plt.subplot(2,2,3)
    plt.plot(CentroidX[CentroidConfidenceIndexes],CentroidY[CentroidConfidenceIndexes], '.k')
    
    Bodyparts[:,0:24:3]=(Bodyparts[:,0:24:3]-ShiftX)/ScaleX 
    Bodyparts[:,1:25:3]=(Bodyparts[:,1:25:3]-ShiftY)/ScaleY 
    return Bodyparts , ScaleX, ScaleY, ShiftX, ShiftY 
