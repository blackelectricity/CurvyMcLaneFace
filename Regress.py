# Regress.py

#EXTERNAL DEPENDENCIES
import cv2
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures

#This class isn't very clean, but it is what it is, just a quick class project
class Regress:

    def __init__( self ):
	#Reset also initializes lists and values for averaging/context information
        self.ResetCurveSmoothing()


    def ResetCurveSmoothing( self ):
	#empty out all the instance variables that are used for smoothing/context
        self.lastModel = None #last curve regression model
        self.lastCount = 0 #count of frames since regression of curve without errors
        self.lastX = [] #list of x-coordinate points of center curve of multiple frames
        self.lastY = [] #list of y-coordinate points of center curve of multiple frames
	self.avgDistance = None #average pixel width between lane lines
	self.curves = []  #list of radius of curvature of multiple frames for averaging use
	self.offsets = [] #list of off-center pixel value for multiple frames for averaging


    def _GetAverageXPositionPerRow( self, splitMask ):
        # if we already have a curve context 
	# remove mask noise from central area inside the lane lines according to known curve
        if self.lastModel is not None and self.lastCount < 40:
            d = int(self.avgDistance) >> 2
            curve_model = self.lastModel
            for y in range(splitMask.shape[0]):
                split = curve_model.predict(np.array([y]).reshape(-1,1))
                splitMask[y,split-d:split+1, 0] = 0
                splitMask[y,split-1:split+d, 2] = 0

	#ignore center line vertical striping from color standardization
	halfw = int(splitMask.shape[1]//2)
	splitMask[:,halfw-3:halfw+3,:] = 0

        #change to 0-1 value range
        binMask = np.clip(splitMask.copy(), 0, 1).astype(np.float32)

        #build 3d array filled with x-position location
        xValues = list(range( splitMask.shape[1] ))
        x3d = np.dstack((xValues,xValues,xValues)).astype(np.float32)

        #multiply mask by x-positions to replace ON values with x-positions
        xfactor = binMask * x3d

        #get average x-position for every y row
        with np.errstate(invalid='ignore', divide='ignore'):
            averageXposition = np.true_divide( \
                xfactor.sum(axis=1), (xfactor>0).sum(axis=1) )
        averageXposition = np.clip( averageXposition, 0, splitMask.shape[1] - 1 )
        averageXposition = averageXposition.astype(np.int)

	#each  y value now holds average x coordinate of the lane lines in that dimension
        return averageXposition


    def _GetAverageDistanceFromLaneToCenter( self, averageXposition, width ):
        avg_dist = 0
        avg_count = 0
 	#avg_dist is half the average distance between the lanes in pixels
	#given the average x positions in the two lane dimensions (0,1) calculate average pixel distance
        for x in averageXposition:
            if (x[2] > 1 and x[0] > 1):
                avg_count += 1
                avg_dist += x[2]-x[0]
        if avg_count > 0 and avg_dist > 0:
            avg_dist = int(abs(avg_dist / avg_count) // 2)
        else:
            avg_dist = int(width // 6)

        #smooth out the new calculation based on previous frame distance calculations
	newavg = int(avg_dist)<<1
        if (newavg > 50):
            self.avgDistance = (self.avgDistance*0.95 + newavg*0.05)
	    avg_dist = int(self.avgDistance // 2)

	#calculate center lane position based on the left/right lanes and distance
        centers = np.zeros(averageXposition.shape[:1], dtype=np.int)
        for y in range(len(averageXposition)):
            x = averageXposition[y]
            if ( x[0] > 1 and x[2] > 1 ):
                centers[y] = int( (x[0] + x[2]) >> 1 )
            elif ( x[0] > 1 ):
                centers[y] = int( x[0] + avg_dist )
            elif ( x[2] > 1 ):
                centers[y] = int( x[2] - avg_dist )

        return centers

    def _ShiftSplitMaskIntoCenter( self, splitMask, centers ):
	#based on the the distance and centers shift the left right masks into the center 
        distance = int(self.avgDistance) >> 1
        right = np.roll(splitMask[:,:,2].copy(), -distance, axis=1).astype(np.bool)
        left = np.roll(splitMask[:,:,0].copy(), distance, axis=1).astype(np.bool)
        middle = np.zeros((splitMask.shape[0], splitMask.shape[1])).astype(np.bool)
	#add current context bias to the mask by plotting points on our current curve into the mask
        for y,x in enumerate(centers):
            if (x > 1):
                x = max(5,min(splitMask.shape[1]-6,x))
                middle[y,x-5:x+5] = True
       	#combine left, right, and current curve into one binary mask 
	full = (left | middle | right )
        return full

    def _FitCurve( self, splitMask, centerShifted ):
	#keep some mask information from prior frames
        tenFrame = len(centerShifted) * 100
        if len(self.lastX) > tenFrame:
            self.lastX = self.lastX[-tenFrame:]
            self.lastY = self.lastY[-tenFrame:]
       
	#append current mask information to lastX and lastY lists 
        for y,row in enumerate(centerShifted):
            for x,v in enumerate(row):
                if v:
                    self.lastX.append(x)
                    self.lastY.append(y)

	#regress a polynomial fit to lastX,lastY information
        try:
            model_ransac = Pipeline(steps=[('kernel',PolynomialFeatures(2)), \
                     ('ransac',linear_model.RANSACRegressor(linear_model.LinearRegression()))])
            model_ransac.fit(np.array(self.lastY).reshape(-1,1), np.array(self.lastX).reshape(-1,1))
            self.lastModel = model_ransac
            self.lastCount = 0
        except Exception as e:
	    #if the polynomail fit fails, use the last model and increment the count of misses (lastCount)
            print(str(e))
            if self.lastModel is not None and self.lastCount < 40:
                model_ransac = self.lastModel
                self.lastCount += 1
            else: 
		#if there's no lastModel or we missed too many times, return lines mask as overlay, don't plot a lane
                return (splitMask.astype(np.int) * 255).astype(np.uint8)

	#using the regression, paint the lane onto the image
        imagenew = splitMask.copy().astype(np.int)
        distance = int(self.avgDistance / 2 * 0.999)
        for y in range(splitMask.shape[0]):
            x = model_ransac.predict(np.array([y]).reshape(-1,1))
            x = max(min(splitMask.shape[1]-1, x) , 0)
            imagenew[y,0:int(x)-int(distance*1.5),2] = 0
            imagenew[y,int(x)+int(distance*1.5):,0] = 0
            imagenew[y,int(x)-distance:int(x)+distance,0] = 1
            imagenew[y,int(x)-distance:int(x)+distance,2] = 1

        imagenew = (imagenew * 255).astype(np.uint8)
        return imagenew


    def _SplitLanes( self, mask ):
        #remove 3rd array dimension ( its size one anyway )
        mask = mask.reshape(mask.shape[:2])
        left_mask = mask.copy()
        right_mask = mask.copy()

        #if we already have a curve, remove outliers from last curve 
        if self.lastModel is not None and self.lastCount < 40:
            curve_model = self.lastModel
            for y in range(mask.shape[0]):
                split = curve_model.predict(np.array([y]).reshape(-1,1))
                left_mask[y, split:] = 0
                right_mask[y, :split] = 0
	else:
		#divide image in half horizontally
		w_ = mask.shape[1] // 2

		#empty right half of left lane mask
		left_mask[:,w_:] =  0 

		#empty left half of right lane mask
		right_mask[:,:w_] =  0 

        #Return new R,G,B mask - R = Left Lane, G = Empty, B = Right Lane
        empty_mask = np.zeros(mask.shape)
        result = np.dstack( ( left_mask*255, empty_mask, right_mask*255 ) ) 
   	return result
 
    def RegressCurve( self, splitMask ):
	#if we don't have a lane distance yet, use 1/3 of the width of the cropped image area
        if self.avgDistance is None:
            self.avgDistance = splitMask.shape[1] // 3

	#split the left and right lanes into the red and blue dimensions of an image
	splitMask = self._SplitLanes( splitMask )

       	#calculate the average X pixel coordinate for each Y for each lane 
	averageXposition = self._GetAverageXPositionPerRow( splitMask )

	#calculate the center lane pixel X coordinate for each Y value
        centers = \
            self._GetAverageDistanceFromLaneToCenter(
                    averageXposition, splitMask.shape[1] )
        
       	#shift the split masks into the center to form one centered lane line
	centerShifted = self._ShiftSplitMaskIntoCenter( splitMask, centers)
                   
       	#fit a curve to the center lane line 
	curveFitted = self._FitCurve( splitMask, centerShifted )

	#Calculate the Center Offset and Radius of Curvature
        y3 = splitMask.shape[0]
        x3 = self.lastModel.predict(np.array(y3).reshape(-1,1))[0][0]

	offCenter = splitMask.shape[1]//2 - x3  #distance from center scrn
        offCenter = offCenter / self.avgDistance #screen pixels/lane width 
	offCenter = offCenter * 370 #distance in CM of USA lane width
	self.offsets.append(offCenter)
	if len(self.offsets) > 30:
		self.offsets = self.offsets[1:]


	#get the coefficients of the curve	
	A = self.lastModel.named_steps['ransac'].estimator_.coef_[0][1]
	B = self.lastModel.named_steps['ransac'].estimator_.coef_[0][2]
	#calculate radius of curvature from polynomial coefficients
	radius = ((1+(2*y3+B)**2)**1.5)/abs(2*A)
	radius = radius / self.avgDistance
	radius = radius * 3.7
	radius = radius / 1000000

	#appned to list of curves used for averaging	
	self.curves.append(radius)
	if len(self.curves) > 30:
		self.curves = self.curves[1:]

	#do the averaging and return the results
	curvature = np.mean(self.curves)
	offset = np.mean(self.offsets)	
	return ( curvature, offset, curveFitted )
