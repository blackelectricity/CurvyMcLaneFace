# Transforms.py

#EXTERNAL DEPENDENCIES
import numpy as np
import cv2

class Transforms:

    def __init__( self ):
        self.matrix = None
        self.inverse = None


    #Setup coordinates to for birdsEye perspective transformations
    #Calculate matrix and inverse matrix based on image size
    def _Setup( self, image ):
	self.oshape0 = int(image.shape[0])
	self.oshape1 = int(image.shape[1])
        #the begin array has the region we are targeting
        #this region will be stretched to create the birds eye view
        begin = np.array([
                    [image.shape[1]*-.5,image.shape[0]*.999],
                    [image.shape[1]*.35,image.shape[0]*.26],
                    [image.shape[1]*.65,image.shape[0]*.26],
                    [image.shape[1]*1.5,image.shape[0]*.999]], dtype=np.float32)
	# This transform area was used for the Appleton video's shorter lane finder
        #            [image.shape[1]*-.3,image.shape[0]*.999],
        #            [image.shape[1]*.25,image.shape[0]*.4],
        #            [image.shape[1]*.75,image.shape[0]*.4],
        #            [image.shape[1]*1.3,image.shape[0]*.999]], dtype=np.float32)

        #the end array has the image corners
        end = np.array([
                    [0,image.shape[0]],
                    [0,0],
                    [image.shape[1],0],
                    [image.shape[1],image.shape[0]]], dtype=np.float32)

	#save the transformation parameters 
        self.matrix = cv2.getPerspectiveTransform( begin, end )
        self.inverse = cv2.getPerspectiveTransform( end, begin )


    #Warp to birds-eye view perspective
    def BirdsEye( self, image ):
        if self.matrix is None: #This is the first pipeline step
            self._Setup( image ) #Setup based on Size of image
        
	im =  cv2.warpPerspective( image, self.matrix, \
                                    image.shape[::-1][1:], \
                                    flags=cv2.INTER_LINEAR)
	im = cv2.resize(im, (500,200))
        return im


    #Reverse the birds-eye perspective transform back to normal
    def HumanEye( self, image ):
	image = cv2.resize(image, (self.oshape1, self.oshape0))
        im = cv2.warpPerspective( image, self.inverse, \
                                    image.shape[::-1][1:], \
                                    flags=cv2.INTER_LINEAR)
        return im


    #Standardize color to enhance lane line contrast in image
    def Standardize( self, image ):
        #Get incoming image shape
        osizey,osizex,osizez = image.shape

        #breakup into rectangles, each will be standardized seperately
	boxesizey = 20
	boxedy = image.shape[0] // boxesizey
	for i in range(boxedy):
            for xsplit in range(2):
		#breakup into rectangle based on loop indexes
                sxs = (osizex >> 1)
                sx = xsplit * sxs
                sx2 = (xsplit + 1)  * sxs
                img = image[i*boxesizey:(i+1)*boxesizey, sx:sx2, :]
    
                #Find 2 standard deviations from the mean value in region
                x = img.copy().reshape(-1,3).astype(np.float32)
                stdx = 2.0 * np.std(x, axis=0)
                avgx = np.mean(x, axis=0)
    
                #Clip original image to 2 stdeviations from mean
                x = np.clip(img.reshape(-1,3), avgx-stdx, avgx+stdx)

                #Scale values back to 0-255
                x = x - np.min(x, axis=0)
                x = x.astype(np.float64) / (np.max(x, axis=0) + 0.00000001)
                x = np.clip((x*255.0).astype(np.uint8), 0, 255)

                #update standardized image region
		image[i*boxesizey:(i+1)*boxesizey, sx:sx2, :] = x.reshape((boxesizey,sxs,3))
        
        return image
