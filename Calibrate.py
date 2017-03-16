# Calibrate.py

#EXTERNAL DEPENDENCIES
import glob, cv2
import numpy as np

class Calibrate:

    #imagesPath is location of calibration images
    #vCorners is number of vertical chessboard corners
    #hCorners is number of horizontal chessboard corners
    #resizeTuple is tuple: (width, height)
    #cropPixels is tuple: ( leftOffset, rightOffset, topOffset, bottomOffset )
    def __init__( self,
                  imagesPath = None,
                  hCorners = 1,
                  vCorners = 1, \
                  resizeTuple = None,
                  cropPixels = None):
            self.cropPixels = cropPixels
            self.distortion = None
            self.matrix = None
            if imagesPath:
                self._Calibrate( imagesPath, resizeTuple, hCorners, vCorners )


    def _Calibrate( self, imagesPath, resizeTuple, hCorners, vCorners ):
        #glob get all image file paths matching imagesPath
        images = glob.glob(imagesPath)

        #open all images with cv2
        images = tuple(map(lambda x: cv2.imread(x), images))
        if not images:
            return

        #if resizing necessary, resize to (w,h) tuple parameter
        if resizeTuple:
            images = tuple(map(lambda x: cv2.resize(x, resizeTuple), images))

        #convert image to grayscale
        images = tuple(map(lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2GRAY), images))

        #cv2 find chessboard corners
        corners = tuple(map(lambda x: cv2.findChessboardCorners(x,(hCorners,vCorners),None), images))

        #filter out unsuccesfull images and return corner points
        corners = tuple(map(lambda x: x[1], filter(lambda x: x[0], corners)))
        if not corners:
            return

        #create objective ground truth grids for each image
        grid = np.zeros((vCorners * hCorners,3), np.float32)
        grid[:,:2] = np.mgrid[0:hCorners, 0:vCorners].T.reshape(-1,2)
        grids = [grid]*len(corners)

        #calibrate the camera
        _, matrix, distortion, _, _ = cv2.calibrateCamera(grids, corners, images[0].shape[:2], None, None)
        self.matrix = matrix
        self.distortion = distortion


    def Undistort( self, image ):
        #Undistort based on prior calibration
        if self.matrix is not None and self.distortion is not None:
            image = cv2.undistort(image,
                        self.matrix, self.distortion, None, self.matrix)

        #Crop pipeline images to area of interest
        if self.cropPixels: #cropPixels is tuple: ( leftOffset, rightOffset, topOffset, bottomOffset )
            cp = self.cropPixels
            image = image[cp[2]:image.shape[0]-cp[3],cp[0]:image.shape[1]-cp[1],:].copy()

        return image


    def Overlay( self, original, image , curve, offCenter):
	#paste result back onto section of image from which it was originally cropped
        if self.cropPixels:
            #cropPixels is tuple: ( leftOffset, rightOffset, topOffset, bottomOffset )
            x = self.cropPixels[0]
            y = self.cropPixels[2]
            x2 = image.shape[1]
            y2 = image.shape[0]
            original[y:y+y2, x:x+x2, :] = image[:,:,:]
            
        #overlay lane curvature and distance to center of lane
        rtext = "R: " + '{0:.2f}'.format(curve) + "km"
        dtext = "Off Center: "  + '{0:.2f}'.format(offCenter) + "cm"

	yv = original.shape[0] - 30
        
	cv2.putText( original, rtext, (20,yv),  \
                     cv2.FONT_HERSHEY_SIMPLEX, \
                     1, (255,255,255), 2 )
        cv2.putText( original, dtext, (original.shape[1]//2,yv),  \
                     cv2.FONT_HERSHEY_SIMPLEX, \
                     1, (255,255,255), 2 )
  
        return original
