# LaneAPI.py

# INTERNAL DEPENDENCIES
from Calibrate import Calibrate
from Pipeline  import Pipeline

class LaneAPI:

    def __init__( self, calibration = None ):
        self.calibration = calibration


    def ProcessVideo( self, videoSourcePath, videoDestinationPath ):
        Pipeline(
            videoSourcePath,
            videoDestinationPath,
            self.calibration
        ).Run()


    # it is OK to NOT do this calibrate before processing
    # if no images are given, it's ok, no lens distortion calibration is done.
    # imagesPath: path of calibration images, e.g. 'subdir/calibration*.jpg'
    # hCorners, vCorners: number of horizontal, vertical interior corners
    # resizeTuple: resize the calibration images to a give size on input
    # if no crop pixels are given, it's ok, there will just be no image cropping prior to pipeline
    # cropPixels: amounts to crop the non-calibration pipline images ( cropLeft, cropRight, cropTop, cropBottom )
    def Calibrate( self,
                   imagesPath = None,
                   hCorners = 1,
                   vCorners = 1,
                   resizeTuple = None,
                   cropPixels = None ):
        self.calibration = Calibrate(
            imagesPath,
            hCorners,
            vCorners,
            resizeTuple,
            cropPixels
        )
