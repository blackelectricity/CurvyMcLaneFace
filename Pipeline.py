# Pipeline.py

#External DEPENDENCIES
import numpy, cv2
from moviepy.editor import VideoFileClip

#Internal DEPENDENCIES
from Calibrate import Calibrate
from Masking import Masking
from Regress import Regress
from Transforms import Transforms


class Pipeline:

    def __init__( self, videoSource, videoDestination, calibration ):
        self.videoSource = videoSource
        self.videoDestination = videoDestination
        self.calibration = calibration
        self.transform = Transforms()
        self.masking = Masking()
        self.regress = Regress()


    def Run(self):
        #Reset any lane curve smoothing averages
        self.regress.ResetCurveSmoothing()

        #Read the video input
        inputVideo = VideoFileClip( self.videoSource )

        #Process each frame with _PipelineProcess
        outputVideo = inputVideo.fl_image( self._PipelineProcess )

        #Save the output video
        outputVideo.write_videofile( self.videoDestination )



    def _PipelineProcess( self, imageFrame ):
	#If your video processing codec is reading in BGR, then transform
        #Pipeline will work assuming RGB so, transform colorspace
        #imageFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)

        #Copy incoming frame to overlay pipeline result on
        originalFrame = imageFrame.copy()

        #Undistort lens and crop input image as necessary
        if self.calibration:
            imageFrame = self.calibration.Undistort( imageFrame )

        #Find and overlay lane lines on undistorted and cropped frame
        outputFrame,curve,offCenter = self._FindLanes( imageFrame )

        #If pipeline cropped the image, shift overlay to crop location
        if self.calibration:
            outputFrame = self.calibration.Overlay( originalFrame, outputFrame, curve, offCenter )

        #Return Final Frame for output video in BGR colorspace
        #cv2.cvtColor(outputFrame, cv2.COLOR_RGB2BGR)

	return outputFrame


    def _FindLanes( self, imageFrame ):
        #Copy the incoming frame to use on the overlay
        originalImage = imageFrame.copy()

        #Transform input frame to birds-eye view perspective
        birdsEye = self.transform.BirdsEye( imageFrame )

        #Standardize color to enhance lane lines contrast
        colorStd = self.transform.Standardize( birdsEye )

        #Use Sobel Gradient Edges and Color Thresholding to Mask Lanes
        laneEdge = self.masking.LaneEdges( colorStd )

        #Regress the Curvature of the Lanes using
        ( curve, offCenter, laneImage ) = \
            self.regress.RegressCurve( laneEdge )

        #Transform the lane overlay back from birds-eye to normal perspective
        humanEyeLanes = self.transform.HumanEye( laneImage )

        #Overlay the results on the incoming imageFrame
        return (self._Overlay( originalImage, humanEyeLanes ), curve, offCenter)


    def _Overlay( self, original, lanes):
        #overlay lanes with transparency
        return cv2.addWeighted( original, 1, lanes, 0.33, 0 )
