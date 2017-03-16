# Masking.py

#EXTERNAL DEPENDENCIES
import numpy as np
import cv2

class Masking:

    def LaneEdges( self, image ):
        #Mask image based on red channel
        r_binary = self._RedMask( image )

        #Mask image based on value channel
        v_binary = self._ValueMask( image )

        #Mask image based on Sobel Edges over the blue channel
        sb_binary = self._SobelBlueMask( image )

        #Aggregate masks to single masking
        mask = self._FullMask( r_binary, sb_binary, v_binary )

        #Crop out data outside possible lane curvatures
        mask = self._CropMask( mask )

        return mask


    def _RedMask( self, image ):
        #pull red channel from RGB image
        r_channel = image[:,:,0]

        #mask channel based on threshold 245-255
        r_binary = np.zeros_like(r_channel)
        r_binary[ (r_channel >= 235) ] = 1

        return r_binary


    def _ValueMask( self, image ):
        #pull value channel from RGB image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:,:,2]

        #mask channel based on threshold 245-255
        v_binary = np.zeros_like(v_channel)
        v_binary[ (v_channel >= 245) ] = 1

        return v_binary


    def _SobelBlueMask( self, image ):
        #pull blue channel from RGB image
        b_channel = image[:,:,2]

        #resize image to 600x600 and make Vertical Sobel Edges
        x = cv2.resize( b_channel, (600,600) )
        x = np.absolute( cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=31) )
        x = np.uint8( 255 * x / (np.max(x) + 0.00000001) )

        #blur sobel edges to give more information to curve regression
        x = cv2.resize(x, (300,300))
        x = cv2.resize(x, (image.shape[1], image.shape[0]))

        s_binary = np.zeros_like(b_channel)
        s_binary[ (x >= 25)  ] = 1

        return s_binary


    def _FullMask( self, maskA, maskB, maskC ):
        mask = np.zeros_like(maskA)
        mask[  (maskA == 1) & (maskB == 1) & (maskC == 1) ] = 255
        return mask


    def _CropMask( self, image ):
	#crop out lower left and lower right corners, assuming car is not riding ontop of a lane
        keep_region = np.array( [
                [ image.shape[1]*.2,    image.shape[0]*.999 ],
                [ image.shape[1]*.22,    image.shape[0]*.55 ],
                [ image.shape[1]*.15,    image.shape[0]*.001 ],
                [ image.shape[1]*.85,    image.shape[0]*.001 ],
                [ image.shape[1]*.78,    image.shape[0]*.55 ],
                [ image.shape[1]*.8,    image.shape[0]*.999 ] ], dtype=np.int)
        keep_area = np.zeros_like( image )
        cv2.fillPoly( keep_area, [keep_region], 255 )
        result = cv2.bitwise_and( image, keep_area )

	#Crop out very bottom center, assuming car is not riding ontop of a lane line	
	halfw = int(result.shape[1] // 2)
	halfh = int(result.shape[0] // 2)
	halfd = int(result.shape[1] // 16)
	result[ halfh:, halfw-halfd:halfw+halfd ] = 0
	return result


    def _SplitLanes( self, mask ):
        #remove 3rd array dimension ( its size one anyway )
        mask = mask.reshape(mask.shape[:2])

        #divide image in half horizontally
        w_ = mask.shape[1] // 2

        #empty right half of left lane mask
        left_mask = mask.copy()
        left_mask[:,w_:] = left_mask[:,w_:] * 0

        #empty left half of right lane mask
        right_mask = mask.copy()
        right_mask[:,:w_] = right_mask[:,:w_] * 0

        #Return new R,G,B mask - R = Left Lane, G = Empty, B = Right Lane
        empty_mask = np.zeros(mask.shape)
        return np.dstack( ( left_mask*255, empty_mask, right_mask*255 ) )
