# Main.py

#THIS SCRIPT RUNS THE LANE FINDER ON THE CHALLENGE VIDEOS
from LaneAPI import LaneAPI
laneFinder = LaneAPI()



#cropPixels is tuple: ( leftOffset, rightOffset, topOffset, bottomOffset )
print( 'project_out starting' )
laneFinder.Calibrate( cropPixels = (0,0,400,30) )
laneFinder.ProcessVideo( './test_videos/harder_challenge_video.mp4',
                         './out_videos/LANES_harder_challenge_video.mp4' )
print( 'project_out finished' )



print( 'project_out starting' )
laneFinder.Calibrate( cropPixels = (0,0,405,30) ) 
laneFinder.ProcessVideo( './test_videos/challenge_video.mp4',
                         './out_videos/LANES_challenge_video.mp4' )
print( 'project_out finished' )



print( 'project_out starting' )
laneFinder.Calibrate( cropPixels = (0,0,381,30) )
laneFinder.ProcessVideo( './test_videos/project_video.mp4',
                         './out_videos/LANES_project_video.mp4' )
print( 'project_out finished' )



#print( 'AppletonLanes_out starting' )
#laneFinder.Calibrate( cropPixels = (0,0,220,100) )
#laneFinder.ProcessVideo( './test_videos/AppletonLanes.mp4',
#                         './out_videos/LANES_AppletonLanes.mp4' )
#print( 'AppletonLanes_out finished' )


#HAPPY HACKING!
