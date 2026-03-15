# my_projects

## Camera perception and object position recognition in ROS2:

In this project I have used yolov8, for real time cone recognition, and stereo cameras to create a disparity map and extract the depth data from detected objects. After extracting 3D points, a message of track is created. 

###

--> percecption_calc.py : 

--> cone_track_node.py : This node is responsable for assembling the camera images, disparity map, yolo bouding boxes and publishing the track message.


