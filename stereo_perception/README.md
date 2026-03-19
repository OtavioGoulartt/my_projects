# Stereo perception:

* Disparity Map Generation (disparity_map_pub.py):
Processes synchronized stereo pairs from OAK-D cameras to compute dense depth maps. It handles raw signal rectification and disparity calculations, providing the foundation for Z-coordinate (depth) extraction.

* 3D Object Tracking (cone_track_node.py):
Integrates real-time YOLOv8 inferences with depth data to localize objects in 3D space. This node constructs structured Track messages, mapping detected classes to their precise(x, y, z) coordinates.

* Perception Mathematics (perception_calc.py):
A centralized Python base class that encapsulates projective geometry, coordinate transformations, and stereo triangulation logic. It ensures mathematical consistency and optimizes data flow across the entire node network.

* Validation (track_validation.py):
Provides real-time visual comparison between estimated Z-coordinates and Ground Truth data using Matplotlib. This node is essential for monitoring sensory accuracy and quantifying depth estimation errors during live operation.

![image alt](https://github.com/OtavioGoulartt/my_projects/blob/dc4198c646feeb90ee3b3ebec8c33809ef25b3de/stereo_perception/cone_gt_comparison.png)
