# Stereo perception:

* Disparity Map Generation (disparity_map_pub.py):
Processes synchronized stereo pairs from OAK-D cameras to compute dense depth maps. It handles raw signal rectification and disparity calculations, providing the foundation for Z-coordinate (depth) extraction.

* 3D Object Tracking (cone_track_node.py):
Integrates real-time YOLOv8 inferences with depth data to localize objects in 3D space. This node constructs structured Track messages, mapping detected classes to their precise(x, y, z)$ coordinates.

* Perception Mathematics (perception_calc.py):
A centralized C++ base class that encapsulates projective geometry, coordinate transformations, and stereo triangulation logic. It ensures mathematical consistency and optimizes data flow across the entire node network.Ground Truth 

* Validation (track_validation.py):
Quantifies system reliability by comparing estimated spatial data against Ground Truth from simulation (e.g., FSDS) or annotated datasets. This component is critical for measuring sensory error and fine-tuning state estimation algorithms
