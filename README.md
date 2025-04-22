# object_detection
A simple testing environment for some object recognition models, to test improve and convert them for c++ usage in the jetson nano.


While a single YOLOv8 model might seem appealing for its unified nature and potentially better feature extraction, the distinct nature of lane detection suggests that a specialized model for this task will be more effective. Combining a lightweight YOLOv5n for object and traffic light detection with a separate, likely segmentation-based, lane detection model is a more robust and potentially higher-performing solution for your future use on a Jetson Nano. Remember to focus heavily on optimizing both models for efficient inference on the Jetson Nano.