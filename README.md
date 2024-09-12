# APS-2023-1
This project aims to detect deforested areas in forests using two different approaches: image processing with filters and machine learning. 

filter_based_detection.py: This script uses OpenCV and edge detection filters to identify deforested areas in the input images. The contours of the deforested regions are drawn over the original image.
ml_based_detection.pyh: This script uses TensorFlow and Keras to train a CNN model on labeled forest images. The model predicts whether a given forest image shows a deforested or normal area.

Future Improvements
Enhance the machine learning model with additional layers and fine-tuning.
Experiment with more advanced filters for image processing.
Integrate both approaches into a web-based application for easier accessibility.

