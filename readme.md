### Detecting Pathologies Through Computer Vision in Ultrasound

#### About

**Overview**
This Streamlit-based web application is designed to assist in the medical analysis by automatically detecting and classifying tumors in ultrasound images as benign or malignant. Utilizing advanced computer vision and machine learning techniques, the app provides healthcare professionals with a rapid, initial assessment tool that aids in the diagnosis and understanding of potential pathologies.

**Features**
- _Image Upload_: Users can upload ultrasound images of tumors for analysis.
- Contour Analysis: The app processes uploaded images to detect and highlight contours of tumors, facilitating visual assessment of size and shape.
- _Sphericity Measurement_: By evaluating the roundness of detected contours, the app assists in identifying characteristics typically associated with benign or malignant tumors.
- _Threshold Adjustment_: Users can adjust the threshold settings for contour detection and sphericity to fine-tune the analysis based on specific case needs.
- _Machine Learning Prediction_: The core of the app lies in its ability to predict whether a tumor is likely benign or malignant, using a trained machine learning model. The prediction is accompanied by a probability score, providing insight into the model's confidence.
- _Interactive Visuals_: The application includes interactive heatmaps and contour overlays, allowing for an in-depth examination of the tumor's properties.

The web application is straightforward to use: simply upload an ultrasound image, and the system will automatically process the image, display relevant visual insights, and provide a prediction for the nature of the tumor.

**Technical Details**

The app employs Convolutional Neural Networks (CNNs) for image analysis and classification, leveraging preprocessed image data to ensure accurate and reliable predictions. Parameters such as contour detection thresholds and sphericity can be adjusted by users to cater to different image qualities and diagnostic requirements.
