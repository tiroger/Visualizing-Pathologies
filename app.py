import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

def read_image(name):
    uploaded_file = st.file_uploader("Upload an " + name, type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        return Image.open(uploaded_file)

def calculate_sphericity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    sphericity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
    return sphericity

def create_contour_mask_with_manual_threshold(image, threshold_value, sphericity_threshold):
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by sphericity
    filtered_contours = [cnt for cnt in contours if calculate_sphericity(cnt) >= sphericity_threshold]
    return filtered_contours

def _bbAndMask(image, cnts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.axis('off')
    ax2.axis('off')
    _bbox(image, cnts, ax1)
    _maskOutline(image, cnts, ax2)
    st.pyplot(fig)

def _bbox(image, cnts, ax):
    ax.imshow(image)
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            [x, y, w, h] = cv2.boundingRect(c)
            ax.add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))

def _maskOutline(image, cnts, ax):
    img = _drawMask(image, cnts, False)
    ax.imshow(img)

def _drawMask(image, cnts, fill):
    markers = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(markers, cnts, -1, (255), thickness=-1 if fill else 2)  # Fill or outline
    heatmap_img = cv2.applyColorMap(markers, cv2.COLORMAP_JET)
    return heatmap_img  # Return modified heatmap image

def _heatmap(image, cnts):
    fig2, ax = plt.subplots()
    ax.axis('off')
    hm = st.slider("Adjust Heatmap Transparency", 0.0, 1.0, 0.5, 0.1)
    img = _drawMask(np.array(image), cnts, True)
    ax.imshow(image, alpha=1.0)  # Show original image
    ax.imshow(img, alpha=hm)    # Overlay heatmap
    st.pyplot(fig2)

def preprocess_image(image, target_size=(150, 150)):
    """Resize and preprocess the image."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize to [0,1] if that's what your model expects
    return image

# Load the model
model = tf.keras.models.load_model('model.keras')

def predict_image(model, image):
    # Preprocess the image to the expected format and size
    processed_image = preprocess_image(image, target_size=(150, 150))  # Adjust according to your model's input size
    # Predict using your loaded model
    predictions = model.predict(processed_image)
    # Return the predictions
    return predictions[0]


def main():
    st.set_page_config(page_title='Pathology Detection', layout='centered')
    st.title('Detecting Pathologies Through Computer Vision in Ultrasound')
    with st.sidebar.title('About'):
        st.markdown("""
        <h2>Detecting Pathologies Through Computer Vision in Ultrasound</h2>
        <h3>About</h3>
        <h4>Overview</h4>
        This Streamlit-based web application is designed to assist in the medical analysis by automatically detecting and classifying tumors in ultrasound images as benign or malignant. Utilizing advanced computer vision and machine learning techniques, the app provides healthcare professionals with a rapid, initial assessment tool that aids in the diagnosis and understanding of potential pathologies.

        <h4>Features</h4>
        <b>Image Upload</b>: Users can upload ultrasound images of tumors for analysis.
        Contour Analysis: The app processes uploaded images to detect and highlight contours of tumors, facilitating visual assessment of size and shape.
        Sphericity Measurement: By evaluating the roundness of detected contours, the app assists in identifying characteristics typically associated with benign or malignant tumors.
        Threshold Adjustment: Users can adjust the threshold settings for contour detection and sphericity to fine-tune the analysis based on specific case needs.
        Machine Learning Prediction: The core of the app lies in its ability to predict whether a tumor is likely benign or malignant, using a trained machine learning model. The prediction is accompanied by a probability score, providing insight into the model's confidence.
        <br>
        <b>Interactive Visuals</b>: The application includes interactive heatmaps and contour overlays, allowing for an in-depth examination of the tumor's properties.e
        <br>
        The web application is straightforward to use: simply upload an ultrasound image, and the system will automatically process the image, display relevant visual insights, and provide a prediction for the nature of the tumor.
        <br>
        <h4>Technical Details</h4>
        The app employs Convolutional Neural Networks (CNNs) for image analysis and classification, leveraging preprocessed image data to ensure accurate and reliable predictions. Parameters such as contour detection thresholds and sphericity can be adjusted by users to cater to different image qualities and diagnostic requirements.
        <br>
        """, unsafe_allow_html=True)
        
    # Display sample images
    st.write("#### Sample Images")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("normal.png", caption="Normal Tissue")
    with col2:
        st.image("benign.png", caption="Benign Tumor")
    with col3:
        st.image("malignant.png", caption="Malignant Tumor")

    st.write("Download and then drag and drop a sample image above for prediction.") 
        
    image = read_image('image')
    
    if image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        
        with col2:
            # Make sure to load your model before calling predict_image
            predictions = predict_image(model, image)  # Ensure 'model' is defined and loaded properly
            class_labels = ['Benign', 'Malignant', 'Normal Tissue']
            predicted_class = class_labels[np.argmax(predictions)]
            predicted_prob = np.max(predictions)

            # Display predicted class and probabilities
            color = 'red' if predicted_class == 'Malignant' else 'blue' if predicted_class == 'Benign' else 'green'
            st.markdown(f'<h1 style="color: {color};">{predicted_class}</h1>', unsafe_allow_html=True)
            st.write(f"The tumor is likely **{predicted_class}** with a probability of **{predicted_prob*100:.2f}%**.")
                        
            # Additionally, display all class probabilities
            st.write("Class probabilities:")
            for i, (label, prob) in enumerate(zip(class_labels, predictions)):
                st.write(f"{label}: {prob*100:.2f}%")

    # Draw the contour and heatmap only if a tumor is detected
    try:
        if predicted_class in ['Benign', 'Malignant']:
            st.write("Contour Analysis:")
            st.write("The contour analysis provides a visual representation of the tumor's size and shape.")
            st.write("Heatmap Analysis:")
            st.write("The heatmap analysis highlights the tumor's location and shape, providing additional visual insights.")
            
            threshold_value = st.slider("Select Threshold", 0, 255, 100, key='threshold')
            sphericity_threshold = st.slider("Select Sphericity Threshold", 0.0, 1.0, 0.0, 0.01, key='sphericity')
            contours = create_contour_mask_with_manual_threshold(image, threshold_value, sphericity_threshold)
            if contours:
                _bbAndMask(np.array(image), contours)
                _heatmap(image, contours)
    except NameError:
        pass
    
    
    st.markdown("---")
    # Disclaimer
    st.markdown("""
                <b>Disclaimer:</b> This app is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. If you have any concerns or questions about your health, you should always consult with a physician or other healthcare professional.
                """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()