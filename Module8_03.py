import cv2
import numpy as np
import streamlit as st
from PIL import Image

def convert_to_grayscale(photo):
    grayscale = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    return grayscale

# Sepia / Vintage Filter.
def sepia(img):
    img_sepia = img.copy()
    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB) 
    img_sepia = np.array(img_sepia, dtype = np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to the range [0, 255].
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype = np.uint8)
    #img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
    return img_sepia

# Vignette Effect.
def vignette(img, level = 2):

    if level == 0:
        level = 2
        
    height, width = img.shape[:2]  
    
    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
        
    # Generating resultant_kernel matrix.
    kernel = Y_resultant_kernel * X_resultant_kernel.T 
    mask = kernel / kernel.max()
    
    img_vignette = np.copy(img)
        
    # Applying the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * mask
    
    return img_vignette

def PencilSketch(image, ksize):
    # Pencil Sketch Filter.
    img_blur = cv2.GaussianBlur(image, (ksize,ksize), 0, 0)
    img_sketch_bw, _ = cv2.pencilSketch(img_blur)

    return img_sketch_bw


# Create title
st.title("Photo Filter application")

# Create file uploader
uploaded_file = st.file_uploader("Upload a photo", type=['png', 'jpeg', 'jpg'])

# Create two placeholder columns
placeholder = st.columns(2)

# Check if file has been uploaded
if uploaded_file is not None:
    # Convert to numpy array to be read by opencv
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    photo = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
   
    # Display input image
    placeholder[0].text("Input Image")
    placeholder[0].image(photo[:, :, ::-1])

    # Convert to grayscale
    grayscale_photo = convert_to_grayscale(photo)

    # Create a dropdown box to select the filter
    option = st.selectbox('Select an option', ['Grayscale', 'Vignette', 'Sepia', 'Pencil sketch'])

    if option == 'Grayscale':
        # Convert to grayscale
        grayscale_photo = convert_to_grayscale(photo)

        # Display output image
        placeholder[1].text("Output Image")
        placeholder[1].image(grayscale_photo)
    elif option == 'Sepia':
        # Apply Sepia filter
        sepia = sepia(photo)

        # Display output image
        placeholder[1].text("Output Image")
        placeholder[1].image(sepia)
    elif option == 'Vignette':
        # Create a slider
        value = st.slider("Level", 0, 11, 2)
        
        # Apply vignette filter
        Vignette = vignette(photo, value)

        # Display output image
        placeholder[1].text("Output Image")
        placeholder[1].image(Vignette)
    elif option == 'Pencil sketch':
        # Create a slider
        ksize = st.slider("Blur kernel size", 1, 11, 5, step=2)
        
        # Apply pencil sketch filter
        sketch = PencilSketch(photo, ksize)

        # Display output image
        placeholder[1].text("Output Image")
        placeholder[1].image(sketch)
