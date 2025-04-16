from PIL import Image
import streamlit as st
import cv2
import numpy as np
import imageio
import io

cv2.ocl.setUseOpenCL(False)
import warnings
warnings.filterwarnings('ignore')

# Define target dimensions for 16:9 aspect ratio
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

def standardize_image(image: np.ndarray, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> np.ndarray:
   
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Compute the new dimensions while preserving the aspect ratio
    scale_factor = min(target_width / w, target_height / h)
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)
    
    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a blank canvas with the target dimensions
    standardized_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Compute centering offsets
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Paste the resized image onto the center of the canvas
    standardized_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return standardized_image


# Image stitching function
def stitch_images(images, crop=True):
    """
    Stitch multiple images together using OpenCV
    
    Args:
        images: List of images to stitch
        crop: Whether to perform advanced cropping to remove black borders
        
    Returns:
        stitched_image: The final stitched panorama
    """
    # Initialize OpenCV's stitcher
    stitcher = cv2.Stitcher_create()
    
    # Convert images to BGR format if they're in RGB
    bgr_images = []
    for img in images:
        if img.shape[2] == 3:  # If it's RGB
            bgr_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            bgr_images.append(img)
    
    # Perform stitching
    status, stitched = stitcher.stitch(bgr_images)
    
    if status == cv2.Stitcher_OK:
        # Convert back to RGB for display
        stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
        
        if crop:
            # Create a 10 pixel border surrounding the stitched image
            stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                         cv2.BORDER_CONSTANT, (0, 0, 0))
            
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(stitched, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            
            # Find contours and get the largest one
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask for the largest contour
            mask = np.zeros(thresh.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            # Create two copies of the mask
            minRect = mask.copy()
            sub = mask.copy()
            
            # Erode until we find the minimum rectangular region
            while cv2.countNonZero(sub) > 0:
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)
            
            # Find contours in the minimum rectangular mask
            contours, _ = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            
            # Extract the final stitched image
            stitched = stitched[y:y + h, x:x + w]
        
        return stitched
    else:
        return None

# Streamlit app
st.title("📸 Image Stitching App")
st.write("Image stitching app for two or more images that works even if the pictures have different scaling, angle (Perspective), spacial position or capturing devices.")
st.write("Upload multiple images to stitch them together. Make sure the images overlap slightly for best results.")

# File uploaders
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Read images
    images = [cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR) for uploaded_file in uploaded_files]

    # Check if images are loaded correctly
    if any(image is None for image in images):
        st.error("Error loading one or more images. Please check the files.")
    else:
        # Display uploaded images
        st.subheader("Uploaded Images")
        cols = st.columns(len(images))
        for col, img in zip(cols, images):
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            col.image(img_rgb, use_container_width=True)

        # Add crop option in UI
        crop_option = st.checkbox("Crop black borders", value=True)
        
        # Stitch all images at once
        st.subheader("Stitched Image")
        with st.spinner("Stitching images..."):
            stitched_image = stitch_images(images, crop=crop_option)
            
            if stitched_image is None:
                st.error("Error stitching images. Please ensure the images overlap sufficiently and have enough unique features.")
            else:
                # Convert to RGB for Streamlit display
                stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
                
                # Display the result
                st.image(stitched_image_rgb, caption="Stitched Image", use_container_width=True)
                
                # Save and provide download with proper color handling
                result_path = "stitched_image.jpg"
                
                # Save in BGR format that OpenCV uses
                cv2.imwrite(result_path, cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
                
                # For download, we need to convert back to RGB
                with Image.open(result_path) as img:
                    img_rgb = img.convert('RGB')
                    img_byte_arr = io.BytesIO()
                    img_rgb.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    st.download_button(
                        label="Download Stitched Image",
                        data=img_byte_arr,
                        file_name="stitched_image.jpg",
                        mime="image/jpeg"
                    )