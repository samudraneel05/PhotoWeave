from PIL import Image
import cv2
import numpy as np
import imageio

cv2.ocl.setUseOpenCL(False)
import warnings
warnings.filterwarnings('ignore')

# Define target dimensions for 16:9 aspect ratio
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Function to standardize images to 16:9 aspect ratio
def standardize_image(image):
    """Resize the image to the target dimensions (16:9 aspect ratio)."""
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = w / h
    
    # Resize while maintaining aspect ratio
    if aspect_ratio > (TARGET_WIDTH / TARGET_HEIGHT):
        # Wider than 16:9
        new_width = TARGET_WIDTH
        new_height = int(TARGET_WIDTH / aspect_ratio)
    else:
        # Taller than 16:9
        new_height = TARGET_HEIGHT
        new_width = int(TARGET_HEIGHT * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the target dimensions and fill it with black
    standardized_image = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    
    # Calculate the position to place the resized image
    y_offset = (TARGET_HEIGHT - new_height) // 2
    x_offset = (TARGET_WIDTH - new_width) // 2
    
    # Place the resized image in the center of the standardized image
    standardized_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return standardized_image

# Image stitching function
def stitch_images(train_image, query_image):
    # Normalize images
    # train_image = standardize_image(train_image)
    # query_image = standardize_image(query_image)

    # Convert images to RGB (OpenCV loads images in BGR by default)
    train_photo = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)

    query_photo = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

    # Feature extraction
    def select_descriptor_methods(image, method='sift'):
        if method == 'sift':
            descriptor = cv2.SIFT_create()
        elif method == 'surf':
            descriptor = cv2.SURF_create()
        elif method == 'brisk':
            descriptor = cv2.BRISK_create()
        elif method == 'orb':
            descriptor = cv2.ORB_create()
        (keypoints, features) = descriptor.detectAndCompute(image, None)
        return (keypoints, features)

    keypoints_train_img, features_train_img = select_descriptor_methods(train_photo_gray, method='sift')
    keypoints_query_img, features_query_img = select_descriptor_methods(query_photo_gray, method='sift')

    # Feature matching
    def create_matching_object(method, crossCheck):
        if method == 'sift' or method == 'surf':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif method == 'orb' or method == 'brisk':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        return bf

    def key_points_matching(features_train_img, features_query_img, method):
        bf = create_matching_object(method, crossCheck=True)
        best_matches = bf.match(features_train_img, features_query_img)
        rawMatches = sorted(best_matches, key=lambda x: x.distance)
        return rawMatches

    matches = key_points_matching(features_train_img, features_query_img, method='sift')

    
    # Homography stitching
    def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):
        keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
        keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])
        if len(matches) > 4:
            points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
            points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
            (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        else:
            return None

    M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)
    if M is None:
        return None

    (matches, Homography_Matrix, status) = M

    # Warping and stitching
    width = query_photo.shape[1] + train_photo.shape[1]
    height = max(query_photo.shape[0], train_photo.shape[0])
    result = cv2.warpPerspective(train_photo, Homography_Matrix, (width, height))
    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo

    return result