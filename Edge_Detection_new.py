import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Step 1: Read the test image using OpenCV
image = cv2.imread('E:/Edge_detection/Image/Project #1 Image_Peppers.jpg', cv2.IMREAD_GRAYSCALE) 


# Step 2: Add Gaussian noise
def add_gaussian_noise(image, mean=0, std_dev=25):
    """
    Add Gaussian noise to the input image.
    """
    gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Clip values to valid range
    return noisy_image

# Add Gaussian noise to the image
gaussian_noisy_image = add_gaussian_noise(image, mean=0, std_dev=25)
gaussian_blurred_image = cv2.GaussianBlur(gaussian_noisy_image, (5, 5), 0)
# Step 3: Wavelet decomposition and scale multiplication
def wavelet_edge_detection(image, wavelet='haar', levels=3):
    coeffs_list = pywt.wavedec2(image, wavelet, level=levels)
    

    # Loop through scales (levels)
    for level in range(1, levels): 
        # Extract horizontal and vertical detail coefficients at the current and next levels
        cH_curr, cV_curr, _ = coeffs_list[level] 
        cH_next, cV_next, _ = coeffs_list[level + 1]  

        # Resize next-level coefficients to match the current level size
        cH_next_resized = cv2.resize(cH_next, cH_curr.shape[::-1], interpolation=cv2.INTER_LINEAR)
        cV_next_resized = cv2.resize(cV_next, cV_curr.shape[::-1], interpolation=cv2.INTER_LINEAR)

        # Multiply horizontal and vertical components between scales
        h_product = np.abs(cH_curr * cH_next_resized)
        v_product = np.abs(cV_curr * cV_next_resized)

       

    # Combine horizontal and vertical edges
    edge_map = np.sqrt(h_product**2 + v_product**2)
    return edge_map

# Compute the edge map with scale multiplication
edge_map = wavelet_edge_detection(gaussian_blurred_image, wavelet='haar', levels=3)

# Step 5: Normalize the edge map
edge_map_normalized = cv2.normalize(edge_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# Step 6: Apply adaptive thresholding 
_, edge_map_thresholded = cv2.threshold(edge_map_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



# Display the results
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Gaussian Noisy Image")
plt.imshow(gaussian_noisy_image, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Edge Map (Scale Multiplication)")
plt.imshow(edge_map, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Thresholded Edge Map")
plt.imshow(edge_map_thresholded, cmap='gray')

plt.tight_layout()
plt.show()
