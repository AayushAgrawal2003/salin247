import cv2
import numpy as np

# Load the image
image_path = '/home/airlab/Desktop/aayush/Data/Images/Reference.png'
original_image = cv2.imread(image_path)

# --- 1. Convert to HSV Color Space ---
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

# --- 2. Define Green Color Range and Create Plant Mask ---
# These values might need tuning for different lighting conditions/cameras
lower_green = np.array([30, 40, 40])
upper_green = np.array([85, 255, 255])
plant_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# --- 3. Create a Mask to Remove the Top Portion ---
height, width, _ = original_image.shape
top_mask = np.zeros((height, width), dtype=np.uint8)
mask_start_row = int(height * 0.4)
top_mask[mask_start_row:, :] = 255

# --- 4. Combine the Masks ---
final_mask = cv2.bitwise_and(plant_mask, top_mask)

# --- 5. Denoise the Mask using Morphological Operations ---
# Create a kernel for the morphological operations. You can change the size
# (e.g., (3, 3) or (7, 7)) to increase or decrease the effect.
kernel = np.ones((5,5), np.uint8)

# Perform an Opening to remove small noise pixels
denoised_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
# Perform a Closing to fill small holes in the plant regions
denoised_mask = cv2.morphologyEx(denoised_mask, cv2.MORPH_CLOSE, kernel)

# --- 6. Apply the Final Denoised Mask to the Original Image ---
# This will show the original colors of the plants on a black background
# NOTE: Using the new 'denoised_mask' instead of 'final_mask'
result = cv2.bitwise_and(original_image, original_image, mask=denoised_mask)

# --- Bonus: For the gray background like in the example ---
# Create a gray background
gray_background = np.full(original_image.shape, (80, 80, 80), dtype=np.uint8)
# Invert the denoised mask to select the background
background_mask = cv2.bitwise_not(denoised_mask)
# Apply the inverted mask to the gray image
background_part = cv2.bitwise_and(gray_background, gray_background, mask=background_mask)
# Combine the detected plants and the new background
final_image_with_gray_bg = cv2.add(result, background_part)

# Saving the images
cv2.imwrite('final_mask_raw.jpg', final_mask)
cv2.imwrite('denoised_mask.jpg', denoised_mask)
cv2.imwrite('result_on_gray_denoised.jpg', final_image_with_gray_bg)