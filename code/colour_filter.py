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

# Create a mask where pixels within the green range are white
plant_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# --- 3. Create a Mask to Remove the Top Portion ---
height, width, _ = original_image.shape
top_mask = np.zeros((height, width), dtype=np.uint8)

# Define the portion to keep (e.g., bottom 60% of the image)
mask_start_row = int(height * 0.40)
top_mask[mask_start_row:, :] = 255  # Set the bottom part to white

# --- 4. Combine the Masks ---
# The final mask is the intersection of the plant mask and the top mask
final_mask = cv2.bitwise_and(plant_mask, top_mask)

# --- 5. Apply the Final Mask to the Original Image ---
# This will show the original colors of the plants on a black background
result = cv2.bitwise_and(original_image, original_image, mask=final_mask)

# --- Bonus: For the gray background like in the example ---
# Create a gray background
gray_background = np.full(original_image.shape, (80, 80, 80), dtype=np.uint8)
# Invert the mask to select the background
background_mask = cv2.bitwise_not(final_mask)
# Apply the inverted mask to the gray image
background_part = cv2.bitwise_and(gray_background, gray_background, mask=background_mask)
# Combine the detected plants and the new background
final_image_with_gray_bg = cv2.add(result, background_part)


# Saving the images
# cv2.imwrite('final_mask.jpg', final_mask)
# cv2.imwrite('result_on_black.jpg', result)
# cv2.imwrite('result_on_gray.jpg', final_image_with_gray_bg)