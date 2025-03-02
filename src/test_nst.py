import cv2
from nst import perform_style_transfer  # Import your function from the main file

# Define paths
CONTENT_PATH = "images/content.jpg"
STYLE_PATH = "images/style.jpg"
OUTPUT_DIR = "images/output"

# Load images
content_img = cv2.imread(CONTENT_PATH)
style_img = cv2.imread(STYLE_PATH)

# Ensure images are loaded
if content_img is None or style_img is None:
    print("Error: Content or style image not found! Check paths.")
else:
    print("Images loaded successfully.")

# Run Style Transfer
output_image_path = perform_style_transfer(content_img, style_img, OUTPUT_DIR)

# Display the output image
output_img = cv2.imread(output_image_path)
cv2.imshow("Stylized Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
