from PIL import Image
import os
#pip install Pillow
# Define the path to the directory containing the original images
source_folder = '/Users/frenwd24/Desktop/AIPython/image_nn/sample_imp_images'

# Define the path to the directory where resized images will be saved
output_folder = '/Users/frenwd24/Desktop/AIPython/image_nn/resize_sample_imp_images'

# Desired dimensions
new_width = 40
new_height = 40

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the source folder
for file_name in os.listdir(source_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check for common image file extensions
        # Open an image file
        with Image.open(os.path.join(source_folder, file_name)) as img:
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
           
            # Save the resized image to the output folder
            img.save(os.path.join(output_folder, file_name))

print("All images have been resized and saved to", output_folder)
