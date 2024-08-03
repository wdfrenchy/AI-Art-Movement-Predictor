from PIL import Image
import os
import csv

# Define the path to the directory containing the images
source_folder = '/Users/frenwd24/Desktop/AIPython/image_nn/resize_sample_non-imp_images'

# Path to the CSV file where pixel data will be saved
output_csv = '/Users/frenwd24/Desktop/AIPython/image_nn/output_sample_pixels.csv'

# Open the CSV file for writing
with open(output_csv, 'a', newline='') as csvfile:
    pixel_writer = csv.writer(csvfile)
    
    # Loop through all files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check for common image file extensions
            # Open an image file
            with Image.open(os.path.join(source_folder, file_name)) as img:
                # Convert image to RGB (in case it's not)
                img = img.convert('RGB')
                
                # Get pixels from the image
                pixels = list(img.getdata())
                
                # Flatten the list of tuples and prepend "1" to represent the image type
                #pixel_row = [1]  # Start with the label - impressionism
                pixel_row = [0] # non impressionism
                pixel_row.extend(value for pixel in pixels for value in pixel)  # Add pixel values
                
                # Write to csv
                pixel_writer.writerow(pixel_row)

print("Pixel data has been written to", output_csv)
