import imageio
import numpy as np

# Function to read an .hdr file
def read_hdr_image(file_path):
    # Read the .hdr file
    hdr_image = imageio.v3.imread(file_path)
    return hdr_image

# Example usage
if __name__ == "__main__":
    hdr_file_path = './out/spot/mesh/probe.hdr'  # Replace with the path to your .hdr file
    hdr_image = read_hdr_image(hdr_file_path)
    normalized_image = np.clip(hdr_image / np.max(hdr_image), 0, 1) * 255
    normalized_image = normalized_image.astype(np.uint8)
    # Print some information about the image
    # Display the image using matplotlib
    imageio.imwrite("light.png",normalized_image)  # Normalize the image for display
    
