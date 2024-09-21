import cv2
import numpy as np
import matplotlib.pyplot as plt
def is_homogeneous(region, threshold):
    """Check if the region is homogeneous based on the intensity threshold."""
    min_val, max_val = np.min(region), np.max(region)
    return (max_val - min_val) <= threshold

def split_and_merge(image, threshold):
    """Segment the image by recursively splitting and merging regions."""
    
    def recursive_split(region):
        rows, cols = region.shape
        if rows <= 1 or cols <= 1:
            return np.zeros_like(region, dtype=np.uint8)
        
        if is_homogeneous(region, threshold):
            return np.ones_like(region, dtype=np.uint8)
        
        
        mid_row, mid_col = rows // 2, cols // 2
        
        
        top_left = region[:mid_row, :mid_col]
        top_right = region[:mid_row, mid_col:]
        bottom_left = region[mid_row:, :mid_col]
        bottom_right = region[mid_row:, mid_col:]
        
        
        segmented_quadrants = np.zeros_like(region, dtype=np.uint8)
        
       
        segmented_quadrants[:mid_row, :mid_col] = recursive_split(top_left)
        segmented_quadrants[:mid_row, mid_col:] = recursive_split(top_right)
        segmented_quadrants[mid_row:, :mid_col] = recursive_split(bottom_left)
        segmented_quadrants[mid_row:, mid_col:] = recursive_split(bottom_right)
        
        return segmented_quadrants

    def merge_regions(segmented):
        """Merge adjacent regions if they are similar."""
       
        return segmented

    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the region splitting and merging algorithm
    segmented_image = recursive_split(image)
    segmented_image = merge_regions(segmented_image)
    
    return segmented_image

image = cv2.imread(r"img.jpg", cv2.IMREAD_GRAYSCALE)
threshold =10
segmented_img = split_and_merge(image, threshold)
gray = np.array(image, dtype=np.uint8)
segmented_img = np.array(segmented_img, dtype=np.uint8)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')  
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(segmented_img, cmap='gray')
plt.title('Region Spliting and Merging')
plt.show()