import cv2
import numpy as np
import pytesseract

# Set path for Tesseract executable (if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to preprocess the image (grayscale, blur, edge detection)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# Function to detect the largest contour that approximates a rectangle (document)
def get_document_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Sort by area, keep largest
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # If it's a 4-point contour, assume it's the document
            return approx
    return None

# Function to apply perspective transform to get a top-down view of the document
def four_point_transform(image, points):
    rect = np.array(points, dtype="float32")
    (tl, tr, br, bl) = rect

    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Enhanced function to extract text from the image using Tesseract with improved preprocessing
def extract_text(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding for better OCR performance
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Optionally apply dilation to clarify text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Extract text using Tesseract with the English language setting
    text = pytesseract.image_to_string(dilated, lang='eng')
    return text

# Main function to handle image input and document extraction
def process_document_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return
    
    # Preprocess the image to find edges
    edges = preprocess_image(image)
    
    # Find the document contour
    document_contour = get_document_contour(edges)

    if document_contour is not None:
        # Apply perspective transformation to get a top-down view of the document
        transformed_image = four_point_transform(image, document_contour.reshape(4, 2))
        cv2.imshow("Transformed Document", transformed_image)  # Show the transformed document

        # Perform OCR on the transformed image
        text = extract_text(transformed_image)
        print("Extracted Text (English):\n", text)
    else:
        # If no contour is found, perform OCR directly on the original image
        print("No document detected, extracting text from original image.")
        text = extract_text(image)
        print("Extracted Text (English):\n", text)

    # Display the original and edge-detected images for reference
    cv2.imshow("Original Image", image)
    cv2.imshow("Edge Detection", edges)

    # Wait until a key is pressed and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the function on your image
image_path = '/mnt/data/Computer-Paragraph.png'  # Path to the uploaded image
process_document_image(image_path)
