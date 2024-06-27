import cv2
import numpy as np
from PIL import Image as PILImage

def draw_border(img, car_bbox, license_plate_bbox, license_plate_text, license_plate_score, thickness=10, line_length=50):
    """
    Draws borders around the car bounding box and the license plate bounding box on the given image.

    Args:
        img (numpy.ndarray): The input image.
        car_bbox (tuple): The coordinates of the car bounding box in the format (x1, y1, x2, y2).
        license_plate_bbox (tuple): The coordinates of the license plate bounding box in the format (x1, y1, x2, y2).
        license_plate_text (str): The text to be displayed on the license plate.
        license_plate_score (float): The score associated with the license plate.
        thickness (int, optional): The thickness of the border lines. Defaults to 10.
        line_length (int, optional): The length of the lines extending from the corners. Defaults to 50.

    Returns:
        numpy.ndarray: The image with the borders drawn.
    """
    car_color = (0, 255, 0)  # Green color in BGR format
    license_plate_color = (0, 0, 255)  # Red color in BGR format

    x1, y1, x2, y2 = map(int, car_bbox)
    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, license_plate_bbox)

    # Draw the border for car bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), car_color, thickness)

    # Draw lines extending from corners
    cv2.line(img, (x1, y1), (x1 + line_length, y1), car_color, thickness)  # Top left corner
    cv2.line(img, (x1, y1), (x1, y1 + line_length), car_color, thickness)  # Top left corner

    cv2.line(img, (x2, y1), (x2 - line_length, y1), car_color, thickness)  # Top right corner
    cv2.line(img, (x2, y1), (x2, y1 + line_length), car_color, thickness)  # Top right corner

    cv2.line(img, (x1, y2), (x1 + line_length, y2), car_color, thickness)  # Bottom left corner
    cv2.line(img, (x1, y2), (x1, y2 - line_length), car_color, thickness)  # Bottom left corner

    cv2.line(img, (x2, y2), (x2 - line_length, y2), car_color, thickness)  # Bottom right corner
    cv2.line(img, (x2, y2), (x2, y2 - line_length), car_color, thickness)  # Bottom right corner

    # Draw the license plate bounding box
    cv2.rectangle(img, (lp_x1, lp_y1), (lp_x2, lp_y2), license_plate_color, 2)

    # Add text for license plate text
    (w, h), _ = cv2.getTextSize(
        license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    # Add background for license plate text
    cv2.rectangle(img, (lp_x1, lp_y1 - 20), (lp_x1 + w, lp_y1), license_plate_color, -1)
    cv2.putText(img, license_plate_text, (lp_x1, lp_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    # Add text for license plate score
    score_text = f"Score: {license_plate_score:.2f}"
    (w, h), _ = cv2.getTextSize(
        score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (lp_x1, lp_y2 + 20), (lp_x1 + w, lp_y2), license_plate_color, -1)
    cv2.putText(img, score_text, (lp_x1, lp_y2 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    return img

# res = {
#     'car': {'bbox': [375, 684, 718, 990]}, 
#     'license_plate': {
#         'bbox': [500, 886, 597, 923], 
#         'text': 'NA13NRU', 
#         'bbox_score': 0.6263312101364136, 
#         'text_score': 0.8679459956177277
#     }
# }

def image_visualization(image, res):
    """
    Visualizes the image with bounding boxes and crops it to the car bounding box (if needed).

    Args:
        image (str): The image file.
        res (dict): The dictionary containing the detection results.

    Returns:
        tuple: A tuple containing the original PIL image and the PIL image with bounding boxes.
    """
    # Extract the bounding box and text information from the detection results
    car_bbox = res['car']['bbox']
    license_plate_bbox = res['license_plate']['bbox']
    license_plate_text = res['license_plate']['text']
    license_plate_score = res['license_plate']['text_score']
    
    # Open image using PIL
    image_pil = PILImage.open(image)
    
    # Convert PIL image to numpy array (RGB)
    image_rgb = np.array(image_pil)
    
    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes on the BGR image
    image_with_boxes = draw_border(image_bgr, car_bbox, license_plate_bbox, license_plate_text, license_plate_score)
    
    # Convert BGR back to RGB for PIL
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Convert back to PIL image
    image_with_bbox = PILImage.fromarray(image_with_boxes_rgb)
    
    # Crop to car bounding box (if needed)
    # image_with_bbox = image_with_bbox.crop(car_bbox)
    
    return image_pil, image_with_bbox