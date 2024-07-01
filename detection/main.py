from ultralytics import YOLO
import numpy as np
from PIL import Image as PILImage
from .util import read_license_plate, upscale_license_plate
from .visualizeimg import image_visualization
from .models.model import VehicleModel
vehicles = [2, 3, 5, 7]

# Load models
coco_model = YOLO("./weights/yolov8n.pt", task="detect")
license_plate_detector = YOLO("./weights/best.pt")


def detect_license_plate(image):
    """
    Detects the license plate in the given image.

    Args:
        image: The input image to be processed.

    Returns:
        A tuple containing the coordinates (x1, y1, x2, y2), score, and class ID of the detected license plate.
    """
    # Use the license plate detector to predict the license plate
    results = license_plate_detector.predict(source=image, conf=0.25)

    # Get the bounding box coordinates of the first detected license plate
    x1, y1, x2, y2, score, class_id = results[0].boxes.data.tolist()[0]

    # Return the license plate coordinates, score, and class ID
    return x1, y1, x2, y2, score, class_id


def validate_number_plate_and_emission(number_plate):
    """
    Validates a given number plate.

    Args:
        number_plate (str): The number plate to be validated.

    Returns:
        True if the number plate is valid, otherwise False.
    """
    try:
        vehicle = VehicleModel.get(VehicleModel.number_plate == number_plate)
        print(vehicle,"Vehicle found")
        return vehicle.emission_done
    except VehicleModel.DoesNotExist:
        print("Vehicle not found")
        return None

def recognize_number_plate_and_validate(image):
    """
    Recognize the number plate from the given image and validate it using the validation function.

    Args:
        image (str): The image file.

    Returns:
        str: The recognized number plate if it is valid, otherwise an error message.
    """
    # Open image using PIL
    image_pil = PILImage.open(image)

    # Detect the license plate
    license_plate = detect_license_plate(image_pil)
    x1, y1, x2, y2, bbox_score, class_id = license_plate
    print(license_plate)
    if license_plate is not None:
        # Recognize the license plate number
        upscaled_image = upscale_license_plate(image_pil, license_plate[:4])
        number_plate, text_score = read_license_plate(upscaled_image)
        print(number_plate, text_score)
        res = {
            "car": {"bbox": [1,1,1,1]},
            "license_plate": {
                "bbox": [x1, y1, x2, y2],
                "text": number_plate,
                "bbox_score": bbox_score,
                "text_score": text_score,
            },
        }
        emission = validate_number_plate_and_emission(number_plate)
        if number_plate is not None:
            if emission is not None:
                if emission:
                    print("Here")
                    image_pil, image_with_bbox = image_visualization(image, res)
                    return image_pil, image_with_bbox, f"Number Plate:{number_plate}\nEmission test done"
                else:
                    return image, image ,f"Number Plate:{number_plate}\nEmission test not done"
            return image, image, "Vehicle not found"
        else:
            return image, image, "Number plate not recognized"
    return image, image, "License plate not detected" 
