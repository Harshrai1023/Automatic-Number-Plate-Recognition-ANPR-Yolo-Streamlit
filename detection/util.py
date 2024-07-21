# Initialize the OCR reader
import string
import easyocr
from RealESRGAN import RealESRGAN
import numpy as np
import torch
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

model_scale = "4" #@param ["2", "4", "8"] {allow-input: false}

model = RealESRGAN(device, scale=int(model_scale))
model.load_weights(f'./weights/RealESRGAN_x4.pth')

reader = easyocr.Reader(["en"], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5","L":"4"}

dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Check the length of the license plate text
    length = len(text)
    if length != 9 and length != 10:
        print("Length invalid", length)
        return False

    # CCNNCONNNN format
    if length == 10:
        if (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
            and (
                text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()
            )
            and (
                text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[2] in dict_char_to_int.keys()
            )
            and (
                text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()
            )
            and (
                text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()
            )
            and (
                text[6] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[7] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[8] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[9] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
        ):
            return True
    elif length == 9:
        if (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
            and (
                text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()
            )
            and (
                text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[2] in dict_char_to_int.keys()
            )
            and (
                text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()
            )
            and (
                text[5] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[6] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[7] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[8] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
        ):
            return True
    else:
        return False

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    # CCNNCONNNN
    license_plate_ = ""

    # Check the length of the license plate text
    if len(text) == 10:
        # Mapping for a 10-character license plate
        mapping = {
            0: dict_int_to_char,
            1: dict_int_to_char,
            2: dict_char_to_int,
            3: dict_char_to_int,
            4: dict_int_to_char,
            5: dict_int_to_char,
            6: dict_char_to_int,
            7: dict_char_to_int,
            8: dict_char_to_int,
            9: dict_char_to_int,
        }
    else:
        # Mapping for a 9-character license plate
        mapping = {
            0: dict_int_to_char,
            1: dict_int_to_char,
            2: dict_char_to_int,
            3: dict_char_to_int,
            4: dict_int_to_char,
            5: dict_char_to_int,
            6: dict_char_to_int,
            7: dict_char_to_int,
            8: dict_char_to_int,
        }

    # Iterate over the mapping keys
    for j in mapping.keys():
        # Check if the character at index j in the text is in the mapping dictionary
        if text[j] in mapping[j].keys():
            # If it is, append the mapped character to the license_plate_ string
            license_plate_ += mapping[j][text[j]]
        else:
            # If it is not, append the original character to the license_plate_ string
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
               If the license plate text cannot be read, None is returned for both elements of the tuple.
    """

    # Perform text detection on the license plate crop
    detections = reader.readtext(np.array(license_plate_crop))

    for detection in detections:
        bbox, text, score = detection

        # Preprocess the detected text
        text = ''.join(e for e in text if e.isalnum())

        # Print the detected text and its confidence score
        print("Utils file: " + text, score)

        # Check if the license plate text complies with the expected format
        isValid = license_complies_format(text)
        if isValid:
            # Format the license plate text and return it along with the confidence score
            return format_license(text), score, isValid
        # If no valid license plate text is found, return None for both elements of the tuple
        return text, score, isValid

def upscale_license_plate(image, bbox):
    """
    Upscale the license plate image using a super-resolution model.

    Args:
        image (PIL.Image.Image): The input license plate image.
        bbox (tuple): The bounding box coordinates of the license plate in the image.

    Returns:
        PIL.Image.Image: The upscaled binary license plate image.

    Raises:
        None

    """
    # Crop the license plate from the image
    image = image.convert('RGB').crop(bbox)

    # Upscale the license plate image using a super-resolution model
    sr_image = model.predict(np.array(image))

    # Save the upscaled image and normal image
    image = cv2.resize(np.array(image), (0,0), fx=4, fy=4) 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./detection/images/crop.png", image)
    sr_image.save("./detection/images/upscaled.png")

    # Convert the upscaled image to binary using a threshold value
    threshold_value = 120
    binary_image = sr_image.convert("L").point(lambda p: 255 if p < threshold_value else 0)

    # Save the binary image
    binary_image.save("./detection/images/upscaled_binary.png")

    # Print a message indicating that the process is finished
    print(f'Finished! Image saved')

    # Return the binary image
    return binary_image
