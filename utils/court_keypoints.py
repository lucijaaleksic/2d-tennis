import cv2
import numpy as np
from PIL import Image
import math

def process_crop(crop):
    """
    Process the crop to get the horizontal line of the baseline and mask of the court lines.

    Args:
        crop (np.array): The crop of the corner detected by the model.
    """
    crop = np.array(crop)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 120)
    lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, 30, 1);
    if lines is None:
        return None, None
    
    horizontal_line = lines[0]
    
    if horizontal_line is None:
        return None
    
    mask = np.zeros_like(edges)
    mask[edges > 0] = 255

    return horizontal_line, mask

def get_crop_keypoint(crop):
    """
    Get the exact keypoint of the corner detected by the model.

    Args:
        crop (np.array): The crop of the corner detected by the model.
    """
    horizontal_line, mask = process_crop(crop)

    if horizontal_line is None:
        return None, None
    
    y_value = horizontal_line[0][1]

    # get the horizontal information
    horizontal_slice = mask[y_value-2:y_value+2, :]
    white_pixels = np.where(horizontal_slice == 255)
    min_h_x = min(white_pixels[1])
    max_h_x = max(white_pixels[1])

    # get the top intersection
    top_intersect = mask[0, :]
    white_pixels = np.where(top_intersect == 255)
    if len(white_pixels[0]) > 0:
        min_t_x = min(white_pixels[0])
        max_t_x = max(white_pixels[0])
    else:
        min_t_x = -1
        max_t_x = -1

    keypoint = (0, 0)

    if min_t_x != -1 and max_t_x != -1: # bottom corners
        if min_h_x == 0:
            keypoint = [max_h_x, y_value]
        else:
            keypoint = [min_h_x, y_value]
    elif min_t_x == -1 and max_t_x == -1: # top corners
        if min_h_x == 0:
            keypoint = [max_h_x, y_value]
        else:
            keypoint = [min_h_x, y_value]

    return keypoint

def get_frame_keypoints(model, image_path, crop_factor=0.08): 
    """
    Get the refined keypoints of the court corners.

    Args:
        model (torch model): The model to detect the keypoints.
        image_path (str): The path to the image.
        crop_factor (float): Factor on how much to crop around the keypoints.
    """
    results = model(image_path)
    keypoints = results[0].keypoints.xy[0]

    original = Image.open(image_path)
    new_keypoints = []

    for keypoint in keypoints:
        x, y = float(keypoint[0]), float(keypoint[1])
        
        img_width, img_height = original.size
        crop_size = crop_factor * min(img_width, img_height)

        left = max(0, x - crop_size)  # Ensure the crop doesn't go out of bounds
        top = max(0, y - crop_size)
        right = min(img_width, x + crop_size)
        bottom = min(img_height, y + crop_size)

        crop = original.crop((left, top, right, bottom))

        crop_kp = get_crop_keypoint(crop)

        if crop_kp:
            crop_kp = (crop_kp[0] + left, crop_kp[1] + top)
            new_keypoints.append(crop_kp)
        else:
            new_keypoints.append([x, y])

    # classify keypoints as Bottom/Top Left/Right corner
    # find the mean between two tops and two bottoms
    ## sort the keypoints by y value    
    keypoints = keypoints.tolist()
    keypoints = sorted(keypoints, key=lambda x: x[1])
    tops = keypoints[:2]
    bottoms = keypoints[2:]

    # sort by x value
    tops = sorted(tops, key=lambda x: x[0])
    bottoms = sorted(bottoms, key=lambda x: x[0])

    keypoints = [ # TODO check
        (tops[0], "Top left corner"),
        (tops[1], "Top right corner"),
        (bottoms[1], "Bottom right corner"),
        (bottoms[0], "Bottom left corner")
    ]

    new_keypoints = sorted(new_keypoints, key=lambda x: x[1])
    tops = new_keypoints[:2]
    bottoms = new_keypoints[2:]

    tops = sorted(tops, key=lambda x: x[0])
    bottoms = sorted(bottoms, key=lambda x: x[0])

    new_keypoints = [
        (tops[0], "Top left corner"),
        (tops[1], "Top right corner"),
        (bottoms[1], "Bottom right corner"),
        (bottoms[0], "Bottom left corner")
    ]

    return new_keypoints, keypoints