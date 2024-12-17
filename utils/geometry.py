import cv2
import numpy as np
import math

def get_homography(pts_src, pts_dst):
    """
    Function to get the homography matrix from source points to destination points.

    Args:
        pts_src: source points
        pts_dst: destination points
    """
    h, _ = cv2.findHomography(pts_src, pts_dst)
    h_inv = np.linalg.inv(h)

    return h

# player geometry
def project_point(point, H):
    """
    Function to project a point from the image to the court using the homography matrix.

    Args:
        point: the point to project
        H: the homography matrix
    """
    point = np.array([point[0], point[1], 1.0])  # Convert to homogeneous coordinates
    projected_point = np.dot(H, point)
    projected_point /= projected_point[2]  # Normalize

    return projected_point[:2]  # Return the (x, y) coordinates

def point_distance(point, line1, line2):
    """
    Function to calculate the distance between a point and a line segment.
    Used to find the closest person to the baselines.

    Args:
        point: the point
        line1: the first point of the line segment
        line2: the second point of the line segment
    """
    x, y, _ = point
    x1, y1 = line1
    x2, y2 = line2
    
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1

    if len_sq != 0:  # In case of a non-zero length line segment
        param = dot / len_sq

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return math.sqrt(dx * dx + dy * dy)
