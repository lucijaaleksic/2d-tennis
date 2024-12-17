from utils.geometry import point_distance
import math

def closest_people(people_positions, new_keypoints):
    """
    Find the closest two people to the baselines.

    Args:
        people_positions: list of people positions
        new_keypoints: list of keypoints of the court
    """
    closest_people = []
    baseline_top = []
    baseline_bottom = []
    for new_keypoint, location in new_keypoints:
        if "Bottom" in location:
            baseline_bottom.append(new_keypoint)
        else:
            baseline_top.append(new_keypoint)
    # sort the baselines points by x
    baseline_top = sorted(baseline_top, key=lambda x: x[0])
    baseline_bottom = sorted(baseline_bottom, key=lambda x: x[0])

    if len(baseline_top) == 0 or len(baseline_bottom) == 0:
        return closest_people
    
    if len(baseline_top) < 2 or len(baseline_bottom) < 2:
        print("Wrong location of the baseline points.")
        return closest_people
    
    # print("Baseline top", baseline_top)
    # print("Baseline bottom", baseline_bottom)

    # find closest person to the bottom baseline
    min_distance = math.inf
    closest_person = None
    for person in people_positions:
        distance = point_distance(person, baseline_bottom[0], baseline_bottom[1])
        if distance < min_distance:
            min_distance = distance
            closest_person = person
    closest_people.append(closest_person)

    # find closest person to the top baseline
    min_distance = math.inf
    closest_person = None
    for person in people_positions:
        distance = point_distance(person, baseline_top[0], baseline_top[1])
        if distance < min_distance:
            min_distance = distance
            closest_person = person
    closest_people.append(closest_person)

    return closest_people

def get_players(yolo_model, new_keypoints, image_path):
    """
    Returns the closest two people to the baselines.

    Args:
        yolo_model: YOLO model
        new_keypoints: list of keypoints of the court
        image_path: path to the image
    """
    results = yolo_model(image_path)
    people_positions = []

    for box in results[0].boxes.data:
        x1, y1, x2, y2, _, _ = box

        # player is defined by the center of his leg stance (bottom y center)
        center_x = (x1 + x2) / 2
        center_y = y2
        people_positions.append((center_x.item(), center_y.item(), box))

    players = closest_people(people_positions, new_keypoints)
    return players
