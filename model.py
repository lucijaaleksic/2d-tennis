import matplotlib.pyplot as plt
from PIL import Image
import math
from ultralytics import YOLO
import cv2
import re

from utils.court_keypoints import *
from utils.geometry import *
from utils.players import *

kp_model = YOLO('./models/court_keypoints/best2.pt')
yolo_model = YOLO('.models/yolo/yolov5s.pt')

## Court dimensions in meters
court_keypoint1 = (0, 23.78) # Top left corner
court_keypoint2 = (10.97, 23.78) # Top right corner
court_keypoint3 = (10.97, 0) # Bottom right corner
court_keypoint4 = (0, 0) # Bottom left corner
court_pts = np.array([court_keypoint1, court_keypoint2, court_keypoint3, court_keypoint4])

class Model:
    def __init__(self):
        self.kp_model = kp_model
        self.yolo_model = yolo_model
 
        self.kps = None
        self.H = None

        self.player1 = None
        self.player2 = None
        self.ball = None # TODO

        self.tracker1 = None
        self.tracker2 = None
        self.tracker_ball = None

        self.initialized = False

    
    def initialize_trackers(self, img_path, player1_bbox, player2_bbox):
        """Initialize trackers with bounding boxes of detected players."""
        self.tracker1 = cv2.TrackerMIL()
        self.tracker2 = cv2.TrackerMIL()

        # Initialize tracker with the initial bounding box
        img = cv2.imread(img_path)

        player1_bbox = (int(player1_bbox[0]), int(player1_bbox[1]), int(player1_bbox[2]), int(player1_bbox[3]))
        player2_bbox = (int(player2_bbox[0]), int(player2_bbox[1]), int(player2_bbox[2]), int(player2_bbox[3]))
        self.tracker1.init(img, player1_bbox)
        self.tracker2.init(img, player2_bbox)

    def keypoints_sanity_check(self, kps):
        # check if it has any numbers
        kp = str(kps[0][1])
        if bool(re.search(r'\d', kp)):
            # sort by the highest y value
            kps = sorted(kps.tolist(), key=lambda x: x[1], reverse=True)
            tops = kps[:2]
            bottoms = kps[2:]

            # check if 2 tops are relatively close to each other
            if abs(1 - tops[0][1] / tops[1][1]) < 0.05 and abs(1 - bottoms[0][1] / bottoms[1][1]) < 0.05:
                return True
            
            # check if 2 bottoms are relatively close to each other
            if abs(1 - tops[0][1] / tops[1][1]) < 0.05 and abs(1 - bottoms[0][1] / bottoms[1][1]) < 0.05:
                return True
            
            return False

        else: 
            # 1) if it contains all 4 points
            kps_set = set([x[1] for x in kps])
            if (len(kps_set) == 4) and "Unknown" not in kps_set:
                # 2) if the top and bottom points are relaitvely in the y-axis
                tops = [x[0] for x in kps if x[1] == "Top left corner" or x[1] == "Top right corner"]
                bottoms = [x[0] for x in kps if x[1] == "Bottom left corner" or x[1] == "Bottom right corner"]
                # 5 % tolerance
                if abs(1 - tops[0][1] / tops[1][1]) < 0.05 and abs(1 - bottoms[0][1] / bottoms[1][1]) < 0.05:
                    return True
            return False

    def refine_keypoints(self, img_path):
        nkps, kps = get_frame_keypoints(self.kp_model, img_path)

        if self.kps is None:
            if not self.keypoints_sanity_check(nkps): # if the refined keypoints are not valid
                # wherever the new keypoints are missing, use the old ones
                for location in ["Bottom left corner", "Bottom right corner", "Top left corner", "Top right corner"]:
                    if location in set([x[1] for x in nkps]):
                        # remove that location from kps and add the one from nkps
                        kps = [x for x in kps if x[1] != location]
                        kps.append([x[0] for x in nkps if x[1] == location][0])

        return kps
    
    def predict(self, img_path):
        if not self.initialized:
            # Get the keypoints
            self.kps = self.refine_keypoints(img_path)
            if self.kps is None:
                print("Invalid keypoints")
                return None
            
            # Get the court geometry
            # 1) Order the keypoints in the correct order
            ordered_kps = []
            for kp in self.kps:
                if kp[1] == "Top left corner":
                    ordered_kps.append(kp)
                elif kp[1] == "Top right corner":
                    ordered_kps.append(kp)
                elif kp[1] == "Bottom right corner":
                    ordered_kps.append(kp)
                elif kp[1] == "Bottom left corner":
                    ordered_kps.append(kp)

            # 2) Get the court geometry
            self.H = get_homography(np.array([x[0] for x in ordered_kps]), court_pts)
            
            # Get the players
            players = get_players(self.yolo_model, self.kps, img_path)
            boxes = [x[2].tolist()[:4] for x in players] # TODO used for tracking

            if len(players) != 2:
                print("Invalid number of players")
                return None
            
            # Initialize the trackers
            # self.initialize_trackers(img_path, boxes[0], boxes[1])
            self.player1 = project_point([players[0][0], players[0][1]], self.H)
            self.player2 = project_point([players[1][0], players[1][1]], self.H)

        else:
            img = cv2.imread(img_path)
            # use the tracking for corners and players - compare the new points with the old ones and itnerpolate
            success1, player1_bbox = self.tracker1.update(img)
            success2, player2_bbox = self.tracker2.update(img)

            if success1 and success2:
                # Transform tracker coordinates to court coordinates
                self.player1 = project_point(player1_bbox, self.H)
                self.player2 = project_point(player2_bbox, self.H)
            else:
                print("Tracker lost player, reinitializing...")
                self.initialized = False  # Re-run detection in the next frame


        return self.player1, self.player2 # TODO ball
    