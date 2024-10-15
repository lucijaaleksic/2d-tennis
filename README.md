# 2d-tennis
![architecture](https://github.com/lucijaaleksic/2d-tennis/blob/main/tennis-architecture.png?raw=true)
## 1. References
- *[Accurate Tennis Court Line Detection on Amateur Recorded Matches](https://arxiv.org/pdf/2404.06977)* This paper will guide the detection of court boundaries, allowing for accurate transformations from a 3D video to a 2D plane.
- *[DeepBall: Deep Neural-Network Ball Detector](https://arxiv.org/pdf/1902.07304v1)* This paper provides insights into ball detection and tracking using a neural network, which will be a key component for tracking the tennis ball during rallies.

## 2. Project Topic
The goal of the project is to process a video of a tennis match or rally and output a 2D match view. This visualization will show the players and the ball on a top-down 2D version of the tennis court, showing player movements and ball trajectory in real-time.

## 3. Project Type
This is basically an object detection focused project which will use different models to detect and track objects.

## 4. Written Summary
### a. Project Description and Approach
The project is about transforming a video of a tennis rally into a simplified 2D version. The pipeline consists of three main components:

1. Court Detection: A model will detect the court's corners and lines from the video. Using the detected corners, a homography matrix will be computed to map the 3D court into a 2D plane.
2. Player and Ball Detection: A pre-trained deep learning model (such as the one described in the "DeepBall" paper) will be used to detect and track the players and the ball throughout the rally.
3. Player and Ball Tracking: An algorithm will be used to track and improve detection results over the player and ball movements over frames.
4. 2D Projection: The detected positions of the players and ball will be transformed using the homography matrix.
The final output will be a 2D animation that mimics the movements and ball trajectories seen in the original video.

### b. Dataset
Court Detection: Videos of tennis rallies from Australian Open will be segmented into frames and labeled.
Player and Ball Detection: Pre-trained models (like DeepBall) will be used.

### c. Work Breakdown Structure
1. Research and Dataset Collection (2-3 days)
Research relevant datasets for tennis matches.
Collect and clean videos for training court detection and validation.
Explore existing annotations for players and ball positions.

3. Court Detection and Homography Calculation (5-7 days)
Implement court detection model using insights from the "Accurate Tennis Court Line Detection" paper.
Calculate the homography matrix based on detected court corners.
Validate and fine-tune the model using collected match videos.

4. Player and Ball Detection using Pre-trained Models (4-5 days)
Integrate pre-trained models from "DeepBall" or similar sources.
Implement tracker to track the players and ball during the match or rally.

4. 2D Projection and Animation (5-7 days)
Transform the detected player and ball positions into 2D court coordinates using the homography matrix.
Build a simple graphical representation of the tennis court.
Develop an animation system to visualize the players and ball in 2D.

5. Integration and Testing (3-4 days)
Integrate the entire pipeline from video input to 2D output.
Test on multiple matches and adjust parameters for smoother performance.

6. Application Development (5-6 days)
Build a user-friendly application/demo to input a match video and output a 2D visualization.
Potentially add features like ball speed, player speed, out detection...

7. Report Writing and Presentation Preparation (2-3 days)
Write the final report, summarizing the methodology, challenges, and results.
Prepare a presentation, including visuals of the 2D animation and system architecture.

Total Estimated Time: 26-28 Days
