# HighwayEnv

Modify the gray scale obervation function in the observation.py. Right now the observation contains both gray image and the kinematics observation. Besides that, I have also tested the variables: (1) scaling, which is a zoom in/out index with respect to the ego car as the center, i.e., meter to pixle ratio; (2) centering_position, which is a ration for the position of ego car's center, for example, [0.3, 0.5], that is 0.3*screen_width, 0.5*screen_height.
