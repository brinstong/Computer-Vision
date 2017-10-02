# Computer-Vision-Assignments & Project

# Project Details
fall_detection_example.py
# This file is the source code of fall detection system

camera_fall_detection_example.py
# The file is modified version of fall_detection_example.py in order get a better camera based presentation
# Usage: Just run the file through terminal.
# When the camera is on, the system will react when single object have both rapid speed change and shape change(for example, use hand to test)


# main function is located at the end of the file
# function usage:

process_video(img_name, Output_path, Flag_video_generate, Flag_video_show)
# The core function to test
# img_name: video path, use camera if img_name == 0
# Output_path: Where to save the generated video if Flag_video_generate is true
# Flag_video_generate: generate video to Output_path if set to True
# Flag_video_show: show the output video while processing the sequences of images
# usage: given the video path, the function can generate a video sequence of the output of fall detection system. And the user can choose to save or show the video.


def process_camera():
# for presentation only
# This is better shown when run the "camera_fall_detection_example.py" file
# use computer camera as input source, and by default show the output video sequence

def process_one_video(dataset_name, i):
# for evaluation only
# usage: the dataset is located 'Output/' folder
# process one video given the dataset folder name and  video index

def process_dataset(dataset_name, size):
# for evaluation only
# usage: the dataset is located 'Output/' folder
# process all videos given the dataset folder name and the number of videos
