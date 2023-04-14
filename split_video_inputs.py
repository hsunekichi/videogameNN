import cv2
import sys
import numpy as np
import os
import shutil

def split_inputs(directory, number):

    # Open the input text file for reading
    with open(directory+'/logs/log'+number+'.txt', 'r') as f:
        # Read all the lines into a list
        lines = f.readlines()

    # Get the number of lines and calculate the midpoint
    num_lines = len(lines)
    midpoint = num_lines // 2

    # Split the lines into two halves
    lines1 = lines[:midpoint]
    lines2 = lines[midpoint:]

    # Open the output text files for writing
    with open(directory+'/logs/log'+number+'_part1.txt', 'w') as f1, open(directory+'/logs/log'+number+'_part2.txt', 'w') as f2:
        # Write the first half of lines to output file 1
        for line in lines1:
            f1.write(line)

        # Write the second half of lines to output file 2
        for line in lines2:
            f2.write(line)



def split_video(directory, number):
    # Read the input video file
    input_video = cv2.VideoCapture(directory+'/videos/vid'+number+'.mp4')

    # Get the video properties
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the midpoint of the video duration
    midpoint = total_frames // 2

    # Create the output video writers
    output_video_1 = cv2.VideoWriter(directory+'/videos/vid'+number+'_part1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    output_video_2 = cv2.VideoWriter(directory+'/videos/vid'+number+'_part2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))


    frames = np.zeros((total_frames, int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)

    print("Loading frames...")
    for i in range(total_frames):
        ret, frame = input_video.read()
        if not ret:
            break
        frames[i] = frame

    print("Splitting frames...")
    vid1 = frames[:midpoint]
    vid2 = frames[midpoint:]

    print("Writing frames...")
    for frame_1 in vid1:
        output_video_1.write(frame_1)

    for frame_2 in vid2:
        output_video_2.write(frame_2)

    # Release the input and output video objects
    input_video.release()
    output_video_1.release()
    output_video_2.release()



def split_video_low_memory(directory, number):
        
    # Read the input video file
    input_video = cv2.VideoCapture(directory+'/videos/vid'+number+'.mp4')

    # Get the video properties
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the midpoint of the video duration
    midpoint = total_frames // 2

    # Create the output video writers
    output_video_1 = cv2.VideoWriter(directory+'/videos/vid'+number+'_part1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    output_video_2 = cv2.VideoWriter(directory+'/videos/vid'+number+'_part2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for i in range(total_frames):

        if i % 1000 == 0:
            print("Frame "+str(i)+"/"+str(total_frames))

        # Read the next frame
        frame = input_video.read()[1]

        if i < midpoint:
            output_video_1.write(frame)
        else:
            output_video_2.write(frame)

    # Release the input and output video objects
    input_video.release()
    output_video_1.release()
    output_video_2.release()



## ------------------------ MAIN ------------------------ ##

if __name__ == '__main__':
    directory = "datasets/"+sys.argv[1]
    number = sys.argv[2]
    
    low_memory = False
    if len(sys.argv) > 3 and sys.argv[3] == "low_memory":
        low_memory = True


    split_inputs(directory, number)
    
    if low_memory:
        split_video_low_memory(directory, number)
    else:
        split_video(directory, number)

    inputs_path = directory+'/logs/log'+number+'.txt'
    video_path = directory+'/videos/vid'+number+'.mp4'

    new_inputs_path = directory+'/logs/old/log'+number+'.txt'
    new_video_path = directory+'/videos/old/vid'+number+'.mp4'

    # Create the destination directory if it doesn't exist
    if not os.path.exists(directory+'/logs/old'):
        os.makedirs(directory+'/logs/old')

    # Create the destination directory if it doesn't exist
    if not os.path.exists(directory+'/videos/old'):
        os.makedirs(directory+'/videos/old')

    # Check if the file exists before deleting it
    if os.path.exists(inputs_path):
        # Move the file to the destination directory
        shutil.move(inputs_path, new_inputs_path)

    # Check if the file exists before deleting it
    if os.path.exists(video_path):
        # Move the file to the destination directory
        shutil.move(video_path, new_video_path)

