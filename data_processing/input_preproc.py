import config as conf
import numpy as np



def discretize_line_input(line):    
    processed_line = [np.uint8]*(len(line)+conf.n_bidirectional_axis)
    
    i_output = 0
    for i_input in range(len(line)):     # Convert the data to binary

        if i_input < conf.n_bidirectional_axis:       # First two values are the X and Y axis movement
            if line[i_input] < -conf.threshold_input:
                processed_line[i_output] = 1
                processed_line[i_output+1] = 0
            elif line[i_input] > conf.threshold_input:
                processed_line[i_output] = 0
                processed_line[i_output+1] = 1
            else:
                processed_line[i_output] = 0
                processed_line[i_output+1] = 0
                
            i_output += 2
        else:
            if line[i_input] > conf.threshold_input:
                processed_line[i_output] = 1
            else:
                processed_line[i_output] = 0
            
            i_output +=1 

    return processed_line



def discretize_inputs(input_lines):

    numFloats = len(input_lines)*len(input_lines[0])
    processed_lines = [np.uint8]*numFloats

    for line in input_lines:
        processed_lines.append(discretize_line_input(line))

    return processed_lines
