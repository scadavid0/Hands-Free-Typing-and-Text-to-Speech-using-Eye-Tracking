# getavgs.py:
 # file to calculate the averages from the timedata.txt file, which contains time elapsed
 # for the main functions and methods in the gaze tracking code for a number of iterations.

import re

def calculate_averages(filename):
    # Dictionary to store the sum and count of each parameter
    parameters = {
        'get camframe': [],
        'process facemesh': [],
        'get landmarks': [],
        'set gaze vars': [],
        'mediapipe gaze refresh': [],
        'pupil transform': [],
        'EMAFilter.filter()': [],
        'display': [],
        'refresh': []
    }

    # Regular expression pattern to match parameter lines
    pattern = r'(\w+(?:\s+\w+)*?):\s+([\d.]+)\s+ms'

    numrefreshes = 0
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() == "-------new refresh------":
                numrefreshes += 1
                continue
            match = re.match(pattern, line.strip())
            if match:
                param, value = match.groups()
                if param in parameters:
                    parameters[param].append(float(value))
    print("numrefreshes:",numrefreshes)

    # Calculate averages
    averages = {param: sum(values) / len(values) if values else 0 
                for param, values in parameters.items()}

    return averages

# Usage
filename = 'timedata.txt'  # Replace with your actual filename
results = calculate_averages(filename)

# Print results
print("Average values for each parameter:")
for param, avg in results.items():
    print(f"{param}: {avg:.4f} ms")