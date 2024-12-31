# analyze_error.py:
# plots the error measurements compared to eight reference points (example in Team-4-report.pdf).
# also prints the mean and std of error distance.
# uncomment the target.measure_error() line in example.py to get error_measure.npz.

import numpy as np
import cv2

data = np.load("error_measure.npz")

window_frame_width    = 2560 # NOTE: adjust based on your computer screen dimensions.
window_frame_height   = 1500

measuring_pins        = [[0.9,0.1],[0.1,0.1],[0.1,0.9],[0.9,0.9],[0.7,0.3],[0.3,0.3],[0.3,0.7],[0.7,0.7]]
measuring_points      = np.array(measuring_pins) * np.array([window_frame_height,window_frame_width])
measuring_points      = measuring_points.astype(np.int32)
measuring_point_num   = measuring_points.shape[0]
measuring_frame_count = 120
gap_frame_count       = 40

target_data     = data['target'].reshape((measuring_point_num,measuring_frame_count,2))
estimation_data = data['estimation'].reshape((measuring_point_num,measuring_frame_count,2))

total_mean_error = data['mean']
total_std_eror   = data['std']

mean_estimation  = np.mean(estimation_data,axis=1)
target_points    = target_data[:,0,:]

bias = mean_estimation - target_points
std_per_point = np.std(estimation_data,axis=1)

print(bias)
print(std_per_point)

def draw_cross(frame,x,y,color,size,thickness,pad_size):
    x = int(x + pad_size)
    y = int(y + pad_size)

    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

def draw_boundry(frame,data,pad_size):
    data     += pad_size
    data      = np.flip(data,axis=1)
    data_mean = np.mean(data, axis=0)
    data_cov  = np.cov(data, rowvar=False)

    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(data_cov)

    # Sort eigenvalues and eigenvectors by descending eigenvalues
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate the angle of rotation of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Define the axes lengths (scale eigenvalues for desired confidence level, e.g., 95%)
    # chi_square_val = 5.991  # 95% confidence interval for 2D data
    chi_square_val = 2  # 95% confidence interval for 2D data
    axes_lengths = 2 * np.sqrt(chi_square_val * eigenvalues)

    for point in data:
        cv2.circle(frame, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 0), thickness=-1)

    # Draw the ellipse
    ellipse_center = (int(data_mean[0]), int(data_mean[1]))
    axes = (int(axes_lengths[0]), int(axes_lengths[1]))
    cv2.ellipse(frame, ellipse_center, axes, angle, 0, 360, (0, 255, 0), 2)

def draw_arrow(frame,target,estimation,pad_size):
    target     += pad_size
    estimation += pad_size
    cv2.arrowedLine(frame, (int(target_points[i,1]),int(target_points[i,0])), (int(mean_estimation[i,1]),int(mean_estimation[i,0])), color=(0,0,0), thickness=2, tipLength=0.3)

def draw_screen_boundry(frame,window_frame_width,window_frame_height,pad_size):
    point0 = (pad_size                   ,pad_size)
    point1 = (pad_size                   ,pad_size+window_frame_height)
    point2 = (pad_size+window_frame_width,pad_size+window_frame_height)
    point3 = (pad_size+window_frame_width,pad_size)

    cv2.line(frame, point0, point1, (0,0,0), thickness=2)
    cv2.line(frame, point1, point2, (0,0,0), thickness=2)
    cv2.line(frame, point2, point3, (0,0,0), thickness=2)
    cv2.line(frame, point3, point0, (0,0,0), thickness=2)

pad_size  = 300
frame = np.ones((pad_size + window_frame_height + pad_size, pad_size + window_frame_width + pad_size, 3), np.uint8) * 255
# # Draw the measuring points
draw_screen_boundry(frame,window_frame_width,window_frame_height,pad_size)
for i in range(measuring_point_num):
    # Draw measuring point
    draw_cross(frame,target_points[i,1],target_points[i,0],color=(0,0,255),size=60,thickness=3,pad_size=pad_size)
    # Draw estimation Mean
    draw_cross(frame,mean_estimation[i,1],mean_estimation[i,0],color=(255,0,0),size=40,thickness=3,pad_size=pad_size)
    # Draw arrow
    draw_arrow(frame,target_points[i,:],mean_estimation[i,:],pad_size=pad_size)
    draw_boundry(frame,estimation_data[i,:,:],pad_size=pad_size)
cv2.imwrite("probability_ellipse.png", frame)
# cv2.imshow("Reference", frame)
# cv2.waitKey(0)


print("Mean of error distance = ",data['mean'])
print("Std of error distance  = ",data['std'])