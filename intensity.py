"""
Thermal infrared image processing to determine Byram's intensity.
"""

import cv2
import numpy as np
from scipy import constants, integrate


# Temperature data from TIR Images
# Makeshift pixel temperatures
list1 = [[300,600,700,500],
         [300,500,600,800],
         [300,400,600,700],
         [300,400,400,600]]
arr1 = np.array(list1)
list2 = [[300,600,700,500],
         [300,600,700,500],
         [300,400,600,700],
         [300,400,500,600]]
arr2 = np.array(list2)
list3 = [[400,600,700,500],
         [600,700,700,500],
         [300,600,500,500],
         [300,400,500,600]]
arr3 = np.array(list3)
# Makeshift pixel temperature sequence
temp_time_arr = np.array([arr1,arr2,arr3], dtype=float)
rows, cols = np.shape(temp_time_arr[0])
time_arr = np.array([1,5,10])
times, = np.shape(time_arr)


# Preprocessing


# Fire Radiative Energy Density (FRED)
# Filter out low temperature pixels
FRED_temp_threshold = 500
FRPD_time_arr = temp_time_arr
FRPD_time_arr[FRPD_time_arr < FRED_temp_threshold] = 0
# Convert pixel temp to pixel FRPD
emissivity = 1
FRPD_time_arr = emissivity * constants.Stefan_Boltzmann * FRPD_time_arr ** 4 /1000
# Convert pixel FRPD to pixel FRED
FRED_arr = np.zeros((rows, cols))
for r in range(rows):
    for c in range(cols):
        FRED_arr[r,c] = integrate.simpson(FRPD_time_arr[:,r,c],time_arr)


# Rate of Spread (ROS)
# Pixel arrival times
ROS_temp_threshold = 600
t_a_arr = np.zeros((rows, cols))
for t in reversed(range(times)):
    for r in range(rows):
        for c in range(rows):
            if temp_time_arr[t,r,c] >= ROS_temp_threshold:
                t_a_arr[r,c] = time_arr[t]
print(t_a_arr)

normal_arr = np.zeros((rows, cols))
for r in range(rows):
    for c in range(cols):
        if t_a_arr[r,c] != 0:
            min_bounds = np.array([max(r - 1, 0), max(c - 1, 0)])
            # Square array of the nearest arrival time pixels
            square_t_a_arr = t_a_arr[min_bounds[0]:r+2,min_bounds[1]:c+2]
            raw_nonzero = cv2.findNonZero(square_t_a_arr)
            # Nonzero locations flipped and squeezed to 2D array (shape: 10,2)
            nonzero = np.flip(np.squeeze(raw_nonzero), axis=1)
            # Distances to surrounding pixels
            distances = ((nonzero[:,0] - r) ** 2 +
                         (nonzero[:,1] - c) ** 2) ** 0.5
            sort_dist = np.sort(distances)
            # Indices of the 2 nearest pixels
                # Uncovered cases: 0 near pixels,
                # 3+ equally near_1 pixels, 2+ equally near_2 pixels
            if sort_dist[1] == sort_dist[2]:
                near_indices = np.where(distances == sort_dist[1])
                near_1_ind = near_indices[0][0]
                near_2_ind = near_indices[0][1]
            else:
                near_1_ind = np.where(distances == sort_dist[1])[0][0]
                near_2_ind = np.where(distances == sort_dist[2])[0][0]

            near_1_loc = nonzero[near_1_ind,:] + min_bounds
            near_2_loc = nonzero[near_2_ind, :] + min_bounds
            fire_line_slope = np.arctan2(np.array(near_2_loc[1],near_1_loc[1]),
                                         np.array(near_2_loc[0],near_1_loc[0])) * \
                              180 / np.pi
            normal_arr[r,c] = fire_line_slope + 90

            if r == 1 and c == 0:
                # print(near_1_loc, near_2_loc)
                print(fire_line_slope)
                # print(f'1st nearest {near_1_ind} 2nd nearest {near_2_ind}')
                # print(nonzero[near_1_ind,:] + min_bounds)
                # print(nonzero[near_2_ind,:])
                # print(distances)
                # near_1 = np.argmin(distances)
                # distances = np.delete(distances,near_1)
                # near_2 = np.argmin(distances)
                # print(near_1,near_2)

print(normal_arr)
# print(FRED_arr)
