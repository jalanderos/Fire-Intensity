"""
Thermal infrared image processing to determine Byram's intensity.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants, integrate


plt.style.use('ggplot')

# Temperature data from TIR Images
# Makeshift pixel temperatures
list1 = [[300, 600, 700, 500],
         [300, 500, 600, 800],
         [300, 400, 400, 400],
         [300, 400, 400, 400]]
arr1 = np.array(list1)
list2 = [[300, 600, 700, 500],
         [300, 600, 700, 500],
         [300, 400, 600, 700],
         [300, 400, 500, 600]]
arr2 = np.array(list2)
list3 = [[400, 600, 700, 500],
         [600, 700, 700, 500],
         [300, 600, 500, 500],
         [300, 600, 600, 600]]
arr3 = np.array(list3)
# Makeshift pixel temperature sequence
temp_time_array = np.array([arr1, arr2, arr3], dtype=float)
rows, cols = np.shape(temp_time_array[0])
time_array = np.array([1, 5, 10])
times, = np.shape(time_array)


# Preprocessing
# Pixel Resolution (GSD) in m
pixel_res = 1


# Fire Radiative Energy Density (FRED)
# Filter out low temperature pixels
FRED_temp_threshold = 500
FRPD_time_array = temp_time_array
FRPD_time_array[FRPD_time_array < FRED_temp_threshold] = 0

# Pixel FRPD
emissivity = 1
FRPD_time_array = emissivity * constants.Stefan_Boltzmann * FRPD_time_array ** 4 / 1000

# Pixel FRED
FRED_array = np.zeros((rows, cols))
for r in range(rows):
    for c in range(cols):
        FRED_array[r, c] = integrate.simpson(FRPD_time_array[:, r, c], time_array)


# Rate of Spread (ROS)
# Pixel arrival times
ROS_temp_threshold = 600
t_a_array = np.zeros((rows, cols))
for t in reversed(range(times)):
    for r in range(rows):
        for c in range(cols):
            if temp_time_array[t, r, c] >= ROS_temp_threshold:
                t_a_array[r, c] = time_array[t]

loc_time_list = []
for t in (range(times)):
    loc_array = np.argwhere(t_a_array == time_array[t])
    # loc_array[:, 0] = rows - loc_array[:, 0] - 1
    # loc_array[:, [0, 1]] = loc_array[:, [1, 0]]
    loc_time_list.append(loc_array)

# Fire line location sequence (c, r)
loc_time_array = np.array(loc_time_list)

# Reduce resolution to 50%
# code

# Pixel fire line normal angles
normal_angle_array = np.zeros((rows, cols))
for t in range(times):
    for r in range(rows):
        for c in range(cols):
            # Locate the 2 nearest active fire pixels
            if t_a_array[r, c] == time_array[t]:
                # Minimum bounds of square_t_a_array
                min_bounds = np.array([max(r - 1, 0), max(c - 1, 0)])

                # Array of the nearest arrival time pixels
                square_t_a_array = t_a_array[min_bounds[0]:r + 2, min_bounds[1]:c + 2]

                # Indices of the nearest active fire pixels
                near_af_pixels = np.argwhere(square_t_a_array == time_array[t])

                # Distances to the nearest active fire pixels
                distances = ((near_af_pixels[:, 0] + min_bounds[0] - r) ** 2 +
                             (near_af_pixels[:, 1] + min_bounds[1] - c) ** 2) ** 0.5
                sort_dist = np.sort(distances)

                # Indices of the 2 nearest  active fire pixels
                # Uncovered cases: 0 near pixels,
                # 3+ equally near_1 pixels, 2+ equally near_2 pixels
                if len(distances) > 2:
                    if sort_dist[1] == sort_dist[2]:
                        [near_1_ind, near_2_ind] =\
                            np.ravel(np.nonzero(distances == sort_dist[1]))
                    else:
                        near_1_ind = np.nonzero(distances == sort_dist[1])[0][0]
                        near_2_ind = np.nonzero(distances == sort_dist[2])[0][0]

                    # Locations of the 2 nearest active fire pixels
                    near_1_loc = near_af_pixels[near_1_ind, :] + min_bounds
                    near_2_loc = near_af_pixels[near_2_ind, :] + min_bounds

                    # Angle of line between the 2 nearest active fire pixels
                    fire_line_angle =\
                        np.arctan2([near_1_loc[0] - near_2_loc[0]],
                                   [near_2_loc[1] - near_1_loc[1]])

                    # Derive normal angle in (0, 180] degrees
                    if fire_line_angle <= -np.pi / 2:
                        normal_angle_array[r, c] = fire_line_angle + (1.5 * np.pi)
                    elif fire_line_angle <= np.pi / 2:
                        normal_angle_array[r, c] = fire_line_angle + np.pi / 2
                    else:
                        normal_angle_array[r, c] = fire_line_angle - np.pi / 2

# Pixel ROS
ROS_array = np.zeros((rows, cols))
for t in range(times - 1):
    dt = time_array[t + 1] - time_array[t]
    for r in range(rows):
        for c in range(cols):
            # Locate intersection of pixel normal and the next fire line
            # Pixel fire line normal angle
            normal_angle = normal_angle_array[r, c]
            if t_a_array[r, c] == time_array[t] and normal_angle > 0:
                # Points forming line through pixel fire line normal
                p1 = np.array([r + (rows * np.sin(normal_angle)),
                               c - (cols * np.cos(normal_angle))])
                p2 = np.array([r - (rows * np.sin(normal_angle)),
                               c + (cols * np.cos(normal_angle))])

                # Distances between pixels on the next fire line and line through normal
                distances = abs(np.around(
                    np.cross(p2 - p1, loc_time_array[t + 1] - p1) /
                    np.linalg.norm(p2 - p1), 5))

                # Intersection point of pixel normal and the next fire line
                if min(distances) < 1:
                    near_ind = np.nonzero(distances == distances.min())[0]
                    if len(near_ind) > 1:
                        p3 = np.average(loc_time_array[t + 1, near_ind], axis=0)
                    else:
                        p3 = loc_time_array[t + 1, near_ind[0]]

                    # Distance advanced by fire line pixel
                    dist_adv = pixel_res * ((r - p3[0]) ** 2 + (c - p3[1]) ** 2) ** 0.5

                    # Pixel ROS
                    ROS_array[r, c] = dist_adv / dt

                    # Plot normal lines
                    pt1 = np.array([r + (0.4 * np.sin(normal_angle)),
                                   c - (0.4 * np.cos(normal_angle))])
                    pt2 = np.array([r - (0.4 * np.sin(normal_angle)),
                                   c + (0.4 * np.cos(normal_angle))])
                    plt.plot([pt1[1], pt2[1]], [-pt1[0], -pt2[0]], color='black')

# Compile pixel ROS
# code


# Byram's Intensity
# Pixel radiative intensity
I_rad_array = FRED_array * ROS_array

# Radiative fraction
radF = 0.17

# Pixel total intensity
I_tot_array = I_rad_array / 0.17
for r in range(rows):
    row = I_tot_array[r]
    if not np.all(row == 0):
        I_tot_array[r] = np.where(row == 0, np.nan, row)

# Row median total intensity
I_tot_rows = np.nanmedian(I_tot_array, axis=1)

# print(t_a_array)
# print(normal_angle_array * 180 / np.pi)
# print(FRED_array)
# print(ROS_array)
# print(I_rad_array)
# print(I_tot_array)
print(I_tot_rows)

# Plot grid points
row_list = np.linspace(0, 1 - rows, rows)
col_list = np.linspace(0, cols - 1, cols)
x_grid, y_grid = np.meshgrid(col_list, row_list)
plt.plot(x_grid, y_grid, linestyle='', marker='o', markersize=4, color='black')

# Plot fire line over time
lines = True
for t in range(times):
    x = loc_time_array[t, :, 1]
    y = - loc_time_array[t, :, 0]
    label = f'{time_array[t]} s'
    if lines:
        plt.plot(x, y, linewidth=4, marker='s', markersize=10, label=label)
    else:
        plt.plot(x, y, linestyle='', marker='s', markersize=10, label=label)

plt.title('Fire Line Over Time')
plt.legend()
plt.xlim(-1, cols)
plt.ylim(-rows, 1)
plt.show()
