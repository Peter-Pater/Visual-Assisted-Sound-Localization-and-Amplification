import math
import numpy as np 
import matplotlib.pyplot as plt
import tqdm
from .signals import *
import scipy.stats

def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points

def lookup(data, phase_diffs, frames, num_sources = 1):
    curr_locs = []
    second_locs = []
    third_locs = []
    for f in tqdm.tqdm(range(0, data.shape[1] - frames, frames//4)):
        frame_data = data[:, f:f+(frames)]
        phase_diff = []
        x_corrs = []
        count = 0

        expected_phases = []
        skip = None
        for i in range(0, frame_data.shape[0] - 1):
            for j in range(i+1, frame_data.shape[0]):
                x_1 = np.squeeze(np.array(frame_data[i]).T)
                x_2 = np.squeeze(np.array(frame_data[j]).T)
                skip, xcorr, expected_phase = find_delta_with_xcorr(x_1, x_2)
                if skip == True:
                    continue

                expected_phases.append(expected_phase)
                x_corrs.append(xcorr)
            if skip == True:
                continue
        if skip == True:
            continue
        print(expected_phases)
        x_corrs = np.array(x_corrs)
        
        # compare to sphere
        mid = x_corrs.shape[1] // 2
        phase_sums = {}
        curr_loc = None
        max_sum = 0
        for loc in phase_diffs:
            curr_sum = 0
            indices = phase_diffs[loc]
            for j in range(len(indices)):
                index = mid + indices[j]
                val = interpolate(x_corrs[j], index)
                curr_sum += val
            phase_sums[loc] = curr_sum
        

        curr_loc = max(phase_sums, key=phase_sums.get)
        # print(phase_diffs[curr_loc])
        # print(x_corrs.shape)

        for r in range(x_corrs.shape[0]):
            idx1 = int(phase_diffs[curr_loc][r])
            for i in range(-5, 5):
                # to_sub = min(1-scipy.stats.norm(0, 3).pdf(i),scipy.stats.norm(0, 3).pdf(i)) 
                if i == 0:
                    x_corrs[r][mid+idx1+i] = 0
                else:
                    x_corrs[r][mid+idx1+i] = 0

        
        mid = x_corrs.shape[1] // 2
        phase_sums = {}
        second_loc = None
        max_sum = 0
        for loc in phase_diffs:
            curr_sum = 0
            indices = phase_diffs[loc]
            for j in range(len(indices)):
                index = mid + indices[j]
                val = interpolate(x_corrs[j], index)
                curr_sum += val
            phase_sums[loc] = curr_sum
        
        second_loc = max(phase_sums, key=phase_sums.get)

        print(phase_sums[(-0.8575468969128732, -0.50974930362117, -0.06905770813482051)])
        print(phase_sums[(second_loc)])

        # for r in range(x_corrs.shape[0]):
        #     idx1 = int(phase_diffs[second_loc][r])
        #     idx2 = int(phase_diffs[second_loc][r]) + 1
        #     x_corrs[r][mid+idx1] = 0
        #     x_corrs[r][mid+idx2] = 0

        
        # mid = x_corrs.shape[1] // 2
        # phase_sums = {}
        # second_loc = None
        # max_sum = 0
        # for loc in phase_diffs:
        #     curr_sum = 0
        #     indices = phase_diffs[loc]
        #     for j in range(len(indices)):
        #         index = mid + indices[j]
        #         val = interpolate(x_corrs[j], index)
        #         curr_sum += val
        #     phase_sums[loc] = curr_sum
        
        # third_loc = max(phase_sums, key=phase_sums.get)







        print(curr_loc)
        print(second_loc)
        curr_locs.append(curr_loc)
        second_locs.append(second_loc)
        # third_locs.append(third_loc)
    return curr_locs, second_locs

if __name__ == "__main__":
    pts = fibonacci_sphere(360)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in pts:
        ax.scatter(i[0], i[1], i[2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()