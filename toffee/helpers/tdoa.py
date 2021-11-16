import numpy as np
from .hyperboloid import *
from .signals import *
import tqdm

def build_implicit_function(pos, data, start, end, reg, sampling_rate, mic_locs_list):
    f = np.array([0., 0., 0.])
    c = 343 
    phase_diffs = []
    
    count = 0

    # r = np.random.randint(0, data.shape[0] - 1)
    for i in range(0, data.shape[0]-1):
        for j in range(i+1, data.shape[0]):
           
            skip, phase_diff = find_delta(data[i, start:end].T, data[j, start:end].T)
            if skip == True:
                continue
            count += 1


            phase_diffs.append(phase_diff)
            delta = (c) * (phase_diff) / sampling_rate
            f += hyperboloid_gradient(pos[0],pos[1],pos[2],mic_locs_list[i],mic_locs_list[j],delta)
    # punish large values
    # print(phase_diffs)
    f += 0.001/np.linalg.norm(pos)
    return f, phase_diffs

def solve_implicit_function(data, frames, rate, mic_locs_list, alpha = 0.05, reg = 0.01):
    ests = []
    curr_guess = np.random.rand(3)
    for f in tqdm.tqdm(range(0, data.shape[1] - frames, frames)):
        grad, phase_diffs = build_implicit_function(curr_guess, data, f, f+frames, reg, rate, mic_locs_list)
        count = 0
        while np.linalg.norm(grad + reg * curr_guess) >= 0.08 and count <= 1000:
            grad, phase_diffs = build_implicit_function(curr_guess, data, f, f+frames, reg, rate,  mic_locs_list)
            curr_guess = curr_guess - alpha * (grad + curr_guess * reg)
            count += 1
            # print(np.linalg.norm(grad + reg * curr_guess))
           
        curr_guess = curr_guess
        # nromalize
        # curr_guess = curr_guess / np.linalg.norm(curr_guess)
        ests.append(curr_guess / np.linalg.norm(curr_guess))
        print(curr_guess)
        # map_plot(data, f, f+frames, rate, mic_locs_list, curr_guess)
    return ests


def plot_implicit(fn, mic1, mic2, delta, fig, ax, color, alpha, bbox=(-2,2)):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    A = np.linspace(xmin, xmax, 60) # resolution of the contour
    B = np.linspace(xmin, xmax, 60) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z,mic1,mic2,delta)
        
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', colors = color, alpha = alpha)

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z,mic1,mic2,delta)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', colors = color, alpha = alpha)
        
    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z,mic1,mic2,delta)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', colors = color, alpha = alpha) 

def map_plot(data, start, end, sampling_rate, mic_locs_list, curr_guess):
    c = 343 
    alpha = 0.2
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    count = 0


    for i in range(0,1): 
        for j in range(i+1, len(data)):
            skip, phase_diff = find_delta(data[i, start:end].T, data[j, start:end].T)
            if skip == True:
                continue
            count += 1


            delta = (c) * (phase_diff) / sampling_rate
            plot_implicit(hyperboloid, mic_locs_list[i], mic_locs_list[j], delta, fig, ax, "purple", alpha)
            
    for loc in mic_locs_list:
        ax.scatter(loc[0], loc[1], loc[2], c = "black")
    
    ax.set_zlim3d(-1,1)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)

    ax.scatter(curr_guess[0], curr_guess[1], curr_guess[2])

    plt.show()

